'''
Taken and modified from the official Torchvision project (v0.10.1)
under the BSD 3-Clause Revised License
https://github.com/pytorch/vision/blob/v0.10.1/torchvision/models/mobilenetv3.py

BSD 3-Clause License

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

from typing import Callable, List, Optional, Any, Sequence, Tuple
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models.mobilenetv2 import _make_divisible, ConvBNActivation
from spacecutter.models import OrdinalLogisticModel
import numpy as np

from network_common import ExperimentModel

__all__ = ['mobilenet_models', 'mobilenetecoc_models', 'mobilenetclm_models', 'MobileNetV3']


def mobilenet_classifier(lastconv_output_channels: int, last_channel: int, num_classes: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(lastconv_output_channels, last_channel),
        nn.Hardswish(inplace=True),
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(last_channel, num_classes),
    )


class MobileNetOrdinalFullyConnected(nn.Module):
    classifiers: nn.ModuleList
    
    def __init__(self, lastconv_output_channels: int, last_channel: int, num_classes: int):
        super().__init__()
        self.classifiers = nn.ModuleList(
            [mobilenet_classifier(lastconv_output_channels, last_channel // (num_classes - 1), 1) for _ in range(num_classes-1)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = [classifier(x) for classifier in self.classifiers]
        x = torch.cat(xs, dim=1)
        x = torch.sigmoid(x)
        return x



class SqueezeExcitation(nn.Module):
    # Implemented as described at Figure 4 of the MobileNetV3 paper
    def __init__(self, input_channels: int, squeeze_factor: int = 4):
        super().__init__()
        squeeze_channels = _make_divisible(input_channels // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)

    def _scale(self, input: Tensor, inplace: bool) -> Tensor:
        scale = F.adaptive_avg_pool2d(input, 1)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        return F.hardsigmoid(scale, inplace=inplace)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input, True)
        return scale * input


class InvertedResidualConfig:
    # Stores information listed at Tables 1 and 2 of the MobileNetV3 paper
    def __init__(self, input_channels: int, kernel: int, expanded_channels: int, out_channels: int, use_se: bool,
                 activation: str, stride: int, dilation: int, width_mult: float):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)


class InvertedResidual(nn.Module):
    # Implemented as described at section 5 of MobileNetV3 paper
    def __init__(self, cnf: InvertedResidualConfig, norm_layer: Callable[..., nn.Module],
                 se_layer: Callable[..., nn.Module] = SqueezeExcitation):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError('illegal stride value')

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(ConvBNActivation(cnf.input_channels, cnf.expanded_channels, kernel_size=1,
                                           norm_layer=norm_layer, activation_layer=activation_layer))

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(ConvBNActivation(cnf.expanded_channels, cnf.expanded_channels, kernel_size=cnf.kernel,
                                       stride=stride, dilation=cnf.dilation, groups=cnf.expanded_channels,
                                       norm_layer=norm_layer, activation_layer=activation_layer))
        if cnf.use_se:
            layers.append(se_layer(cnf.expanded_channels))

        # project
        layers.append(ConvBNActivation(cnf.expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer,
                                       activation_layer=nn.Identity))

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return result


class MobileNetV3(ExperimentModel):

    def __init__(
            self,
            inverted_residual_setting: List[InvertedResidualConfig],
            last_channel: int,
            activation_function: Callable[[], nn.Module],
            classifier: Callable[[int, int, int], nn.Module] = mobilenet_classifier,
            num_classes: int = 1000,
            block: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            **kwargs: Any
    ) -> None:
        """
        MobileNet V3 main class

        Args:
            inverted_residual_setting (List[InvertedResidualConfig]): Network structure
            last_channel (int): The number of channels on the penultimate layer
            num_classes (int): Number of classes
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for mobilenet
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
        """
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (isinstance(inverted_residual_setting, Sequence) and
                  all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(ConvBNActivation(3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(ConvBNActivation(lastconv_input_channels, lastconv_output_channels, kernel_size=1,
                                       norm_layer=norm_layer, activation_layer=nn.Hardswish))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = classifier(lastconv_output_channels, last_channel, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def regularized_parameters(self) -> List[nn.parameter.Parameter]:
        return list(self.parameters())

    def on_batch_end(self):
        pass

    def scores(self, x: Tensor) -> Tensor:
        return self.forward(x)


class MobileNetOrdinalECOC(MobileNetV3):
    target_class: torch.Tensor

    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
        if 'classifier' in kwargs:
            raise TypeError("Cannot specify classifier for OBD classifier")
        kwargs['classifier'] = MobileNetOrdinalFullyConnected
        super().__init__(*args, **kwargs)

        num_classes = kwargs.get('num_classes', 1000)

        # Reference vectors for each class, for predictions
        target_class = np.ones((num_classes, num_classes - 1), dtype=np.float32) 
        target_class[np.triu_indices(num_classes, 0, num_classes - 1)] = 0.0
        self.target_class = torch.tensor(target_class).float()

    def scores(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward(x)
        return -torch.cdist(x, self.target_class.to(x.device))

class MobileNetOrdinalCLM(ExperimentModel):
    ordinal_classifier: OrdinalLogisticModel

    def __init__(self, mobilenet: MobileNetV3, num_classes: int,
                 margin: float = 0.0, min_val: float = -1.0e6):
        super().__init__()
        self.ordinal_classifier = OrdinalLogisticModel(mobilenet, num_classes)
        self.margin = margin
        self.min_val = torch.tensor(min_val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ordinal_classifier(x)

    def scores(self, x: torch.Tensor) -> torch.Tensor:
        # No need to implement scores because it is only
        # used by VGG.predict, which we are overriding
        raise NotImplementedError

    def predict(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        self.eval()
        probas = self.forward(x)
        labels = probas.argmax(dim=1)
        labels, probas = map(lambda t: t.detach().cpu().numpy(), (labels, probas))
        return labels, probas

    def regularized_parameters(self) -> List[nn.parameter.Parameter]:
        return self.ordinal_classifier.predictor.regularized_parameters()

    def clip(self) -> None:
        cutpoints = self.ordinal_classifier.link.cutpoints.data
        for i in range(cutpoints.shape[0] - 1):
            cutpoints[i].clamp_(self.min_val,
                                cutpoints[i + 1] - self.margin)

    def on_batch_end(self) -> None:
        self.clip()


def _mobilenet_v3_conf(arch: str, width_mult: float = 1.0, reduced_tail: bool = False, dilated: bool = False,
                       **kwargs: Any):
    reduce_divider = 2 if reduced_tail else 1
    dilation = 2 if dilated else 1

    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)

    if arch == "mobilenet_v3_large":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
            bneck_conf(16, 3, 64, 24, False, "RE", 2, 1),  # C1
            bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 72, 40, True, "RE", 2, 1),  # C2
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 3, 240, 80, False, "HS", 2, 1),  # C3
            bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
            bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
            bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2, dilation),  # C4
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = adjust_channels(1280 // reduce_divider)  # C5
    elif arch == "mobilenet_v3_small":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),  # C1
            bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),  # C2
            bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),  # C3
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation),  # C4
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = adjust_channels(1024 // reduce_divider)  # C5
    else:
        raise ValueError("Unsupported model type {}".format(arch))

    return inverted_residual_setting, last_channel


def _mobilenet(arch: str, **kwargs: Any) -> MobileNetV3:
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(arch, **kwargs)
    return MobileNetV3(inverted_residual_setting, last_channel, **kwargs)


def _mobilenetecoc(arch: str, **kwargs) -> MobileNetOrdinalECOC:
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(arch, **kwargs)
    return MobileNetOrdinalECOC(inverted_residual_setting, last_channel, **kwargs)


def _mobilenetclm(arch: str, **kwargs) -> MobileNetOrdinalCLM:
    num_classes = kwargs.pop('num_classes')
    mobilenet = _mobilenet(arch, num_classes=1, **kwargs)
    return MobileNetOrdinalCLM(mobilenet, num_classes)


mobilenet_models = {
    'mobilenet_v3_large': partial(_mobilenet, arch='mobilenet_v3_large'),
    'mobilenet_v3_small': partial(_mobilenet, arch='mobilenet_v3_small'),
}

mobilenetecoc_models = {
    'mobilenet_v3_large': partial(_mobilenetecoc, arch='mobilenet_v3_large'),
    'mobilenet_v3_small': partial(_mobilenetecoc, arch='mobilenet_v3_small'),
}

mobilenetclm_models = {
    'mobilenet_v3_large': partial(_mobilenetclm, arch='mobilenet_v3_large'),
    'mobilenet_v3_small': partial(_mobilenetclm, arch='mobilenet_v3_small'),
}
