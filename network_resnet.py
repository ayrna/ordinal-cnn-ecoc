'''
Taken and modified from the official Torchvision project (v0.10.1)
under the BSD 3-Clause Revised License
https://github.com/pytorch/vision/blob/v0.10.1/torchvision/models/resnet.py

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

from spacecutter.models import OrdinalLogisticModel
import torch
from torch import Tensor
import torch.nn as nn
from typing import Tuple, Type, Callable, Union, List, Optional
from functools import partial
from network_common import ExperimentModel, OrdinalFullyConnected
import numpy as np


__all__ = ['resnet_models', 'resnetecoc_models', 'resnetclm_models', 'ResNet', 'ResNetOrdinalCLM']



def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        activation_function: Callable[[], nn.Module],
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.activation = activation_function()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        activation_function: Callable[[], nn.Module],
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.activation = activation_function()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out


class ResNet(ExperimentModel):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        *,
        activation_function: Callable[[], nn.Module],
        classifier: Callable[[int, int], nn.Module] = nn.Linear,
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.activation = activation_function()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], activation_function)
        self.layer2 = self._make_layer(block, 128, layers[1], activation_function, stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], activation_function, stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], activation_function, stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = classifier(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

        self.features = nn.Sequential(
            self.conv1,
            self.bn1,
            self.activation,
            self.maxpool,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.avgpool,
            nn.Flatten(start_dim=1)
        )

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    activation_function, stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, activation_function, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, activation_function, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.features(x)
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


class ResNetOrdinalECOC(ResNet):
    target_class: torch.Tensor

    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
        if 'classifier' in kwargs:
            raise TypeError("Cannot specify classifier for OBD classifier")
        kwargs['classifier'] = OrdinalFullyConnected
        super(ResNetOrdinalECOC, self).__init__(*args, **kwargs)

        num_classes = kwargs.get('num_classes', 1000)

        # Reference vectors for each class, for predictions
        target_class = np.ones((num_classes, num_classes - 1), dtype=np.float32) 
        target_class[np.triu_indices(num_classes, 0, num_classes - 1)] = 0.0
        self.target_class = torch.tensor(target_class).float()

    def scores(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward(x)
        return -torch.cdist(x, self.target_class.to(x.device))


class ResNetOrdinalCLM(ExperimentModel):
    ordinal_classifier: OrdinalLogisticModel

    def __init__(self, resnet: ResNet, num_classes: int,
                 margin: float = 0.0, min_val: float = -1.0e6):
        super(ResNetOrdinalCLM, self).__init__()
        self.ordinal_classifier = OrdinalLogisticModel(resnet, num_classes)
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


cfgs = {
    '18': (BasicBlock, [2, 2, 2, 2]),
    '34': (BasicBlock, [3, 4, 6, 3]),
    '50': (Bottleneck, [3, 4, 6, 3]),
    '101': (Bottleneck, [3, 4, 23, 3]),
    '152': (Bottleneck, [3, 8, 36, 3]),
}

def _resnetecoc(cfg: str, **kwargs) -> ResNet:
    return ResNetOrdinalECOC(*cfgs[cfg], **kwargs)


def _resnetclm(cfg: str, **kwargs) -> ResNetOrdinalCLM:
    num_classes = kwargs.pop('num_classes')
    resnet = ResNet(*cfgs[cfg],
                    num_classes=1,
                    **kwargs)
    return ResNetOrdinalCLM(resnet, num_classes)


def _resnet(cfg: str, **kwargs) -> ResNet:
    return ResNet(*cfgs[cfg], **kwargs)


resnet_models = {
    'resnet18': partial(_resnet, cfg='18'),
    'resnet34': partial(_resnet, cfg='34'),
    'resnet50': partial(_resnet, cfg='50'),
    'resnet101': partial(_resnet, cfg='101'),
    'resnet152': partial(_resnet, cfg='152'),
}

resnetecoc_models = {
    'resnet18': partial(_resnetecoc, cfg='18'),
    'resnet34': partial(_resnetecoc, cfg='34'),
    'resnet50': partial(_resnetecoc, cfg='50'),
    'resnet101': partial(_resnetecoc, cfg='101'),
    'resnet152': partial(_resnetecoc, cfg='152'),
}

resnetclm_models = {
    'resnet18': partial(_resnetclm, cfg='18'),
    'resnet34': partial(_resnetclm, cfg='34'),
    'resnet50': partial(_resnetclm, cfg='50'),
    'resnet101': partial(_resnetclm, cfg='101'),
    'resnet152': partial(_resnetclm, cfg='152'),
}