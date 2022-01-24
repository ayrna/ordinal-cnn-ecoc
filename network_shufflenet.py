'''
Taken and modified from the official Torchvision project (v0.10.1)
under the BSD 3-Clause Revised License
https://github.com/pytorch/vision/blob/v0.10.1/torchvision/models/shufflenetv2.py

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

from typing import Callable, List, Tuple
from functools import partial

import torch
import torch.nn as nn
from torch import Tensor
from spacecutter.models import OrdinalLogisticModel
import numpy as np

from network_common import OrdinalFullyConnected, ExperimentModel

__all__ = ['shufflenet_models', 'shufflenetecoc_models', 'shufflenetclm_models', 'ShuffleNetV2', 'ShuffleNetOrdinalCLM']


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        activation_function: Callable[[], nn.Module]
    ) -> None:
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                activation_function()
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            activation_function(),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            activation_function()
        )

    @staticmethod
    def depthwise_conv(
        i: int,
        o: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False
    ) -> nn.Conv2d:
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(ExperimentModel):
    def __init__(
        self,
        stages_repeats: List[int],
        stages_out_channels: List[int],
        activation_function: Callable[[], nn.Module] = partial(nn.ReLU, inplace=True),
        num_classes: int = 1000,
        classifier: Callable[[int, int], nn.Module] = nn.Linear,
        inverted_residual: Callable[..., nn.Module] = InvertedResidual
    ) -> None:
        super(ShuffleNetV2, self).__init__()

        inverted_residual = partial(inverted_residual, activation_function=activation_function)

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            activation_function()
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Static annotations for mypy
        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential
        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            activation_function()
        )

        self.fc = classifier(output_channels, num_classes)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # globalpool
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def regularized_parameters(self) -> List[nn.parameter.Parameter]:
        return []

    def scores(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def on_batch_end(self):
        pass


class ShuffleNetOrdinalECOC(ShuffleNetV2):
    target_class: torch.Tensor

    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
        if 'classifier' in kwargs:
            raise TypeError("Cannot specify classifier for OBD classifier")
        kwargs['classifier'] = OrdinalFullyConnected
        super(ShuffleNetOrdinalECOC, self).__init__(*args, **kwargs)

        num_classes = kwargs.get('num_classes', 1000)

        # Reference vectors for each class, for predictions
        target_class = np.ones((num_classes, num_classes - 1), dtype=np.float32) 
        target_class[np.triu_indices(num_classes, 0, num_classes - 1)] = 0.0
        self.target_class = torch.tensor(target_class).float()

    def scores(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward(x)
        return -torch.cdist(x, self.target_class.to(x.device))

class ShuffleNetOrdinalCLM(ExperimentModel):
    ordinal_classifier: OrdinalLogisticModel

    def __init__(self, shufflenet: ShuffleNetV2, num_classes: int,
                 margin: float = 0.0, min_val: float = -1.0e6):
        super(ShuffleNetOrdinalCLM, self).__init__()
        self.ordinal_classifier = OrdinalLogisticModel(shufflenet, num_classes)
        self.margin = margin
        self.min_val = torch.tensor(min_val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ordinal_classifier(x)

    def scores(self, x: torch.Tensor) -> torch.Tensor:
        # No need to implement scores because it is only
        # used by self.predict, which we are overriding
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

    def convolutional_part(self) -> nn.Module:
        return self.ordinal_classifier.predictor.convolutional_part()

    def fully_connected_part(self) -> nn.Module:
        return nn.Sequential(self.ordinal_classifier.predictor.fully_connected_part(),
                             self.ordinal_classifier.link)


cfgs = {
    "x0_5": ([4, 8, 4], [24, 48, 96, 192, 1024]),
    "x1_0": ([4, 8, 4], [24, 116, 232, 464, 1024]),
    "x1_5": ([4, 8, 4], [24, 176, 352, 704, 1024]),
    "x2_0": ([4, 8, 4], [24, 244, 488, 976, 2048]),
}

def _shufflenetecoc(cfg: str, **kwargs) -> ShuffleNetV2:
    return ShuffleNetOrdinalECOC(*cfgs[cfg], **kwargs)


def _shufflenetclm(cfg: str, **kwargs) -> ShuffleNetOrdinalCLM:
    num_classes = kwargs.pop('num_classes')
    shufflenet = ShuffleNetV2(*cfgs[cfg],
                        num_classes=1,
                        **kwargs)
    return ShuffleNetOrdinalCLM(shufflenet, num_classes)


def _shufflenet(cfg: str, **kwargs) -> ShuffleNetV2:
    return ShuffleNetV2(*cfgs[cfg],
                        **kwargs)

shufflenet_models = {
    'shufflenet_x0_5': partial(_shufflenet, cfg='x0_5'),
    'shufflenet_x1_0': partial(_shufflenet, cfg='x1_0'),
    'shufflenet_x1_5': partial(_shufflenet, cfg='x1_5'),
    'shufflenet_x2_0': partial(_shufflenet, cfg='x2_0'),
}

shufflenetecoc_models = {
    'shufflenet_x0_5': partial(_shufflenetecoc, cfg='x0_5'),
    'shufflenet_x1_0': partial(_shufflenetecoc, cfg='x1_0'),
    'shufflenet_x1_5': partial(_shufflenetecoc, cfg='x1_5'),
    'shufflenet_x2_0': partial(_shufflenetecoc, cfg='x2_0'),
}

shufflenetclm_models = {
    'shufflenet_x0_5': partial(_shufflenetclm, cfg='x0_5'),
    'shufflenet_x1_0': partial(_shufflenetclm, cfg='x1_0'),
    'shufflenet_x1_5': partial(_shufflenetclm, cfg='x1_5'),
    'shufflenet_x2_0': partial(_shufflenetclm, cfg='x2_0'),
}