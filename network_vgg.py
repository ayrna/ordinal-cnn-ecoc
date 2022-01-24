'''
Taken and modified from the official Torchvision project (v0.10.1)
under the BSD 3-Clause Revised License
https://github.com/pytorch/vision/blob/v0.10.1/torchvision/models/vgg.py

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


from functools import partial
from itertools import chain
from typing import Callable, List, Tuple, Union, cast

import numpy as np
import torch
import torch.nn as nn
import torchvision.models.vgg as vgg
from spacecutter.models import OrdinalLogisticModel

from network_common import ExperimentModel


__all__ = ['VGG', 'VGGOrdinalECOC', 'VGGOrdinalCLM', 'vggecoc_models', 'vggclm_models', 'vgg_models']

class VGG(ExperimentModel):
    classifier: nn.Sequential

    def __init__(
        self,
        features: nn.Module,
        num_classes: int,
        activation_function: Callable[[], nn.Module],
        init_weights: bool = True
    ) -> None:
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            features,
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(start_dim=1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            activation_function(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            activation_function(),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def regularized_parameters(self) -> List[nn.parameter.Parameter]:
        return list(chain(self.classifier[0].parameters(),
                          self.classifier[3].parameters()))

    def scores(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def on_batch_end(self):
        pass


class VGGOrdinalFullyConnected(nn.Module):
    classifiers: nn.ModuleList
    
    def __init__(self, input_size: int, num_classes: int, activation_function: Callable[[], nn.Module]):
        super(VGGOrdinalFullyConnected, self).__init__()
        hidden_size = 4096 // (num_classes - 1)
        self.classifiers = nn.ModuleList([nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation_function(),
            nn.Dropout(),
            nn.Linear(hidden_size, hidden_size),
            activation_function(),
            nn.Dropout(),
            nn.Linear(hidden_size, 1),
        ) for _ in range(num_classes-1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = [classifier(x) for classifier in self.classifiers]
        x = torch.cat(xs, dim=1)
        x = torch.sigmoid(x)
        return x


class VGGOrdinalECOC(ExperimentModel):
    classifier: VGGOrdinalFullyConnected
    target_class: torch.Tensor

    def __init__(
        self,
        features: nn.Module,
        num_classes: int,
        activation_function: Callable[[], nn.Module],
        init_weights: bool = True
    ) -> None:
        super(VGGOrdinalECOC, self).__init__()
        self.features = nn.Sequential(
            features,
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(start_dim=1)
        )
        self.classifier = VGGOrdinalFullyConnected(input_size=512 * 7 * 7,
                                                activation_function=activation_function,
                                                num_classes=num_classes)
        if init_weights:
            self._initialize_weights()

        # Reference vectors for each class, for predictions
        target_class = np.ones((num_classes, num_classes - 1), dtype=np.float32) 
        target_class[np.triu_indices(num_classes, 0, num_classes - 1)] = 0.0
        self.target_class = torch.tensor(target_class).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def regularized_parameters(self) -> List[nn.parameter.Parameter]:
        return list(chain.from_iterable(chain(cf[0].parameters(),  # type: ignore
                                              cf[3].parameters())  # type: ignore
                                        for cf in self.classifier.classifiers))

    def scores(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward(x)
        return -torch.cdist(x, self.target_class.to(x.device))

    def on_batch_end(self):
        pass


class VGGOrdinalCLM(ExperimentModel):
    ordinal_classifier: OrdinalLogisticModel

    def __init__(self, features: nn.Module, num_classes: int,
                 activation_function: Callable[[], nn.Module], init_weights: bool = True,
                 margin: float = 0.0, min_val: float = -1.0e6):
        super(VGGOrdinalCLM, self).__init__()
        vgg = VGG(features, num_classes=1, activation_function=activation_function, init_weights=init_weights)
        self.ordinal_classifier = OrdinalLogisticModel(vgg, num_classes)
        self.margin = margin
        self.min_val = min_val

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

    def convolutional_part(self) -> nn.Module:
        return self.ordinal_classifier.predictor.convolutional_part()

    def fully_connected_part(self) -> nn.Module:
        return nn.Sequential(self.ordinal_classifier.predictor.fully_connected_part(),
                             self.ordinal_classifier.link)


def make_layers(cfg: List[Union[str, int]], activation_function: Callable[[], nn.Module], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), activation_function()]
            else:
                layers += [conv2d, activation_function()]
            in_channels = v
    return nn.Sequential(*layers)


def _vggecoc(cfg_letter: str, activation_function: Callable[[], nn.Module], batch_norm: bool = False, **kwargs):
    return VGGOrdinalECOC(make_layers(vgg.cfgs[cfg_letter], activation_function, batch_norm),
                          activation_function=activation_function, **kwargs)


def _vggclm(cfg_letter: str, activation_function: Callable[[], nn.Module], batch_norm: bool = False, **kwargs):
    return VGGOrdinalCLM(make_layers(vgg.cfgs[cfg_letter], activation_function, batch_norm),
                         activation_function=activation_function, **kwargs)


def _vgg(cfg_letter: str, activation_function: Callable[[], nn.Module], batch_norm: bool = False, **kwargs):
    return VGG(make_layers(vgg.cfgs[cfg_letter], activation_function, batch_norm),
               activation_function=activation_function, **kwargs)


vggecoc_models = {
    'vgg11': partial(_vggecoc, cfg_letter='A'),
    'vgg13': partial(_vggecoc, cfg_letter='B'),
    'vgg16': partial(_vggecoc, cfg_letter='D'),
    'vgg19': partial(_vggecoc, cfg_letter='E'),
}

vggclm_models = {
    'vgg11': partial(_vggclm, cfg_letter='A'),
    'vgg13': partial(_vggclm, cfg_letter='B'),
    'vgg16': partial(_vggclm, cfg_letter='D'),
    'vgg19': partial(_vggclm, cfg_letter='E'),
}

vgg_models = {
    'vgg11': partial(_vgg, cfg_letter='A'),
    'vgg13': partial(_vgg, cfg_letter='B'),
    'vgg16': partial(_vgg, cfg_letter='D'),
    'vgg19': partial(_vgg, cfg_letter='E'),
}

