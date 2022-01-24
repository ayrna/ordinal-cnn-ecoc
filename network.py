from network_vgg import vgg_models, vggecoc_models, vggclm_models
from network_resnet import resnet_models, resnetecoc_models, resnetclm_models
from network_shufflenet import shufflenet_models, shufflenetecoc_models, shufflenetclm_models
from network_mobilenet import mobilenet_models, mobilenetecoc_models, mobilenetclm_models


__all__ = ['nominal_models', 'ecoc_models', 'clm_models']

nominal_models = vgg_models | resnet_models | shufflenet_models | mobilenet_models
ecoc_models = vggecoc_models | resnetecoc_models | shufflenetecoc_models | mobilenetecoc_models
clm_models = vggclm_models | resnetclm_models | shufflenetclm_models | mobilenetclm_models