import models

MODEL_DISPATCHER = {
    'resnet18': models.ResNet18,
    'squeezenet': models.SqueezeNet,
    'efficientnet': models.EfficientNetB3
}
