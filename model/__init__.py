import model.cifar10
import model.vgg16

__all__ = ["cifar10", "vgg16"]

MODELS = {
    "CIFAR10": model.cifar10.CIFAR10,
    "SimpleVGG16": model.vgg16.SimpleVGG16
}