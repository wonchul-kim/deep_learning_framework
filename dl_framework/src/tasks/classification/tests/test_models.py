from torchsummary import summary

from dl_framework.src.tasks.classification.src.models.from_torchvision.torchvision_model import TorchvisionModel



def test_torchvision_model():
    model_name = 'resnet18'
    in_channels = 3
    num_classes = 10

    model = TorchvisionModel(model_name, in_channels, num_classes).to('cuda')

    summary(model, (3, 224, 224))



