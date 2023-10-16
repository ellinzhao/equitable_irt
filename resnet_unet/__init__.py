import torch

from .resnet_unet import ResNetUNet


model = ResNetUNet(n_class=3)
model.load_state_dict(torch.load('./resnet_unet/model.pth', map_location=torch.device('cpu')))
model.eval()

__all__ = ['model', 'ResNetUNet']
