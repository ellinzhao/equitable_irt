import os

import torch

from .resnet_unet import ResNetUNet


path = os.path.join(os.getcwd(), 'equitable_irt',
                    'dataset_generation', 'sn_estimation', 'model.pth')
try:
    SN_MODEL = ResNetUNet(n_class=3)
    SN_MODEL.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    SN_MODEL.eval()
except FileNotFoundError:
    SN_MODEL = None
    print('Surface normals model not loaded')

__all__ = ['SN_MODEL', 'ResNetUNet']
