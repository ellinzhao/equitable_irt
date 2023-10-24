import torch
from torch import nn


class FaceFeatureLoss(nn.Module):

    def __init__(self):
        super(FaceFeatureLoss, self).__init__()

    def forward(self, est_base, mask, base_rois):
        # est_base:  b, 1, h, w
        # mask:      b, 10, h, w
        # base_rois: b, 10
        criterion = nn.MSELoss()
        est_rois = torch.sum(est_base * mask, axis=(-2, -1))
        est_rois = est_rois / torch.sum(mask, axis=(-2, -1))
        loss = criterion(est_rois, base_rois)

        # Should there be a loss involving Tskin or is_base
        return loss
