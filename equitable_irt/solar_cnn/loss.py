import torch
from torch import nn


class FaceFeatureLoss(nn.Module):

    def __init__(self):
        super(FaceFeatureLoss, self).__init__()
        self.criterion1 = nn.MSELoss()
        self.criterion2 = nn.MSELoss()

    def extract_rois(self, est_base, mask):
        est_rois = torch.sum(est_base * mask, axis=(-2, -1))
        est_rois = est_rois / torch.sum(mask, axis=(-2, -1))
        return est_rois

    def forward(self, est_base, mask, base_rois, tskin):
        # est_base:  b, 1, h, w
        # mask:      b, 10, h, w
        # base_rois: b, 10
        base_rois = base_rois.squeeze(-1)
        base_rois = base_rois.squeeze(1)

        est_rois = self.extract_rois(est_base, mask)
        loss = self.criterion1(est_rois.float(), base_rois)

        # Should there be a loss involving Tskin or is_base
        return loss
