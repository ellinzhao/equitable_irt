import lpips
import torch
from torch import nn
from dataset import BG_THRES_C
from dataset import MEAN_C, STD_C
from dataset import TEMP_VMAX_C, TEMP_VMIN_C
from dataset import UNSCALE_FN 


device = torch.device('cuda')
LPIPS_FN = lpips.LPIPS(net='alex', spatial=False).to(device)
MSE_FN = nn.MSELoss()
MAE_FN = nn.L1Loss()


def normalize_im(a):
    vmin = (TEMP_VMIN_C - MEAN_C[0]) / STD_C[0]
    vmax = (TEMP_VMAX_C - MEAN_C[0]) / STD_C[0]
    a = a - vmin
    a = a / (vmax - vmin)
    a = a * 2 - 1
    return a


def lpips_loss(est, gt, mask, weight=1, avg=True):
    # LPIPS takes in 3-channel images with values in range (-1, 1)
    est = est * weight
    gt = gt * weight
    est = normalize_im(est)
    gt = normalize_im(gt)
    out = LPIPS_FN(est.expand(-1, 3, -1, -1), gt.expand(-1, 3, -1, -1))
    if avg:
        return out.mean()
    else:
        return out


def foreground_mask(x1, x2=None):
    mask = x1 > BG_THRES_C
    if x2 is not None:
        mask = mask & (x2 > BG_THRES_C)
    return mask


def masked_loss(est, gt, mask=None, weight=1, fn=MAE_FN):
    est = est * weight
    gt = gt * weight
    if mask is not None:
        loss = fn(est[mask.bool()], gt[mask.bool()])
    else:
        loss = fn(est, gt)
    return loss


def loss_fn(est, gt_base, gt_temp, weight=1, lpips_weight=1, mae_temp_weight=1, epoch=0, print_val=True):
    est_base = est[:, 0:1]
    est_temp = est[:, 0:1] + est[:, 1:2]
    mask = foreground_mask(gt_base, gt_temp)

    # Slight optimization, don't calculate LPIPS if it won't be used for backprop
    if lpips_weight == 0:
        loss_lpips = torch.tensor([0]).to(est_base.device)
    else:
        loss_lpips = lpips_weight * lpips_loss(est_base, gt_base, mask)

    loss_base = masked_loss(est_base, gt_base, None, weight)
    loss_temp = mae_temp_weight * masked_loss(est_temp, gt_temp, None, weight)

    # print(loss_lpips.item(), loss_base.item(), loss_temp.item())
    return loss_lpips + loss_base + loss_temp


def val_loss_fn(est, gt_base, gt_temp):
    est_base = est[:, 0:1]
    est_base = UNSCALE_FN(est_base)
    gt_base = UNSCALE_FN(gt_base)
    loss = masked_loss(est_base, gt_base, None)
    return loss


def get_forehead_values(im, gt_base, gt_temp):
    mask = foreground_mask(gt_temp, gt_base)
    crop_fn = lambda x: x[:, :, 2:7, 13:-13]
    masked_mean = lambda x, m: (x * m).sum(dim=(-2, -1)) / m.sum(dim=(-2, -1))

    est = crop_fn(im)
    mask = crop_fn(mask)
    est = UNSCALE_FN(est)
    return masked_mean(est, mask)


def forehead_loss_fn(est, gt_base, gt_temp, labels=1):
    # labels is 0 when classifier estimates solar loading
    est_base = est[:, 0:1]
    est = get_forehead_values(est_base * labels, gt_temp, gt_base)
    gt = get_forehead_values(gt_base, gt_temp, gt_base)
    return MAE_FN(est, gt)


def calc_dl_loss(model, dl, device, fn=val_loss_fn):
    count = 0
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for input in dl:
            images, label, _ = input
            images = images.to(device)
            label = label.to(device)
            temp = images[:, 0:1]
            base = images[:, 1:2]
            est = model(temp)

            loss = fn(est, base, temp)
            total_loss += loss.item()
            count += 1
    model.train()
    return total_loss / count
