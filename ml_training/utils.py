import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.nn import init

from dataset import TRAIN_TFORM
from dataset import VAL_TFORM
from dataset import SolarDataset
from expt_config import ExptConfig
from dataset import UNSCALE_FN


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


def read_configs_csv(gc, url):
    gsheets = gc.open_by_url(url)
    sheets = gsheets.worksheet('Sheet1').get_all_values()
    df = pd.DataFrame(sheets[1:], columns=sheets[0])
    df = df.astype({'lr': 'float',
                    'batch_size': 'int',
                    'lpips_weight': 'float',
                    'mae_temp_weight': 'float',
                    'fever_prob': 'float',
                    'epochs': 'int',
                    'grad_clip': 'float',
                    'model': 'str',
                    'sl_weight': 'float'})
    return df


def df_to_config(row):
    p = row.to_dict()
    folder = p.pop('folder')
    c = ExptConfig(**p)
    return folder, c


def load_split_csv(dataset_dir, split_type):
    assert split_type in ['train', 'test', 'val']
    path = os.path.join(dataset_dir, f'{split_type}.csv')
    return pd.read_csv(path)['0'].to_numpy()


def get_fold(folds_path, i, dataset_dir):
    if i == 'test':
        train = list(load_split_csv(dataset_dir, 'train'))
        train += list(load_split_csv(dataset_dir, 'val'))
        test = list(load_split_csv(dataset_dir, 'test'))
    else:
        path = os.path.join(f'{folds_path}/fold{i}', 'train.csv')
        train = list(pd.read_csv(path)['0'])
        path = os.path.join(f'{folds_path}/fold{i}', 'val.csv')
        test = list(pd.read_csv(path)['0'])
    filter_data = lambda lst: [s for s in lst if 'derek' not in s and 'howard' not in s]
    train = filter_data(train)
    test = filter_data(test)
    return train, test


def get_dataloader(sids, config, dataset_dir, dataset_dict, train=True, **kwargs):
    tform = TRAIN_TFORM if train else VAL_TFORM
    ds = SolarDataset(
        dataset_dir, dataset_dict, sids,
        train=train, transform=tform,
        fever_prob=config.fever_prob,
        **kwargs,
    )
    # Shuffle data for training only
    dl = DataLoader(ds, batch_size=config.batch_size, shuffle=train)
    return dl


def get_training_data(fold_num, config, dataset_dir, dataset_dict, **kwargs):
    train_sids, val_sids = get_fold('./folds_0419', fold_num, dataset_dir)
    train_dl = get_dataloader(train_sids, config, dataset_dir, dataset_dict, train=True, **kwargs)
    val_dl = get_dataloader(val_sids, config, dataset_dir, dataset_dict, train=False, **kwargs)
    return train_dl, val_dl


def get_subject_recon(sid, dataset_dir, dataset_dict, model, i=0.5, scale=True, **kwargs):
    model.eval()
    ds = SolarDataset(
        dataset_dir, dataset_dict, [sid],
        transform=VAL_TFORM, train=False,
        session_filter='cool', fever_prob=0,
        **kwargs,
    )
    device = next(model.parameters()).device
    i = int(i * len(ds))
    images, _, _ = ds[i]
    images = images.unsqueeze(0).to(device)
    temp = images[:, 0:1]
    base = images[:, 1:2]

    with torch.no_grad():
        est = model(temp)
    base = base.squeeze(0)
    temp = temp.squeeze(0)
    est = est.squeeze(0)
    out = [est[0], est[0] + est[1], base[0], temp[0]]
    out = [x.detach().cpu() for x in out]
    if scale:
        out = [UNSCALE_FN(x) for x in out]
    return out


def dataset_recons(sids, dataset_dir, dataset_dict, model):
    all_data = []
    for sid in sids:
        ims = get_subject_recon(sid, dataset_dir, dataset_dict, model, i=0.3)
        all_data += [ims]
    return all_data
