import matplotlib.pyplot as plt


def plot_epoch_losses(train, val1, val2, save_path):
    fig = plt.figure(figsize=(1.5, 1), dpi=200)
    ax = fig.add_subplot(111)
    ax.plot(train, c='blue')
    ax2 = ax.twinx()
    ax2.plot(val1, c='red')
    if val2 is not None:
        ax2.plot(val2, c='green')
    fig.savefig(f'{save_path}/losses.png')


def plot_images(subject_data, save_path, cmap='gray'):
    n = len(subject_data)
    plt.figure(figsize=(5, n), dpi=150)
    for i, ims in enumerate(subject_data):
        est_base, est_temp, gt_base, gt_temp = ims
        ims = [est_base, gt_base, est_temp, gt_temp]
        vmaxes = [gt_base.max(), gt_base.max(), gt_temp.max(), gt_temp.max()]
        vmin = gt_base.min()
        for j, im in enumerate(ims):
            plt.subplot(n, 4, i * 4 + j + 1)
            plt.imshow(im, cmap=cmap, vmin=vmin, vmax=vmaxes[j])
            plt.axis(False)
            plt.colorbar()
    if save_path:
        plt.savefig(f'{save_path}/val.png')
        plt.close()
