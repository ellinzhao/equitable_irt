import torch


# https://github.com/RutvikB/Image-Reconstruction-using-Convolutional-Autoencoders-and-PyTorch/blob/main/Conv_AE_Pytorch.py


class ConvAutoencoder(torch.nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.MaxPool2d(2, stride=1),
            torch.nn.Conv2d(64, 16, 3, stride=1, padding=1),  # b, 8, 3, 3
            torch.nn.ReLU(True),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=1, mode='nearest'),
            torch.nn.Conv2d(16, 64, 3, stride=1, padding=1),  # b, 16, 10, 10
            torch.nn.ReLU(True),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.Upsample(scale_factor=1, mode='nearest'),
            torch.nn.Conv2d(64, 1, 3, stride=1, padding=2),  # b, 8, 3, 3
            torch.nn.Sigmoid()
        )

    def forward(self, x, normals, lvec):
        coded = self.encoder(x)
        decoded = self.decoder(coded)

        delta = torch.sum(normals * lvec, axis=1).unsqueeze(axis=1)
        torch.nn.ReLU(inplace=True)(delta)

        return coded, decoded, delta
