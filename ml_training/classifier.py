import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, in_ch=64):
        super(Classifier, self).__init__()

        self.mlp = nn.Sequential(
            nn.Flatten(1, 3),
            nn.Linear(in_ch * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        return self.mlp(x)
