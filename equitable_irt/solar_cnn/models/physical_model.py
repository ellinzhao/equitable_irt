import torch
import torch.nn as nn
import torch.nn.functional as F


def get_encoder(in_channels=1):
    encoder = nn.Sequential(
        nn.Conv2d(in_channels, 64, 3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(True),
        nn.Dropout2d(p=0.2),
        nn.MaxPool2d(2, stride=1),
        nn.Conv2d(64, 16, 3, stride=1, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(True),
        nn.Dropout2d(p=0.2),
        nn.MaxPool2d(2, stride=1),
    )
    return encoder


class SolarClassifier:
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=16)

        self.fc1 = nn.Linear(16 * 16 * 16, 200)
        self.fc3 = nn.Linear(200, 2)

    def forward(self, x):
        x = (F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 16 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LightEstimator(nn.Module):
    """ Estimate lighting from normal, albedo and conv features
    """
    def __init__(self):
        super(LightEstimator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.AvgPool2d(16, stride=1, padding=0)
        self.fc = nn.Linear(16 * 47 * 47, 3)

    def forward(self, x):
        out = self.conv(x)
        out = self.pool(out)
        out = out.view(-1, 16 * 47 * 47)
        out = self.fc(out)
        return out


class SolarClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=16)

        self.fc1 = nn.Linear(16 * 16 * 16, 200)
        self.fc2 = nn.Linear(200, 2)

    def forward(self, x):
        x = (F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 16 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SLModel(nn.Module):
    def __init__(self):
        super(SLModel, self).__init__()
        self.temp_encoder = get_encoder(1)
        self.normal_encoder = get_encoder(3)
        self.light_estimator = LightEstimator()
        self.classifier = SolarClassifier()
        self.relu = nn.ReLU()

    def recon(self, t, sn, light):
        delta = torch.sum(sn * light[:, :, None, None], axis=1).unsqueeze(axis=1)
        delta = self.relu(delta)
        return t - delta, delta

    def forward(self, x, normal):
        temp_feat = self.temp_encoder(x)
        normal_feat = self.normal_encoder(normal)

        all_features = torch.cat((temp_feat, normal_feat), dim=1)
        pred_light = self.light_estimator(all_features)

        pred_tbase, pred_delta = self.recon(x, normal, pred_light)

        t_labels = self.classifier(x)
        tbase_labels = self.classifier(pred_tbase)

        return t_labels, tbase_labels, pred_light, pred_tbase, pred_delta
