import torch
import torch.nn as nn
import torch.nn.functional as F


class TNetLite(nn.Module):
    """
    T-Net without BatchNorm (safe for batch_size=1).
    """
    def __init__(self, k=7):
        super().__init__()
        self.k = k

        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

    def forward(self, x):
        B = x.size(0)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = torch.max(x, 2)[0]  # (B, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Add identity
        id_mat = torch.eye(self.k, device=x.device).flatten().unsqueeze(0).repeat(B, 1)
        x = x + id_mat

        return x.view(-1, self.k, self.k)


class PointNetSegLite(nn.Module):
    """
    BatchNorm-free PointNetSegLite for large LiDAR scans (N x 7).
    Works with CPU + batch_size=1.
    """
    def __init__(self, num_classes, input_dim=7):
        super().__init__()
        self.k = input_dim

        self.tnet = TNetLite(k=input_dim)

        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.conv4 = nn.Conv1d(1088, 512, 1)
        self.conv5 = nn.Conv1d(512, 256, 1)
        self.conv6 = nn.Conv1d(256, 128, 1)
        self.conv7 = nn.Conv1d(128, num_classes, 1)

    def forward(self, x):
        # x: (B, N, k)
        x = x.transpose(2, 1)  # -> (B, k, N)

        trans = self.tnet(x)
        x = torch.bmm(trans, x)

        x = F.relu(self.conv1(x))
        local_feat = x

        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        global_feat = torch.max(x, 2, keepdim=True)[0]
        global_feat = global_feat.repeat(1, 1, local_feat.size(2))

        x = torch.cat([local_feat, global_feat], dim=1)

        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.conv7(x)

        return x  # (B, num_classes, N)
