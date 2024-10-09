import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self, game, num_resBlocks, num_hidden, num_features, device):
        super().__init__()

        self.device = device
        self.startBlock = nn.Sequential(
            nn.Conv1d(7, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU()
        )

        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )

        # Policy head
        self.policyConv = nn.Conv1d(
            num_hidden, num_hidden, kernel_size=3, padding=1)
        self.policyBN = nn.BatchNorm1d(num_hidden)
        self.policyFlatten = nn.Flatten()
        self.policyLinear = nn.Sequential(
            nn.Linear(num_hidden * game.idx_count + num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, game.action_size)
        )

        # Value head
        self.valueConv = nn.Conv1d(num_hidden, 2, kernel_size=3, padding=1)
        self.valueBN = nn.BatchNorm1d(2)
        self.valueFlatten = nn.Flatten()
        self.valueLinear = nn.Sequential(
            nn.Linear(2 * game.idx_count + num_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

        self.to(device)
    # @timed

    def forward(self, x, features):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)

        # Policy head
        policy = self.policyConv(x)
        policy = self.policyBN(policy)
        policy = F.relu(policy)
        policy = self.policyFlatten(policy)
        policy = torch.cat([policy, features], dim=1)
        policy = self.policyLinear(policy)

        # Value head
        value = self.valueConv(x)
        value = self.valueBN(value)
        value = F.relu(value)
        value = self.valueFlatten(value)
        value = torch.cat([value, features], dim=1)
        value = self.valueLinear(value)

        return policy, value


class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv1d(num_hidden, num_hidden,
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_hidden)
        self.conv2 = nn.Conv1d(num_hidden, num_hidden,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x
