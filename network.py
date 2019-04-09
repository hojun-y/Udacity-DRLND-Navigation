import torch.nn as nn
from config import config


class DQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(37 * config['history_len'], config['fc1'])
        self.bn1 = nn.BatchNorm1d(config['fc1'])
        self.fc2 = nn.Linear(config['fc1'], config['fc2'])
        self.bn2 = nn.BatchNorm1d(config['fc2'])
        self.fc3 = nn.Linear(config['fc2'], 4)

        self.act = nn.ReLU()

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.bn1(self.act(self.fc1(x)))
        x = self.bn2(self.act(self.fc2(x)))
        q = self.fc3(x)
        return q
