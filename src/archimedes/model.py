import torch
import torch.nn as nn
import torch.nn.functional as F

class TPN(nn.Module):
    """
    Tactical Perception Network (TPN)
    A convolutional network that takes a tensor representation of the board
    and outputs a policy and a value.
    """
    def __init__(self):
        super(TPN, self).__init__()
        # Input shape: (N, 22, 8, 8)

        # Body
        self.conv1 = nn.Conv2d(22, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # Policy Head
        self.policy_conv = nn.Conv2d(128, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * 8 * 8, 4672)

        # Value Head
        self.value_conv = nn.Conv2d(128, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # Body
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Policy Head
        p = F.relu(self.policy_conv(x))
        p = p.view(p.size(0), -1) # Flatten
        p = self.policy_fc(p)

        # Value Head
        v = F.relu(self.value_conv(x))
        v = v.view(v.size(0), -1) # Flatten
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v
