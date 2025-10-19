import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data

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

class SAN(nn.Module):
    """
    Strategic Abstraction Network (SAN)
    A Graph Neural Network that takes a graph representation of the board
    and outputs a goal vector, plan embeddings, and a plan policy.
    """
    def __init__(self, in_channels=16, G_dims=20, P_count=5, P_dims=256):
        super(SAN, self).__init__()

        # Body
        self.conv1 = GCNConv(in_channels, 128)
        self.conv2 = GCNConv(128, 128)

        # Heads
        self.goal_head = nn.Linear(128, G_dims)
        self.plan_embedding_head = nn.Linear(128, P_count * P_dims)
        self.plan_policy_head = nn.Linear(128, P_count)

        self.P_count = P_count
        self.P_dims = P_dims

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Body
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # Global Pooling
        x = global_mean_pool(x, batch)

        # Heads
        goal_vector = torch.sigmoid(self.goal_head(x))
        plan_embeddings = self.plan_embedding_head(x).view(-1, self.P_count, self.P_dims)
        plan_policy = self.plan_policy_head(x)

        return goal_vector, plan_embeddings, plan_policy

class PlanToMoveMapper(nn.Module):
    """
    An MLP that maps a plan embedding to a policy bias vector.
    """
    def __init__(self, P_dims=256, policy_dims=4672):
        super(PlanToMoveMapper, self).__init__()
        self.fc1 = nn.Linear(P_dims + policy_dims, 1024)
        self.fc2 = nn.Linear(1024, policy_dims)

    def forward(self, plan_embedding, policy_logits):
        # Concatenate the plan embedding and policy logits
        x = torch.cat([plan_embedding, policy_logits], dim=1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
