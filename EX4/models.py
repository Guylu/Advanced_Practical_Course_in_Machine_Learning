import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleModel(nn.Module):
    # A simple MLP that depends on the 3x3 area around the snakes head.
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # nhwc -> nchw
        center = x[:, :, 3:6, 3:6]
        # The 9 cells around the snakes head (including the head), encoded as one-hot.
        center = torch.flatten(center, start_dim=1)
        # center = F.relu(self.linear1(center))
        center = self.linear2(center)
        softmax = torch.nn.Softmax(dim=1)
        return softmax(center)


class UrolledModel(nn.Module):
    # A simple MLP that depends on the 3x3 area around the snakes head.
    def __init__(self, input_size, hidden_size, output_size):
        super(UrolledModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # nhwc -> nchw
        center = x[:, :, 3:6, 3:6]
        # The 9 cells around the snakes head (including the head), encoded as one-hot.
        center = torch.flatten(center, start_dim=1)
        center = F.relu(self.linear1(center))
        center = self.linear2(center)
        return center


class Conv2LNet(nn.Module):
    def __init__(self):
        super(Conv2LNet, self).__init__()
        n_act = 3
        self.conv1 = nn.Conv2d(10, 20, 5)
        self.conv2 = nn.Conv2d(20, 40, 5)

        self.linear1 = nn.Linear(40 * 1 * 1, n_act)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # nhwc -> nchw
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = out.view(-1, 40 * 1 * 1)
        out = self.linear1(out)
        out = self.softmax(out)
        return out
