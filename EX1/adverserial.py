from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage.color
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import math
from torch.utils.data import Dataset

import dataset
import models

PATH = './naive4.pth'
d = dataset.get_dataset_as_torch_dataset("./data/dev.pickle")
criterion = nn.CrossEntropyLoss()
net = models.SimpleModel()
net.load_state_dict(torch.load(PATH))
# pic 142 is of a cat that looked nice :)
empty = torch.zeros_like(d.the_list[142][0])
optimizer = torch.optim.Adam([empty], weight_decay=800)
empty.requires_grad_(True)
for a in net.parameters():
    a.requires_grad = False

inputs = d.the_list[142][0]
labels = d.the_list[142][1]
# "train":
for k in range(200):
    c = inputs + empty

    # forward + backward + optimize
    outputs = net(c.unsqueeze(0))
    p = torch.argmax(outputs, 1)
    print(p)
    loss = criterion(outputs, torch.tensor(1).unsqueeze(0))
    loss.backward()
    optimizer.step()

# print
img = inputs / 2 + 0.5  # unnormalize
npimg = img.numpy()
plt.imshow(np.transpose(npimg, (1, 2, 0)))
plt.show()
print(empty)
