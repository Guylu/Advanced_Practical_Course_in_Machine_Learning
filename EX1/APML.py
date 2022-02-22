
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

if __name__ == '__main__':
    # augmentations:
    # transform = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomVerticalFlip(),
    #     transforms.RandomGrayscale(p=0.1),
    #     transforms.RandomPerspective(),
    #     transforms.RandomRotation(10),
    #     transforms.RandomRotation(20),
    #     transforms.RandomRotation(30),
    #     transforms.RandomRotation(40),
    #     transforms.RandomRotation(50),
    #     transforms.RandomRotation(60),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])
    #
    # testloader = torch.utils.data.DataLoader(
    #     dataset.get_dataset_as_torch_dataset("./data/dev.pickle"))
    #
    # trainloader = torch.utils.data.DataLoader(
    #     dataset.get_dataset_as_torch_dataset("./data/train.pickle"))
    #
    # trainloader = dataset.MyDataset(trainloader, transform=transform)
    # data = dataset.get_dataset_as_torch_dataset("./data/train.pickle")
    #
    # # this was for trying to train only cats/trucks
    # # data.the_list = [a for a in data.the_list if a[1] != 0]
    #
    # # batch learning:
    # l = [b for a, b in data.the_list]
    # cls_count = np.histogram(l, bins=3)[0]
    # cls_weights = 1. / torch.tensor(cls_count, dtype=torch.float)
    #
    # weights = cls_weights[l]
    # weighted_sampler = WeightedRandomSampler(weights, num_samples=len(weights),
    #                                          replacement=True)
    # train_loader_m = DataLoader(data, batch_size=5)
    #
    # trainloader = train_loader_m
    # # load network
    net = models.SimpleModel()
    # criterion = nn.CrossEntropyLoss()
    # sm = torch.nn.Softmax()
    # # lr = 1e-4
    # optimizer = optim.Adam(net.parameters())
    # k = 20
    # for epoch in range(k):  # loop over the dataset multiple times
    #     running_loss = 0.0
    #     for i, data in enumerate(trainloader, 0):
    #         # get the inputs; data is a list of [inputs, labels]
    #         inputs, labels = data
    #
    #         # zero the parameter gradients
    #         optimizer.zero_grad()
    #
    #         # forward + backward + optimize
    #         outputs = net(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #
    #         # print statistics
    #         running_loss += loss.item()
    #         # if i % 50 == 49:  # print every 2000 mini-batches
    #         #     print('[%d, %5d] loss: %.3f' %
    #         #           (epoch + 1, i + 1, running_loss / 2000))
    #         #     running_loss = 0.0
    #     print(epoch / k)
    #
    # print('Finished Training')

    PATH = './207029448.ckpt'

    data_test = dataset.get_dataset_as_torch_dataset("./data/test.pickle")

    # testing trucks and cats:
    # data_test.the_list = [a for a in data_test.the_list if a[1] != 0]
    testloader = DataLoader(data_test)

    dataiter = iter(testloader)
    images, labels = dataiter.next()
    net = models.SimpleModel()
    net.load_state_dict(torch.load(PATH))

    # graphing learning rate(code for generating the loss array was omitted):
    # x = [i for i in range(len(losses[0]))]
    # fig, axes = plt.subplots(4, 3, figsize=(8, 6))
    # # fig.suptitle('This is the Figure Title', fontsize=15)
    # for i in range(4):
    #     for j in range(3):
    #         if 3 * i + j >= len(losses):
    #             break
    #         axes[i, j].plot(x, losses[3 * i + j])
    #         axes[i, j].set_title("Learning Rate: " + str(lrs[3 * i + j]) +
    #                              "\n Final Loss: " +
    #                              str(round(losses[3 * i + j][-1], 2)))
    #
    # plt.show()

    # evaluation:
    correct = 0
    total = 0
    predicted_all = []
    labels_all = []
    conf = []
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            predicted_all.extend(list(predicted.numpy()))
            labels_all.extend(list(labels.numpy()))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # if i not in [24, 30, 38, 79, 90, 112, 168, 188, 200, 220, 223, 248,
            #              251] and \
            #         predicted[-1].item() == 2:
            #     probabilities = sm(outputs)
            #     print(str(i) + ":  " + str(probabilities[-1][-1].item()))  #
            #     conf.append(probabilities[-1][-1].item())
            #     # Converted to
            #     # probabilities
            #     plt.title(str(i))
            #     plt.imshow(dataset.un_normalize_image(images[-1]))
            #     plt.show()
            #     print(str(labels[-1].item()) + "    "
            #           + str(predicted[-1].item()))

            # if predicted != labels and predicted[-1].item() == 2:
            #     plt.imshow(dataset.un_normalize_image(images[-1]))
            #     plt.show()
            #     print(str(labels[-1].item()) + "    "
            #           + str(predicted[-1].item()))
    # print(sum(conf) / len(conf))
    print((labels_all))
    print(confusion_matrix(labels_all, predicted_all))
    print('Accuracy of the network : %d %%' % (
            100 * correct / total))
