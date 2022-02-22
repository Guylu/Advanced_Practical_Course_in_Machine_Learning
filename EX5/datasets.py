import torch
from tqdm import tqdm
import numpy as np
from torchvision import datasets
import matplotlib.pyplot as plt


# every image is represented as [c, h, w], each pixel is in the range [0,1], and has a method for presenting it.
class MNIST(torch.utils.data.Dataset):
    """
    mnist dataset. images are labeled by digit (class) and img_id (content)
    """

    def __init__(self, num_samples):
        self.data = []

        dataset = datasets.MNIST('./data', train=False, download=True)

        for idx, sample in enumerate(tqdm(dataset)):
            if idx < num_samples:
                img = np.array(sample[0])
                img = np.expand_dims(img, 0)
                img = img.astype(np.float32) / 255.0
                self.data.append((img, sample[1], idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample

    @staticmethod
    def present(img):
        img = img.squeeze()
        plt.imshow(img)
        plt.show()

    @staticmethod
    def present_imgs(GT_imgs, imgs, convert_from, convert_to, p, epoch=0, sig=0):
        fig, ax = plt.subplots(2, len(imgs))
        fig.suptitle("Converting " + str(convert_from) + "'s to " + str(convert_to) + "\nsig:" + str(sig) + " "
                                                                                                           "epoch:" + str(
            epoch))

        for i, img in enumerate(imgs):
            img = img.squeeze()
            gt_img = GT_imgs[i].squeeze()
            ax[0, i].imshow(gt_img)
            ax[1, i].imshow(img)

        # plt.show()
        plt.savefig(p + "/Converting " + str(convert_from) + "'s to " + str(
            convert_to) + "_" + str(epoch) + ".png")
