import models
import datasets
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import io
import tensorflow as tf
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import imageio
import skimage.color

num = 2000
k = 25
d = datasets.MNIST(num)
indexes = np.random.choice(num // 11, 4, replace=False)


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


sig_list = [0, 1e-100, 1e-20, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1, 2, 3, 4, 5, 10, 20, 50, 100, 500,
            1000]
num_of_run = 14
Home_Folder = "L1_" + str(num_of_run)
Path("plots/" + Home_Folder).mkdir(parents=True, exist_ok=True)
Path("logs/" + Home_Folder).mkdir(parents=True, exist_ok=True)
for sig in sig_list:
    print(sig)
    p = "plots/" + Home_Folder + "/sig_" + str(sig) + "_k-" + str(k) + "_" + str(num)
    Path(p).mkdir(parents=True, exist_ok=True)
    p_tensor = "logs/" + Home_Folder + "/sig_" + str(sig) + "_k-" + str(k) + "_" + str(num)
    Path(p_tensor).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=p_tensor + "/")
    Net_Path = "MyNet_" + Home_Folder + "_sig-" + str(sig) + "_k-" + str(k) + "_" + str(num) + ".pt"
    Con_Path = "ContentMat_" + Home_Folder + "_sig-" + str(sig) + "_k-" + str(k) + "_" + str(num) + ".pt"
    Cls_Path = "ClassMat_" + Home_Folder + "_sig-" + str(sig) + "_k-" + str(k) + "_" + str(num) + ".pt"

    #################################################################3

    content_mat = torch.randn((num, k), requires_grad=True)
    class_mat = torch.randn((10, k), requires_grad=True)

    net = models.GeneratorForMnistGLO(2 * k)

    optimizer = optim.Adam(net.parameters())
    optimizer_class = optim.Adam([class_mat])
    optimizer_content = optim.Adam([content_mat])  # ,weight_decay=

    dataloader = DataLoader(d, batch_size=32)

    ll = None
    if Home_Folder[:3] == "L1_":
        ll = lambda x, y: nn.L1Loss()(x, y)
    elif Home_Folder[:3] == "L2_":
        ll = lambda x, y: nn.MSELoss()(x, y)
    elif Home_Folder[:3] == "L1+":
        ll = lambda x, y: nn.MSELoss()(x, y) + nn.L1Loss()(x, y)

    ep = 50
    im_list = []
    for epoch in range(ep):  # loop over the dataset multiple times
        for i, data in enumerate(dataloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, idx = data
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(torch.cat((class_mat[labels], content_mat[idx] + torch.randn(k) * sig),
                                    dim=1).unsqueeze(0))

            loss = ll(inputs, outputs)
            loss.backward()

            optimizer.step()
            optimizer_class.step()
            optimizer_content.step()

            writer.add_scalar('loss', loss, i + epoch * len(dataloader))
        im_list.append(outputs[0].detach().numpy())


        def TestConversion2(cls, idxs):
            outputs = []
            GT_imgs = []
            for idx in idxs:
                img, label, idx = d.__getitem__(idx)
                GT_imgs.append(img)

                im_cont_emb = content_mat[idx, :]
                im_cls_emb = class_mat[cls, :]

                lat_code = torch.cat((im_cls_emb, im_cont_emb), dim=0).unsqueeze(0)

                output = net(lat_code)
                output = output.detach().numpy()
                outputs.append(output)

            d.present_imgs(GT_imgs, outputs, convert_from, convert_to, p, epoch, sig)


        number__ = []
        convert_from = 4
        convert_to = 3

        for asd in range(num):
            if d.data[asd][1] == convert_from:
                number__.append(asd)

        number__random = np.array(number__)[indexes]

        TestConversion2(convert_to, number__random)

        for immmm in range(16):
            plt.imshow(outputs[immmm].detach().numpy().squeeze())
            plt.savefig("plots/" + Home_Folder + "/pic_num_" + str(immmm) + "___" + str(epoch) + ".png")

    figure = plt.figure(figsize=(10, 10))
    figure.suptitle("sig:" + str(sig))

    used_im_list = im_list[0::2]
    for i_j in range(len(used_im_list)):
        # Start next subplot.
        plt.subplot(5, 5, i_j + 1, title=" epoch: " + str(i_j * 2))
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.axis('off')
        plt.grid(b=None)
        plt.imshow(used_im_list[i_j].squeeze(), cmap=plt.cm.binary)

    Path("plots/" + Home_Folder + "/imgs_epochs").mkdir(parents=True, exist_ok=True)
    epoch_learning_path = "plots/" + Home_Folder + "/imgs_epochs/sig_" + str(sig) + ".png"
    plt.savefig(epoch_learning_path)

    imim = np.asarray(imageio.imread(epoch_learning_path), dtype=np.float64)
    imim = skimage.color.rgb2gray(imim) / 255
    qq = np.expand_dims(imim, axis=0)
    writer.add_image('image', qq, sig_list.index(sig))

    torch.save(net.state_dict(), Net_Path)
    torch.save(class_mat, Cls_Path)
    torch.save(content_mat, Con_Path)

    ##################################################################

    net = models.GeneratorForMnistGLO(2 * k)
    net.load_state_dict(torch.load(Net_Path))
    content_mat = torch.load(Con_Path)
    class_mat = torch.load(Cls_Path)


    def TestConversion(cls, idxs):
        outputs = []
        GT_imgs = []
        for idx in idxs:
            img, label, idx = d.__getitem__(idx)
            GT_imgs.append(img)

            im_cont_emb = content_mat[idx, :]
            im_cls_emb = class_mat[cls, :]

            lat_code = torch.cat((im_cls_emb, im_cont_emb), dim=0).unsqueeze(0)

            output = net(lat_code)
            output = output.detach().numpy()
            outputs.append(output)

        d.present_imgs(GT_imgs, outputs, convert_from, convert_to, p)


    for i in range(10):
        for j in range(10):
            number__ = []
            convert_from = i
            convert_to = j

            for asd in range(num):
                if d.data[asd][1] == convert_from:
                    number__.append(asd)

            number__random = np.array(number__)[indexes]

            TestConversion(convert_to, number__random)

    rand_num = torch.randn(k)
    outputss = []
    for i in range(10):
        code = torch.cat((class_mat[i, :], rand_num), dim=0).unsqueeze(0)
        output = net(code)
        output = output.detach().numpy()
        outputss.append(output)

    fig, ax = plt.subplots(1, len(outputss))
    fig.suptitle("Generates plots for MNIST data set\nWith sigma =" + str(sig))

    for i, img in enumerate(outputss):
        img = img.squeeze()
        ax[i].imshow(img)
    plt.axis('off')
    plt.savefig(
        "plots/" + Home_Folder + "/generate_from_scratch_sig_" + str(sig_list.index(sig)) + "____.png")
