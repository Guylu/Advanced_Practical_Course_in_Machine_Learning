import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance_matrix
import scipy
import sklearn
import plotly
import plotly.express as px
from sklearn.manifold import LocallyLinearEmbedding
import pandas as pd
import umap
import pydiffmap
import netflix.preprocess as pp
from sklearn.manifold import SpectralEmbedding
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
import skfuzzy as fuzz
from sklearn.cluster import KMeans



# movie id:
# CustomerID,Rating,Date

# - MovieIDs range from 1 to 17770 sequentially.
# - CustomerIDs range from 1 to 2649429, with gaps. There are 480189 users.
# - Ratings are on a five star (integral) scale from 1 to 5.
# - Dates have the format YYYY-MM-DD.

# movie id mapping is in movie_titles.csv

def digits_example():
    '''
    Example code to show you how to load the MNIST data and plot it.
    '''

    # load the MNIST data:
    digits = datasets.load_digits()
    data = digits.data / 255.
    labels = digits.target

    # plot examples:
    plt.gray()
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.axis('off')
        plt.imshow(np.reshape(data[i, :], (8, 8)))
        plt.title("Digit " + str(labels[i]))
    plt.show()


def swiss_roll_example():
    '''
    Example code to show you how to load the swiss roll data and plot it.
    '''

    # load the dataset:
    X, color = datasets._samples_generator.make_swiss_roll(n_samples=2000)

    # plot the data:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
    plt.show()


def faces_example(path):
    '''
    Example code to show you how to load the faces data.
    '''

    with open(path, 'rb') as f:
        X = pickle.load(f)

    num_images, num_pixels = np.shape(X)
    d = int(num_pixels ** 0.5)
    print("The number of images in the data set is " + str(num_images))
    print("The image size is " + str(d) + " by " + str(d))

    # plot some examples of faces:
    plt.gray()
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(np.reshape(X[i, :], (d, d)))
    plt.show()


def plot_with_images(X, images, title, image_num=25):
    '''
    A plot function for viewing images in their embedded locations. The
    function receives the embedding (X) and the original images (images) and
    plots the images along with the embeddings.

    :param X: Nxd embedding matrix (after dimensionality reduction).
    :param images: NxD original data matrix of images.
    :param title: The title of the plot.
    :param num_to_plot: Number of images to plot along with the scatter plot.
    :return: the figure object.
    '''

    n, pixels = np.shape(images)
    img_size = int(pixels ** 0.5)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)

    # get the size of the embedded images for plotting:
    x_size = (max(X[:, 0]) - min(X[:, 0])) * 0.08
    y_size = (max(X[:, 1]) - min(X[:, 1])) * 0.08

    # draw random images and plot them in their relevant place:
    for i in range(image_num):
        img_num = np.random.choice(n)
        x0, y0 = X[img_num, 0] - x_size / 2., X[img_num, 1] - y_size / 2.
        x1, y1 = X[img_num, 0] + x_size / 2., X[img_num, 1] + y_size / 2.
        img = images[img_num, :].reshape(img_size, img_size)
        ax.imshow(img, aspect='auto', cmap=plt.cm.gray, zorder=100000,
                  extent=(x0, x1, y0, y1))

    # draw the scatter plot of the embedded data points:
    ax.scatter(X[:, 0], X[:, 1], marker='.', alpha=0.7)

    return fig


def MDS(X, d):
    '''
    Given a NxN pairwise distance matrix and the number of desired dimensions,
    return the dimensionally reduced data points matrix after using MDS.

    :param X: NxN distance matrix.
    :param d: the dimension.
    :return: Nxd reduced data point matrix.
    '''

    n = X.shape[0]
    ones = np.ones(n).reshape((n, 1))
    delta = distance_matrix(X, X)
    H = np.eye(n) - (1 / n) * ones.dot(ones.T)
    S = -0.5 * (H.dot(delta.dot(H)))
    w, v = np.linalg.eig(S)  # values, vectors
    return (w * v)[:, :d]


def LLE(X, d, k):
    '''
    Given a NxD data matrix, return the dimensionally reduced data matrix after
    using the LLE algorithm.

    :param X: NxD data matrix.
    :param d: the dimension.
    :param k: the number of neighbors for the weight extraction.
    :return: Nxd reduced data matrix.
    '''

    embedding = LocallyLinearEmbedding(k, d)
    X_transformed = embedding.fit_transform(X)

    return X_transformed


def DiffusionMap(X, d, sigma, t):
    '''
    Given a NxD data matrix, return the dimensionally reduced data matrix after
    using the Diffusion Map algorithm. The k parameter allows restricting the
    kernel matrix to only the k nearest neighbor of each data point.

    :param X: NxD data matrix.
    :param d: the dimension.
    :param sigma: the sigma of the gaussian for the kernel matrix transformation.
    :param t: the scale of the diffusion (amount of time steps).
    :return: Nxd reduced data matrix.
    '''

    pairwise_dists = distance_matrix(X, X)
    K = np.exp(-pairwise_dists ** 2 / (sigma ** 2))
    D = np.diag(np.sum(K, axis=1))
    A = np.linalg.inv(D).dot(K)
    w, v = np.linalg.eig(A)  # values, vectors
    return ((w ** t) * v)[:, 1:d + 1]


def plot_in_plotly(res, method, labels, s=6, addon="", discrete=True):
    """
        plots data in 3D
        res : 3D array
        method : method used for graph
        labels : labels for data (could just pass though np.zeros(res.shape[0])
        s : size of point
        addon : additional info for title of graph
        discrete : whether the labels data is discrete or continuous
    """
    df = pd.DataFrame(res, columns=[method + '_1', method + '_2', method + '_3'])
    if discrete:
        df['labels'] = labels.astype(str)
    else:
        df['labels'] = labels
    fig = px.scatter_3d(df, x=method + '_1', y=method + '_2', z=method + '_3', color='labels')
    fig.update_layout(
        title=method + " Plot " + addon
    )
    fig.update_traces(marker=dict(size=s))
    fig.show()


def plot_2d_reg(res, method, labels, addon=""):
    fig = plt.figure()
    plt.title(method + " Plot " + addon)
    plt.scatter(res[:, 0], res[:, 1], c=labels, cmap=plt.cm.Spectral)
    plt.show()


def plot_reg(res, method, labels, addon=""):
    fig = plt.figure()
    plt.title(method + " Plot" + addon)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(res[:, 0], res[:, 1], res[:, 2], c=labels, cmap=plt.cm.Spectral)
    plt.show()
    df = px.data.iris()
    fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width',
                        color='species')
    fig.show()


def ran_2d_set(n=1000, k=100):
    rand_2d = np.random.multivariate_normal(np.array([10, 30]), np.array([[5, 0], [0, 1]]), 1000)
    rand_2d_in_high_dims = np.zeros(shape=(n, k))
    rand_2d_in_high_dims[:rand_2d.shape[0], :rand_2d.shape[1]] = rand_2d

    O = np.linalg.qr(np.random.normal(0, 1, (k, k)))[0]

    embedded = O.dot(rand_2d_in_high_dims.T).T
    noised = embedded + np.random.normal(0, 1, k).reshape(1, k)

    s2 = np.linalg.svd(rand_2d, compute_uv=False)
    s100 = np.linalg.svd(rand_2d_in_high_dims, compute_uv=False)
    s100_turned = np.linalg.svd(embedded, compute_uv=False)
    s100_turned_noised = np.linalg.svd(noised, compute_uv=False)

    plt.scatter(np.arange(1,len(s2)+1), s2)
    plt.title("Singular Values for 2D data")
    plt.show()

    plt.scatter(np.arange(1, len(s100) + 1), s100)
    plt.title("Singular Values for 2D data embedded in 100D")
    plt.show()

    plt.scatter(np.arange(1, len(s100_turned) + 1), s100_turned)
    plt.title("Singular Values for 2D data embedded in 100D Turned randomly in space")
    plt.show()

    plt.scatter(np.arange(1, len(s100_turned_noised) + 1), s100_turned_noised)
    plt.title("Singular Values for 2D data embedded in 100D Turned randomly in space, \nwith added noise")
    plt.show()


    return rand_2d, noised


def q6():
    orig, data = ran_2d_set()
    labels = np.linalg.norm(orig, axis=1)
    res_mds = MDS(data, 2)
    plot_2d_reg(res_mds, "MDS", labels)
    res_mds = MDS(data, 3)
    plot_in_plotly(res_mds, "MDS", labels, addon="on 2D random data", discrete=False)


def q7_1():
    data, labels = datasets._samples_generator.make_swiss_roll(n_samples=2000)

    res_mds = MDS(data, 2)
    plot_2d_reg(res_mds, "MDS", labels, addon="on swiss roll data")
    res_mds = MDS(data, 3)
    plot_in_plotly(res_mds, "MDS", labels, addon="on swiss roll data", discrete=False)

    res_lle = LLE(data, 2, 12)
    plot_2d_reg(res_lle, "LLE", labels, addon="on swiss roll data")
    res_lle = LLE(data, 3, 12)
    plot_in_plotly(res_lle, "LLE", labels, addon="on swiss roll data", discrete=False)

    res_dm = DiffusionMap(data, 2, 1, 5)
    plot_2d_reg(res_dm, "DM", labels, addon="on swiss roll data")
    res_dm = DiffusionMap(data, 3, 1, 5)
    plot_in_plotly(res_dm, "DM", labels, addon="on swiss roll data", discrete=False)


def q7_2(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    for i in range(2, 10):
        res_mds = MDS(data, i)
        fig = plot_with_images(res_mds, data, "MDS with " + str(i) + " dims", 100)
        plt.savefig("./plots/faces_tests/MDS_" + str(i) + "dims.png")
    for i in range(3, 20):
        res_lle = LLE(data, 2, i)
        fig = plot_with_images(res_lle, data, "LLE with " + str(i) + " neighbors", 100)
        plt.savefig("./plots/faces_tests/LLE" + str(i) + ".png")

    for j in range(2, 20):
        res_dm = DiffusionMap(data, 2, 2200, j)
        fig = plot_with_images(res_dm, data, "DM with t:" + str(j), 100)
        plt.savefig("./plots/faces_tests/DM_" + str(j) + " time.png")



def topics():
    from imdb import IMDb
    # create an instance of the IMDb class
    ia = IMDb()

    mov = ia.search_movie("Jade")


def umap_test(data):
    embedding = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='correlation').fit_transform(data)
    return embedding


def real_DM_test(data):
    mydmap = pydiffmap.diffusion_map.DiffusionMap.from_sklearn(n_evecs=2, alpha=0.5, epsilon='bgh', k=5)
    dmap = mydmap.fit_transform(data)
    fig = plot_with_images(dmap, data, "DM", 100)
    fig.show()


if __name__ == '__main__':
    with open("faces.pickle", 'rb') as f:
        data = pickle.load(f)
    q6()
    with open("faces.pickle", 'rb') as f:
        data = pickle.load(f)
    q7_1()
    res_dm = DiffusionMap(data, 5, 1, 5)
    fig = plot_with_images(res_dm, data, "DM", res_dm.shape[0])
    fig.show()
    plot_in_plotly(res_dm, "DM", np.zeros(res_dm.shape[0]))

    digits = datasets.load_digits()
    data = digits.data / 255.
    labels = digits.target
    fig = plot_with_images(umap_test(data), data, "", 100)
    fig.show()

    q6()
    res_mds = MDS(data, 3)
    res_lle = LLE(data, 3, 5)
    res_dm = DiffusionMap(data, 3, 1, 5)

    res = res_mds
    method = "MDS"

    plot_reg(res, labels, "on MNIST data")
    plot_in_plotly(res, method, labels, addon="on MNIST data")
    fig = plot_with_images(res, data, "")
    fig.show()

    res = pp.create_initial_data(True, True)
    a = np.array(res[1])
    for i in range(a.shape[0]):
        f = False
        for j in range(2, a.shape[1]):
            if a[i, j] != 0:
                if f == False:
                    f = True
                    continue
                a[i, j] = 0
    fin = np.zeros(a.shape[0])
    for i in range(a.shape[0]):
        for j in range(2, a.shape[1]):
            if a[i, j] == 1:
                fin[i] = j
    gen = res[1]
    my = gen.columns[2:]
    my = np.array(my)
    my = np.append(my, ["No"])
    fin -= 2
    fin[np.where(fin == -2)] = 27
    fin2 = np.zeros_like(fin, dtype='object')
    for i in range(len(fin)):
        fin2[i] = my[int(fin[i])]

    little_batch_of_poeple = res[0][:, 0:50000].todense()

    print("loaded little batch")

    a = np.array(little_batch_of_poeple)
    hmr = np.count_nonzero(a, axis=1)
    summ = np.sum(a, axis=1)
    rev = np.divide(summ, hmr, where=hmr != 0)

    p_new = TruncatedSVD(n_components=3).fit_transform(res[0][:, 0:50000])
    print("PCAed it up!")
    plot_in_plotly(p_new, "DM", summ, addon=" color as reviews", discrete=True)
    with open("dm3.pkl", 'rb') as f:
        dm3 = pickle.load(f)
    with open("tsne3.pkl", 'rb') as f:
        tsne = pickle.load(f)
    plot_in_plotly(dm3, "DM", summ, addon=" color as reviews", discrete=True)
    print("lle")
    lle3 = LLE(dm3, 2, 20)
    print("ploting")
    plot_in_plotly(lle3, "lle", summ, addon=" color as reviews", discrete=True)

    kmeans = KMeans(n_clusters=5)
    kmeans.fit(tsne)
    y_kmeans = kmeans.predict(tsne)
    plt.title("DM (" + str(5) + " centroids)")
    plt.scatter(tsne[:, 0], tsne[:, 1], c=y_kmeans, s=50, cmap='viridis')
    plt.xlim([tsne[:, 0].min(), tsne[:, 0].max()])
    plt.ylim([tsne[:, 1].min(), tsne[:, 1].max()])

    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

    plt.savefig("./plots/netflix/KMeans_on_DM" + str(i) + ".png")

    for i in range(3, 30):
        print("k: " + str(i))
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(tsne)
        y_kmeans = kmeans.predict(tsne)
        plt.title("DM (" + str(i) + " centroids)")
        plt.scatter(tsne[:, 0], tsne[:, 1], c=y_kmeans, s=50, cmap='viridis')
        plt.xlim([tsne[:, 0].min(), tsne[:, 0].max()])
        plt.ylim([tsne[:, 1].min(), tsne[:, 1].max()])

        centers = kmeans.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

        plt.savefig("./plots/netflix/KMeans_on_DM" + str(i) + ".png")

    for i in range(3, 30):
        print("lle " + str(i))
        lle = LLE(dm3, 2, i)
        plt.title("LLE (" + str(i) + " neighbors) color as reviews")
        plt.scatter(lle[:, 0], lle[:, 1], c=summ, cmap=plt.cm.Spectral)
        plt.legend()
        plt.savefig("./plots/netflix/LLEDM_" + str(i) + "_neighbors.png")
    print("done")
    plot_in_plotly(dm3, "DM", summ, addon=" color as reviews", discrete=True)
    p_new = TruncatedSVD(n_components=20).fit_transform(res[0][:, 0:50000])
    print("PCAed it up!")

    mds = sklearn.manifold.MDS(n_components=3).fit_transform(p_new)
    print("MDSed it up!")

    with open("tsne3.pkl", 'rb') as f:
        tsne = pickle.load(f)

    plt.title("TSNE color as genres")
    plt.scatter(tsne[:, 0], tsne[:, 1], c=fin, cmap=plt.cm.Spectral)
    plt.legend()
    plt.show()
    plot_in_plotly(tsne, "TSNE", fin2, addon=" color as genres", discrete=True)

    plt.title("TSNE color as reviews")
    plt.scatter(tsne[:, 0], tsne[:, 1], c=summ, cmap=plt.cm.Spectral)
    plt.legend()
    plt.show()
    plot_in_plotly(tsne, "TSNE", summ, addon=" color as reviews", discrete=False)

    p_new = TruncatedSVD(n_components=3).fit_transform(little_batch_of_poeple)
    plt.title("TruncatedSVD color as genres")
    plt.scatter(p_new[:, 0], p_new[:, 1], c=fin, cmap=plt.cm.Spectral)
    plt.legend()
    plt.show()

    plot_in_plotly(p_new, "TruncatedSVD", fin2, addon=" color as genres", discrete=True)
    plt.title("TruncatedSVD color as reviews")
    plt.scatter(p_new[:, 0], p_new[:, 1], c=summ, cmap=plt.cm.Spectral)
    plt.legend()
    plt.show()
    plot_in_plotly(p_new, "TruncatedSVD", summ, addon=" color as reviews", discrete=False)

    um = umap.UMAP(n_neighbors=8, min_dist=0.3, metric='correlation').fit_transform(p_new)
    plt.title("UMAP with year of release")
    plt.xlim([um.min(), um.max()])
    plt.ylim([um.min(), um.max()])
    plt.scatter(um[:, 0], um[:, 1], c=np.array(gen["year_of_release"]), cmap=plt.cm.Spectral)
    plt.legend()
    plt.show()

    print("going on dm")
    with open("dm2.pkl", 'rb') as f:
        dm = pickle.load(f)

    tsne3 = TSNE(n_components=3).fit_transform(p_new)
    with open("tnse3.pkl", "wb") as f:
        pickle.dump(tsne3, f)

    print("going on graphs")
    plt.title("DM with year of release")
    plt.xlim([dm.min(), dm.max()])
    plt.ylim([dm.min(), dm.max()])
    plt.scatter(dm[:, 0], dm[:, 1], c=np.array(gen["year_of_release"], dtype=np.int16), cmap=plt.cm.Spectral)
    plt.legend()
    plt.show()
    plot_in_plotly(dm3, "DM", np.log(summ), addon="on Netflix data set in 3D(logged)", discrete=False)
    plot_in_plotly(dm3, "DM", fin, addon="on Netflix data set in 3D", discrete=False)
    plot_in_plotly(dm3, "DM", fin2, addon="on Netflix data set in 3D", discrete=False)
    plot_in_plotly(dm3, "DM", np.array(gen["year_of_release"]),
                   addon="on Netflix data set in 3D by release date", discrete=True)
    res, method, labels, s, addon, discrete = dm, "DM", np.array(
        gen["year_of_release"]), 6, "on Netflix data set in " \
                                    "3D by release date", True
    df = pd.DataFrame(res, columns=[method + '_1', method + '_2'])
    if discrete:
        df['labels'] = labels.astype(str)
    else:
        df['labels'] = labels
    fig = px.scatter(df, x=method + '_1', y=method + '_2', color='labels')
    fig.update_layout(
        title=method + " Plot " + addon
    )
    fig.update_traces(marker=dict(size=s))
    fig.show()

    pass

    fuzz.arglcut()
