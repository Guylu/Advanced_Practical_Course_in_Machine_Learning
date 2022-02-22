import umap
import pandas as pd
import matplotlib.pyplot as plt
import colorcet
import matplotlib.colors
import matplotlib.cm
import numpy as np
import umap.plot
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import pydiffmap

from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import TSNE,MDS
from scipy.spatial import distance_matrix

data = pd.read_csv("C:/Users/guylu/Desktop/your_file.csv")
# counts = counts.T -> if needed we need columns as features

#               python implementations:


# PCA:
pca = PCA(n_components=3)
data_pca3 = pca.fit_transform(data)

# MDS:
mds = MDS(n_components=3)
data_mds3 = mds.fit_transform(data)

# LLE:
num_neighbors = 20
lle = LocallyLinearEmbedding(num_neighbors, 3)
data_lle3 = lle.fit_transform(data)

# DM:
k = 20
mydmap = pydiffmap.diffusion_map.DiffusionMap.from_sklearn(n_evecs=3, k=k)
data_dm3 = mydmap.fit_transform(data)

# t-SNE:

data_tsne3 = TSNE(n_components=3).fit_transform(data)

# UMAP:

data_umap3 = umap.UMAP(random_state=42, n_neighbors=11, min_dist=0.05, n_components=3).fit_transform(data)


#                         My implementation:


# PCA:

# I will add but its just centering and scaling data
# and then taking S = X * x^T
# and using scipy to diagonalize S


# MDS:


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


# Diffusion Map:


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


# now you can plot what you want

import winsound

duration = 1000  # milliseconds
freq = 440  # Hz
winsound.Beep(freq, duration)
from bertopic import BERTopic
