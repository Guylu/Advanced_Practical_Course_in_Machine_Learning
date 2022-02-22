import umap
import pandas as pd
import datashader as ds
import datashader.transfer_functions as tf
import datashader.bundling as bd
import matplotlib.pyplot as plt
import colorcet
import matplotlib.colors
import matplotlib.cm
import bokeh.plotting as bpl
import bokeh.transform as btr
import holoviews as hv
import holoviews.operation.datashader as hd
import numpy as np
import umap.plot

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import plotly.express as px
import umap.utils as utils
import umap.aligned_umap

pca = pd.read_csv("C:/Users/guylu/Desktop/pca.csv")
idents = pd.read_csv("C:/Users/guylu/Desktop/idents.csv")
idents2 = idents.replace({"Homeostatic.1": 1, "Homeostatic.2": 2, "Homeostatic.3": 3, "Reactive.1": 4,
                          "Reactive.2":
                              5})
l = idents2.values.tolist()
l[:] = [x[0] for x in l]
l[:] = [x / 5 for x in l]
g = idents.values.tolist()
g[:] = [x[0] for x in g]
g2 = np.array(g)


# em_den_correlation = umap.UMAP(random_state=42,
#                                n_neighbors=30,
#                                min_dist=0.3,
#                                n_components=3,
#                                metric="correlation",
#                                densmap=True).fit_transform(pca)

# umap.plot.points(em_den_correlation, labels=g2, width=500, height=500, theme="fire")
# plt.title("Seurat Inspired Run With UMAP 3d")
# plt.show()


# u = em_den_correlation
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(u[:, 0], u[:, 1], u[:, 2], c=l, s=100)
# plt.title("3d denUMAP of def R vals", fontsize=18)
# plt.show()


def plot_in_plotly(res, method, labels, s=6, addon="", discrete=True, save=False):
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
    df['labels'] = labels

    fig2 = px.scatter_3d(df, x=method + '_1', y=method + '_2', z=method + '_3', color='labels')
    fig2.update_layout(
        title=method + " Plot " + addon,
        template="plotly_dark",
        scene = dict(
        xaxis=dict(showgrid=False, showticklabels=False, title=''),
        yaxis=dict(showgrid=False, showticklabels=False, title=''),
        zaxis=dict(showgrid=False, showticklabels=False, title=''))
    )
    fig2.update_traces(marker=dict(size=s))
    if save:
        fig2.write_html("sparse umap plots/" + method + " Plot " + addon + ".html")
    else:
        fig2.show()

import plotly.graph_objects as go

#
# def axis_bounds(embedding):
#     left, right = embedding.T[0].min(), embedding.T[0].max()
#     bottom, top = embedding.T[1].min(), embedding.T[1].max()
#     adj_h, adj_v = (right - left) * 0.1, (top - bottom) * 0.1
#     return [left - adj_h, right + adj_h, bottom - adj_v, top + adj_v]
#
#
# p = np.array(pca)
# constant_dict = {i:i for i in range(p.shape[0])}
# constant_relations = [constant_dict for i in range(9)]
#
# neighbors_mapper = umap.AlignedUMAP(
#     n_neighbors=[3,4,5,7,11,16,22,29,37,45,54],
#     min_dist=[0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45],
#     alignment_window_size=2,
#     alignment_regularisation=1e-3,
# ).fit(
#     [p for i in range(10)], relations=constant_relations
# )
#
# fig, axs = plt.subplots(5,2, figsize=(10, 20))
# ax_bound = axis_bounds(np.vstack(neighbors_mapper.embeddings_))
# for i, ax in enumerate(axs.flatten()):
#     ax.scatter(*neighbors_mapper.embeddings_[i].T, s=2, c=l, cmap="Spectral")
#     ax.axis(ax_bound)
#     ax.set(xticks=[], yticks=[])
# plt.tight_layout()
#
# plt.show()
# for k in ["euclidean", "minkowski", "haversine", "cosine", "correlation", "seuclidean"]:
#     for i in [3, 4, 5, 7, 11, 16, 22, 29, 37, 45, 54]:
#         for j in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.7]:
#             em = umap.UMAP(random_state=42,
#                            n_neighbors=i,
#                            min_dist=j,
#                            n_components=3,
#                            metric=k,
#                            densmap=True).fit_transform(pca)
#             plot_in_plotly(em, "UMAP", g2, addon="met_" + str(k) + "_n_" + str(i) + "_d_" + str(j), save=True,
#                            s=2)

data = pd.read_csv("C:/Users/guylu/Desktop/scaled__.csv")
data = data.T

for i in [3, 4, 5, 7, 11, 16, 22, 29, 37, 45, 54]:
    for j in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.7]:
        mapper = umap.UMAP(metric='cosine', random_state=42, low_memory=True, n_neighbors=i,
                           min_dist=j,
                           n_components=3).fit_transform(data)
        plot_in_plotly(mapper, "UMAP", g2[1:], addon=" no dense no pca n_" + str(i) + "_d_" + str(j),
                       save=True,
                       s=2)
from sklearn.decomposition import TruncatedSVD


mapper_arc_cm = None

fig = go.Figure()
method = "UMAP"
df = pd.DataFrame(mapper_arc_cm, columns=[method + '_1', method + '_2', method + '_3'])
for i in range(5):
    fig.add_trace(
        go.Scatter3d(x =mapper_arc_cm[:,0] ,y =mapper_arc_cm[:,1],z =mapper_arc_cm[:,2],visible=False,
                     name="Distance from Archetype:" + str(i),mode='markers',
    marker=dict(
        size=2,
        color=dist[:,i],                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )))
    fig.update_layout(
        title=method + " Plot " ,
        template="plotly_dark"
    )

# Make 10th trace visible
fig.data[0].visible = True

steps = []
for i in range(len(fig.data)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              {"title": "Slider switched to distance from Archetype: " + str(i)}],  # layout attribute
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active=5,
    currentvalue={"prefix": "Distance from Archetype: "},
    pad={"t": 50},
    steps=steps
)]

fig.update_layout(
    sliders=sliders
)

fig.write_html("sparse umap plots/" + method + " Plot cool" + ".html")






from sklearn.decomposition import NMF


nmf = NMF(n_components=5, random_state=42)