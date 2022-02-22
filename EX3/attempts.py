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

import winsound
import pydiffmap
import scipy.linalg as spla

from sklearn.decomposition import TruncatedSVD
from scipy.spatial.distance import cdist
import plotly.graph_objects as go


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
        template="plotly_dark"
    )
    fig2.update_traces(marker=dict(size=s))
    if save:
        fig2.write_html("sparse umap plots/" + method + " Plot " + addon + ".html")
    else:
        fig2.show()


data = pd.read_csv("C:/Users/guylu/Desktop/scaled__.csv")
import winsound

duration = 1000  # milliseconds
freq = 440  # Hz
winsound.Beep(freq, duration)
mapper2g = umap.UMAP(output_metric='gaussian_energy', metric='cosine', random_state=42, low_memory=True,
                     n_neighbors=11, min_dist=0.05, n_components=3).fit_transform(data)
plot_in_plotly(mapper2g, UMAP_, g2, addon=GE, save=True, s=2)
winsound.Beep(frequency, duration)
winsound.Beep(freq, duration)
plot_in_plotly(mapper2g, "UMAP_", g2, addon="GE", save=True, s=2)
winsound.Beep(freq, duration)
data = data.T
winsound.Beep(freq, duration)
mapper2g = umap.UMAP(output_metric='gaussian_energy', metric='cosine', random_state=42, low_memory=True,
                     n_neighbors=11, min_dist=0.05, n_components=3).fit_transform(data)
plot_in_plotly(mapper2g, "UMAP_", g2, addon="GE", save=True, s=2)
winsound.Beep(freq, duration)
import pydiffmap

winsound.Beep(freq, duration)
mapper2g_den = umap.UMAP(densmap=True, output_metric='gaussian_energy', metric='cosine', random_state=42,
                         low_memory=True,
                         n_neighbors=11, min_dist=0.05, n_components=3).fit_transform(data)
plot_in_plotly(mapper2g_den, "UMAP_", g2, addon="GE den", save=True, s=2)
winsound.Beep(freq, duration)
winsound.Beep(freq, duration)
mapper_den = umap.UMAP(densmap=True, metric='cosine', random_state=42,
                       low_memory=True,
                       n_neighbors=11, min_dist=0.05, n_components=3).fit_transform(data)
plot_in_plotly(mapper_den, "UMAP_", g2, addon="res den", save=True, s=2)
winsound.Beep(freq, duration)
arc = pd.read_csv(
    "C:/Users/guylu/Desktop/Archtypes_scaled__data_5_1_Sysal_arcOrig__orig_take_only_first_3_cols_.csv")
arc = pd.read_csv(
    "C:/Users/guylu/Desktop/Archtypes_scaled__data_5_1_Sysal_arcOrig__orig_take_only_first_3_cols.csv")
frames = [data, arc]
marged = pd.concat(frames)
merged = pd.concat(frames)
frames = [data.T, arc.T]
merged = pd.concat(frames)
merged = pd.concat(frames, axis=1)
merged = pd.concat(frames, axis=0)
arc = pd.read_csv(
    "C:/Users/guylu/Desktop/Archtypes_scaled__data_5_1_Sysal_arcOrig__orig_take_only_first_3_cols.csv",
    header=None)
arc.columns = data.columns
frames = [data, arc]
merged = pd.concat(frames)
-0.22849 in merged[30078:30079]
g3 = np.append(g2, ["A1", "A2", "A3", "A4", "A5"])
winsound.Beep(freq, duration)
mapper_den = umap.UMAP(metric='cosine', random_state=42,
                       low_memory=True,
                       n_neighbors=11, min_dist=0.05, n_components=3).fit_transform(merged)
plot_in_plotly(mapper_den, "UMAP_", g3, addon="res den", save=True, s=2)
winsound.Beep(freq, duration)
d = data._values
g2 == "Homoestatic.1"
np.where(g2 == "Homoestatic.1")
np.where(g2 == "Homoestatic.1")[0]
c1 = np.where(g2 == "Homoestatic.1")[0]
c1 = np.where(g2 == "Homoestatic.1")
c1 = np.where(g2 == "Homeostatic.1")
c2 = np.where(g2 == "Homeostatic.2")
c3 = np.where(g2 == "Homeostatic.3")
c4 = np.where(g2 == "Reactive.1")
c5 = np.where(g2 == "Reactive.2")
cm1 = np.mean(d[c1])
cm1 = np.mean(d[c1], axis=0)
cm2 = np.mean(d[c2], axis=0)
cm3 = np.mean(d[c3], axis=0)
cm4 = np.mean(d[c4], axis=0)
cm5 = np.mean(d[c5], axis=0)
merged2 = np.append(merged, cm1)
merged2 = np.append(merged, cm1, axis=1)
merged2 = np.append(merged, cm1, axis=0)
merged2 = np.empty((30088, 2000))
merged2[0:30073, :] = merged
merged2[0:30083, :] = merged
merged2[30084:30085, :] = cm1
merged2[30085:30086, :] = cm2
merged2[30086:30087, :] = cm3
merged2[30087:30088, :] = cm4
merged2[30088:30089, :] = cm5
np.mean(d[c3], axis=0)
merged2[30084:30085, :]
merged2[30083:30084, :] = cm1
merged2[30084:30085, :] = cm2
merged2[30085:30086, :] = cm3
merged2[30086:30087, :] = cm4
merged2[30087:30088, :] = cm5
g4 = np.empty(30088)
g4[0:30083] = g3
g4 = np.empty(30088, dtype="string")
g4 = np.empty(30088, dtype=g3.dtype)
g4[0:30083] = g3
g4[30084:30088]
g4[30083:30088]
g4[30082:30088]
g4[30083:30088]
g4[30083:30088] = np.array(["CM1", "CM2", "CM3", "CM4", "CM5"])
winsound.Beep(freq, duration)
mapper_arc_cm = umap.UMAP(metric='cosine', random_state=42,
                          low_memory=True,
                          n_neighbors=11, min_dist=0.05, n_components=3).fit_transform(merged2)
plot_in_plotly(mapper_arc_cm, "UMAP_", g4, addon="reg_n_11_d_0.05_arch_CM", save=True, s=2)
winsound.Beep(freq, duration)
winsound.Beep(freq, duration)
mapper_den = umap.UMAP(densmap=True, metric='cosine', random_state=42,
                       low_memory=True,
                       n_neighbors=11, min_dist=0.05, n_components=3).fit_transform(data)
plot_in_plotly(mapper_den, "DenUMAP_", g2, addon="den_n_11_d_0.05", save=True, s=2)
winsound.Beep(freq, duration)
winsound.Beep(freq, duration)
mapper_fit = umap.UMAP(metric='cosine', random_state=42,
                       low_memory=True,
                       n_neighbors=11, min_dist=0.05, n_components=2).fit(data)
winsound.Beep(freq, duration)
umap.plot.points(mapper_fit, values=g2, theme='fire')
umap.plot.points(mapper_fit, values=l, theme='fire')
umap.plot.points(mapper_fit, values=np.array(l), theme='fire')
plt.show()
umap.plot.points(mapper_fit, values=np.array(l))
plt.show()
umap.plot.points(mapper_fit, values=np.array(l), theme="inferno")
plt.show()
winsound.Beep(freq, duration)
mapper_rr = umap.UMAP(densmap=True, metric='cosine', random_state=42,
                      low_memory=True,
                      n_neighbors=11, min_dist=0.05, n_components=3).fit_transform(data)
plot_in_plotly(mapper_rr, "UMAP_", g2, addon="reg_n_11_d_0.05", save=True, s=2)
winsound.Beep(freq, duration)
winsound.Beep(freq, duration)
mapper_rr = umap.UMAP(metric='cosine', random_state=42,
                      low_memory=True,
                      n_neighbors=11, min_dist=0.05, n_components=3).fit_transform(data)
plot_in_plotly(mapper_rr, "UMAP_", g2, addon="reg_n_11_d_0.05", save=True, s=2)
winsound.Beep(freq, duration)
umap.plot.points(mapper_fit, values=np.array(l))
plt.show()
umap.plot.connectivity(mapper_fit, show_points=True)
plt.show()
umap.plot.connectivity(mapper_fit, show_points=True, theme="fire")
plt.show()
umap.plot.connectivity(mapper_fit, edge_bundling='hammer')
plt.show()
umap.plot.connectivity(mapper_fit, edge_bundling='hammer', theme="fire")
plt.show()
winsound.Beep(freq, duration)
mapper_p = umap.UMAP(metric='cosine', random_state=42,
                     low_memory=True,
                     n_neighbors=15, min_dist=0.1, n_components=2).fit(data)
umap.plot.points(mapper_p, values=np.array(l), theme="fire")
plt.show()
umap.plot.connectivity(mapper_p, show_points=True)
plt.show()
umap.plot.connectivity(mapper_p, show_points=True, theme="fire")
plt.show()
umap.plot.connectivity(mapper_p, edge_bundling='hammer')
plt.show()
umap.plot.connectivity(mapper_p, edge_bundling='hammer', theme="fire")
plt.show()
winsound.Beep(freq, duration)
winsound.Beep(freq, duration)
W = arc.T
W = W._values
V = data.T
V2 = np.repeat(V[:, :, np.newaxis], 30078, axis=0)
V2 = np.repeat(V[:, :, np.newaxis], 30078, axis=2)
V2 = np.repeat(V[:, :, np.newaxis], 30078, axis=0)
np.linalg.solve(W, V)
Q, R = la.qr(W)
Q, R = np.linag.qr(W)
Q, R = np.linalg.qr(W)
import scipy.linalg as spla

X = spla.solve_triangular(R, Q.T.dot(V), lower=False)
Q, R = np.linalg.qr(W.T)
X = spla.solve_triangular(R, Q.T.dot(V), lower=False)
X = spla.solve_triangular(R, Q.T.dot(V.T), lower=False)
Q, R = np.linalg.qr(W)
X = spla.solve_triangular(R, Q.T.dot(V), lower=False)
X = spla.solve_triangular(R, Q.T.dot(V[:, 0]), lower=False)
V[:, 0]
V = V._values
V[:, 0]
X = spla.solve_triangular(R, Q.T.dot(V[:, 0]), lower=False)
X = spla.solve_triangular(R, Q.T.dot(V), lower=False)
n = W.shape[1]
n
x2 = spla.solve_triangular(R[:n], Q.T[:n].dot(V), lower=False)
X - x2
X3 = la.solve(W.T.dot(W), W.T.dot(V))
X3 = np.linalg.solve(W.T.dot(W), W.T.dot(V))
X - X3
np.max(X - X3)
np.var(X, axis=0)
np.var(X, axis=1)
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=3, n_iter=7, random_state=42).fit_transform(X.T)
plot_in_plotly(svd, "PCA", g2, addon=" archetypes as axis",
               save=True,
               s=2)
mapper_arch_def = umap.UMAP(random_state=42, n_components=3).fit_transform(X.T)
plot_in_plotly(mapper_arch_def, "UMAP", g2, addon=" archetypes as axis",
               save=True,
               s=2)
history
mapper_arch_def_den = umap.UMAP(random_state=42, n_components=3, densmap=True).fit_transform(X.T)
plot_in_plotly(mapper_arch_def_den, "Den_UMAP", g2, addon=" archetypes as axis den",
               save=True,
               s=2)
from scipy.spatial.distance import cdist

dist = cdist(data._values, arc._values)
dist = cdist(merged2, arc._values)
plot_in_plotly(mapper_arc_cm, "UMAP", dist[:, 0], addon="dist1",
               save=True,
               s=2)
for i in range(5):
    plot_in_plotly(mapper_arc_cm, "UMAP", dist[:, i], addon="dist_" + str(i),
                   save=True,
                   s=2)
for i in range(5):
    plot_in_plotly(mapper_arc_cm, "UMAP", np.log(dist[:, i]), addon="dist_" + str(i),
                   save=True,
                   s=2)
for i in range(5):
    plot_in_plotly(mapper_arc_cm, "UMAP", -dist[:, i], addon="dist_" + str(i),
                   save=True,
                   s=2)
for i in range(5):
    plot_in_plotly(mapper_arc_cm, "UMAP", -np.log(dist[:, i]), addon="dist_" + str(i),
                   save=True,
                   s=2)
import plotly.graph_objects as go

fig = go.Figure()
method = "UMAP"
df = pd.DataFrame(mapper_arc_cm, columns=[method + '_1', method + '_2', method + '_3'])
for i in range(5):
    df['labels'] = -np.log(dist[:, i])
    fig.add_trace(
        go.Scatter3d(df, x=method + '_1', y=method + '_2', z=method + '_3', color='labels', visible=False,
                     name="𝜈 = " + str(i)))
# Make 10th trace visible
fig.data[0].visible = True
steps = []
for i in range(len(fig.data)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              {"title": "Slider switched to step: " + str(i)}],  # layout attribute
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)
sliders = [dict(
    active=5,
    currentvalue={"prefix": "Frequency: "},
    pad={"t": 50},
    steps=steps
)]
fig.update_layout(
    sliders=sliders
)
fig.show()
trace_list = []
for t in np.arange(0, 30, 1):
    trace_list.extend(
        list(create_traces(t, node_lists)))  # On every timestep t I want to visualize 5 traces.
data = []
for idx, item in enumerate(
        trace_list):  # I believe the key is that the format for data should be as though list1+list2
    data.extend(trace_list[idx])
fig_network = go.Figure(data=data, layout=go.Layout(
    title='Virus spread in a Network',
    titlefont_size=16,
    hovermode='closest',
    margin=dict(b=20, l=5, r=5, t=40),
    scene=dict(
        xaxis=dict(axis),
        yaxis=dict(axis),
        zaxis=dict(axis),
    )))
steps = []
i = 0
day = 0
while i < len(fig_network.data):
    step = dict(method='update',
                args=[{"visible": [False] * len(fig_network.data)},
                      # set all traces on invisible, and only the traces you want visible you put visible.
                      ]
                )
    step['args'][0]['visible'][
        i] = True  # In my case a day is 5 traces, the first of 5 I don't want to see, the other 4 I do want to see. Every trace that is not True in that step is not visible.
    fig_network.data[i]['showlegend'] = False  # the edge trace
    i += 1
    step['args'][0]['visible'][i] = True
    fig_network.data[i]['showlegend'] = True
    i += 1
    step['args'][0]['visible'][i] = True
    fig_network.data[i]['showlegend'] = True
    i += 1
    step['args'][0]['visible'][i] = True
    fig_network.data[i]['showlegend'] = True
    i += 1
    step['args'][0]['visible'][i] = True
    fig_network.data[i]['showlegend'] = True
    i += 1
    steps.append(step)
    day += 1
sliders = [dict(active=0, steps=steps)]
fig_network.layout.update(sliders=sliders)
method = "UMAP"
df = pd.DataFrame(mapper_arc_cm, columns=[method + '_1', method + '_2', method + '_3'])
for i in range(5):
    df['labels'] = -np.log(dist[:, i])
    fig.add_trace(
        go.Scatter3d(x=mapper_arc_cm[:, 0], y=mapper_arc_cm[:, 1], z=mapper_arc_cm[:, 2], visible=False,
                     name="𝜈 = " + str(i)))
# Make 10th trace visible
fig.data[0].visible = True
steps = []
for i in range(len(fig.data)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              {"title": "Slider switched to step: " + str(i)}],  # layout attribute
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)
sliders = [dict(
    active=5,
    currentvalue={"prefix": "Frequency: "},
    pad={"t": 50},
    steps=steps
)]
fig.update_layout(
    sliders=sliders
)
fig.show()
method = "UMAP"
df = pd.DataFrame(mapper_arc_cm, columns=[method + '_1', method + '_2', method + '_3'])
for i in range(5):
    df['labels'] = -np.log(dist[:, i])
    fig.add_trace(
        go.Scatter3d(x=mapper_arc_cm[:, 0], y=mapper_arc_cm[:, 1], z=mapper_arc_cm[:, 2], visible=False,
                     name="𝜈 = " + str(i)))
# Make 10th trace visible
fig.data[0].visible = True
steps = []
for i in range(len(fig.data)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              {"title": "Slider switched to step: " + str(i)}],  # layout attribute
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)
sliders = [dict(
    active=5,
    currentvalue={"prefix": "Frequency: "},
    pad={"t": 50},
    steps=steps
)]
fig.update_layout(
    sliders=sliders
)
fig.write_html("sparse umap plots/" + method + " Plot sf" + ".html")
method = "UMAP"
df = pd.DataFrame(mapper_arc_cm, columns=[method + '_1', method + '_2', method + '_3'])
for i in range(5):
    df['labels'] = -np.log(dist[:, i])
    fig.add_trace(
        go.Scatter3d(x=mapper_arc_cm[:, 0], y=mapper_arc_cm[:, 1], z=mapper_arc_cm[:, 2], visible=False,
                     name="𝜈 = " + str(i), c=dist[:, i]))
# Make 10th trace visible
fig.data[0].visible = True
steps = []
for i in range(len(fig.data)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              {"title": "Slider switched to step: " + str(i)}],  # layout attribute
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)
sliders = [dict(
    active=5,
    currentvalue={"prefix": "Frequency: "},
    pad={"t": 50},
    steps=steps
)]
fig.update_layout(
    sliders=sliders
)
fig.write_html("sparse umap plots/" + method + " Plot sf" + ".html")
method = "UMAP"
df = pd.DataFrame(mapper_arc_cm, columns=[method + '_1', method + '_2', method + '_3'])
for i in range(5):
    df['labels'] = -np.log(dist[:, i])
    fig.add_trace(
        go.Scatter3d(x=mapper_arc_cm[:, 0], y=mapper_arc_cm[:, 1], z=mapper_arc_cm[:, 2], visible=False,
                     name="𝜈 = " + str(i), color=dist[:, i]))
# Make 10th trace visible
fig.data[0].visible = True
steps = []
for i in range(len(fig.data)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              {"title": "Slider switched to step: " + str(i)}],  # layout attribute
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)
sliders = [dict(
    active=5,
    currentvalue={"prefix": "Frequency: "},
    pad={"t": 50},
    steps=steps
)]
fig.update_layout(
    sliders=sliders
)
fig.write_html("sparse umap plots/" + method + " Plot sf" + ".html")
method = "UMAP"
df = pd.DataFrame(mapper_arc_cm, columns=[method + '_1', method + '_2', method + '_3'])
for i in range(5):
    df['labels'] = -np.log(dist[:, i])
    fig.add_trace(
        go.Scatter3d(x=mapper_arc_cm[:, 0], y=mapper_arc_cm[:, 1], z=mapper_arc_cm[:, 2], visible=False,
                     name="𝜈 = " + str(i), surfacecolor=dist[:, i]))
# Make 10th trace visible
fig.data[0].visible = True
steps = []
for i in range(len(fig.data)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              {"title": "Slider switched to step: " + str(i)}],  # layout attribute
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)
sliders = [dict(
    active=5,
    currentvalue={"prefix": "Frequency: "},
    pad={"t": 50},
    steps=steps
)]
fig.update_layout(
    sliders=sliders
)
fig.write_html("sparse umap plots/" + method + " Plot sf" + ".html")
method = "UMAP"
df = pd.DataFrame(mapper_arc_cm, columns=[method + '_1', method + '_2', method + '_3'])
for i in range(5):
    df['labels'] = -np.log(dist[:, i])
    fig.add_trace(
        go.Scatter3d(x=mapper_arc_cm[:, 0], y=mapper_arc_cm[:, 1], z=mapper_arc_cm[:, 2], visible=False,
                     name="𝜈 = " + str(i), lables=dist[:, 1]))
# Make 10th trace visible
fig.data[0].visible = True
steps = []
for i in range(len(fig.data)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              {"title": "Slider switched to step: " + str(i)}],  # layout attribute
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)
sliders = [dict(
    active=5,
    currentvalue={"prefix": "Frequency: "},
    pad={"t": 50},
    steps=steps
)]
fig.update_layout(
    sliders=sliders
)
fig.write_html("sparse umap plots/" + method + " Plot sf" + ".html")
method = "UMAP"
df = pd.DataFrame(mapper_arc_cm, columns=[method + '_1', method + '_2', method + '_3'])
for i in range(5):
    df['labels'] = -np.log(dist[:, i])
    fig.add_trace(
        go.Scatter3d(x=mapper_arc_cm[:, 0], y=mapper_arc_cm[:, 1], z=mapper_arc_cm[:, 2], visible=False,
                     name="𝜈 = " + str(i), mode='markers',
                     marker=dict(
                         size=12,
                         color=dist[:, i],  # set color to an array/list of desired values
                         colorscale='Viridis',  # choose a colorscale
                         opacity=0.8
                     )))
# Make 10th trace visible
fig.data[0].visible = True
steps = []
for i in range(len(fig.data)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              {"title": "Slider switched to step: " + str(i)}],  # layout attribute
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)
sliders = [dict(
    active=5,
    currentvalue={"prefix": "Frequency: "},
    pad={"t": 50},
    steps=steps
)]
fig.update_layout(
    sliders=sliders
)
fig.write_html("sparse umap plots/" + method + " Plot sf" + ".html")
method = "UMAP"
df = pd.DataFrame(mapper_arc_cm, columns=[method + '_1', method + '_2', method + '_3'])
for i in range(5):
    df['labels'] = -np.log(dist[:, i])
    fig.add_trace(
        go.Scatter3d(x=mapper_arc_cm[:, 0], y=mapper_arc_cm[:, 1], z=mapper_arc_cm[:, 2], visible=False,
                     name="𝜈 = " + str(i), mode='markers',
                     marker=dict(
                         size=2,
                         color=dist[:, i],  # set color to an array/list of desired values
                         colorscale='Viridis',  # choose a colorscale
                         opacity=0.8
                     )))
# Make 10th trace visible
fig.data[0].visible = True
steps = []
for i in range(len(fig.data)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              {"title": "Slider switched to step: " + str(i)}],  # layout attribute
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)
sliders = [dict(
    active=5,
    currentvalue={"prefix": "Frequency: "},
    pad={"t": 50},
    steps=steps
)]
fig.update_layout(
    sliders=sliders
)
fig.write_html("sparse umap plots/" + method + " Plot sf" + ".html")
fig = go.Figure()
method = "UMAP"
df = pd.DataFrame(mapper_arc_cm, columns=[method + '_1', method + '_2', method + '_3'])
for i in range(5):
    df['labels'] = -np.log(dist[:, i])
    fig.add_trace(
        go.Scatter3d(x=mapper_arc_cm[:, 0], y=mapper_arc_cm[:, 1], z=mapper_arc_cm[:, 2], visible=False,
                     name="𝜈 = " + str(i), mode='markers',
                     marker=dict(
                         size=2,
                         color=dist[:, i],  # set color to an array/list of desired values
                         colorscale='Viridis',  # choose a colorscale
                         opacity=0.8
                     )))
# Make 10th trace visible
fig.data[0].visible = True
steps = []
for i in range(len(fig.data)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              {"title": "Slider switched to step: " + str(i)}],  # layout attribute
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)
sliders = [dict(
    active=5,
    currentvalue={"prefix": "Frequency: "},
    pad={"t": 50},
    steps=steps
)]
fig.update_layout(
    sliders=sliders
)
fig.write_html("sparse umap plots/" + method + " Plot sf" + ".html")
fig = go.Figure()
method = "UMAP"
df = pd.DataFrame(mapper_arc_cm, columns=[method + '_1', method + '_2', method + '_3'])
for i in range(5):
    df['labels'] = -np.log(dist[:, i])
    fig.add_trace(
        go.Scatter3d(x=mapper_arc_cm[:, 0], y=mapper_arc_cm[:, 1], z=mapper_arc_cm[:, 2], visible=False,
                     name="Distance from Archetype:" + str(i), mode='markers',
                     marker=dict(
                         size=2,
                         color=dist[:, i],  # set color to an array/list of desired values
                         colorscale='Viridis',  # choose a colorscale
                         opacity=0.8
                     )))
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
fig = go.Figure()
method = "UMAP"
df = pd.DataFrame(mapper_arc_cm, columns=[method + '_1', method + '_2', method + '_3'])
for i in range(5):
    fig.add_trace(
        go.Scatter3d(x=mapper_arc_cm[:, 0], y=mapper_arc_cm[:, 1], z=mapper_arc_cm[:, 2], visible=False,
                     name="Distance from Archetype:" + str(i), mode='markers',
                     marker=dict(
                         size=2,
                         color=np.log(dist[:, i]),  # set color to an array/list of desired values
                         colorscale='Viridis',  # choose a colorscale
                         opacity=0.8
                     )))
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
fig.write_html("sparse umap plots/" + method + " Plot cool logged" + ".html")
fig = go.Figure()
method = "UMAP"
df = pd.DataFrame(mapper_arc_cm, columns=[method + '_1', method + '_2', method + '_3'])
for i in range(5):
    fig.add_trace(
        go.Scatter3d(x=mapper_arc_cm[:, 0], y=mapper_arc_cm[:, 1], z=mapper_arc_cm[:, 2], visible=False,
                     name="Distance from Archetype:" + str(i), mode='markers',
                     marker=dict(
                         template="plotly_dark",
                         size=2,
                         color=np.log(dist[:, i]),  # set color to an array/list of desired values
                         colorscale='Viridis',  # choose a colorscale
                         opacity=0.8
                     )))
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
fig.write_html("sparse umap plots/" + method + " Plot cool logged" + ".html")
fig = go.Figure()
method = "UMAP"
df = pd.DataFrame(mapper_arc_cm, columns=[method + '_1', method + '_2', method + '_3'])
for i in range(5):
    fig.add_trace(
        go.Scatter3d(x=mapper_arc_cm[:, 0], y=mapper_arc_cm[:, 1], z=mapper_arc_cm[:, 2], visible=False,
                     name="Distance from Archetype:" + str(i), mode='markers', template="plotly_dark",
                     marker=dict(
                         size=2,
                         color=np.log(dist[:, i]),  # set color to an array/list of desired values
                         colorscale='Viridis',  # choose a colorscale
                         opacity=0.8
                     )))
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
fig.write_html("sparse umap plots/" + method + " Plot cool logged" + ".html")
fig = go.Figure()
method = "UMAP"
df = pd.DataFrame(mapper_arc_cm, columns=[method + '_1', method + '_2', method + '_3'])
for i in range(5):
    fig.add_trace(
        go.Scatter3d(x=mapper_arc_cm[:, 0], y=mapper_arc_cm[:, 1], z=mapper_arc_cm[:, 2], visible=False,
                     name="Distance from Archetype:" + str(i), mode='markers',
                     marker=dict(
                         size=2,
                         color=np.log(dist[:, i]),  # set color to an array/list of desired values
                         colorscale='Viridis',  # choose a colorscale
                         opacity=0.8
                     )))
    fig.update_layout(
        title=method + " Plot ",
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
fig.write_html("sparse umap plots/" + method + " Plot cool logged" + ".html")
fig = go.Figure()
method = "UMAP"
df = pd.DataFrame(mapper_arc_cm, columns=[method + '_1', method + '_2', method + '_3'])
for i in range(5):
    fig.add_trace(
        go.Scatter3d(x=mapper_arc_cm[:, 0], y=mapper_arc_cm[:, 1], z=mapper_arc_cm[:, 2], visible=False,
                     name="Distance from Archetype:" + str(i), mode='markers',
                     marker=dict(
                         size=2,
                         color=dist[:, i],  # set color to an array/list of desired values
                         colorscale='Viridis',  # choose a colorscale
                         opacity=0.8
                     )))
    fig.update_layout(
        title=method + " Plot ",
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
svd_data_50 = TruncatedSVD(n_components=50, n_iter=7, random_state=42).fit_transform(data)
mapper = umap.UMAP(metric='cosine', random_state=42, low_memory=True, n_neighbors=11,
                   min_dist=0.05,
                   n_components=3).fit_transform(svd_data_50)
plot_in_plotly(mapper, "UMAP", g2, addon=" no dense yes pca n_",
               save=True,
               s=2)
winsound.Beep(freq, duration)
for i in [3, 5, 7, 10, 13, 15, 17, 20]:
    for j in [0.05, 0.1, 0.3]:
        mapper = umap.UMAP(metric='cosine', random_state=42, low_memory=True, \
                           n_neighbors=i,
                           min_dist=j,
                           n_components=3).fit_transform(svd_data_50)
        plot_in_plotly(mapper, "UMAP", g2, addon=" no dense yes pca n_" + str(i) + "_d_" + str(j),
                       save=True,
                       s=2)
winsound.Beep(freq, duration)
fig = go.Figure()
method = "UMAP"
df = pd.DataFrame(mapper_arc_cm, columns=[method + '_1', method + '_2', method + '_3'])
for i in range(5):
    fig.add_trace(
        go.Scatter3d(x=mapper_arc_cm[:, 0], y=mapper_arc_cm[:, 1], z=mapper_arc_cm[:, 2], visible=False,
                     name="Distance from Archetype:" + str(i), mode='markers',
                     marker=dict(
                         size=2,
                         color=dist[:, i],  # set color to an array/list of desired values
                         colorscale='Viridis',  # choose a colorscale
                         opacity=0.8
                     )))
    fig.update_layout(
        title=method + " Plot ",
        template="plotly_dark"
    )
    name = 'eye = (x:2, y:2, z:0.1)'
    camera = dict(
        eye=dict(x=2, y=2, z=0.1)
    )
    fig.update_layout(scene_camera=camera, title=name)
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
fig.write_html("sparse umap plots/" + method + " Plot cool_camera" + ".html")
fig = go.Figure()
method = "UMAP"
df = pd.DataFrame(mapper_arc_cm, columns=[method + '_1', method + '_2', method + '_3'])
for i in range(5):
    fig.add_trace(
        go.Scatter3d(x=mapper_arc_cm[:, 0], y=mapper_arc_cm[:, 1], z=mapper_arc_cm[:, 2], visible=False,
                     name="Distance from Archetype:" + str(i), mode='markers',
                     marker=dict(
                         size=2,
                         color=dist[:, i],  # set color to an array/list of desired values
                         colorscale='Viridis',  # choose a colorscale
                         opacity=0.8
                     )))
    fig.update_layout(
        title=method + " Plot ",
        template="plotly_dark"
    )
    name = 'eye = (x:2, y:2, z:0.1)'
    camera = dict(
        eye=dict(x=2, y=2, z=0.1)
    )
    fig.update_layout(scene_camera=camera, title=name)
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
fig.write_image("sparse umap plots/" + method + " Plot cool_camera1" + ".html")
fig = go.Figure()
method = "UMAP"
df = pd.DataFrame(mapper_arc_cm, columns=[method + '_1', method + '_2', method + '_3'])
for i in range(5):
    fig.add_trace(
        go.Scatter3d(x=mapper_arc_cm[:, 0], y=mapper_arc_cm[:, 1], z=mapper_arc_cm[:, 2], visible=False,
                     name="Distance from Archetype:" + str(i), mode='markers',
                     marker=dict(
                         size=2,
                         color=dist[:, i],  # set color to an array/list of desired values
                         colorscale='Viridis',  # choose a colorscale
                         opacity=0.8
                     )))
    fig.update_layout(
        title=method + " Plot ",
        template="plotly_dark"
    )
    name = 'eye = (x:2, y:2, z:0.1)'
    camera = dict(
        eye=dict(x=2, y=2, z=0.1)
    )
    fig.update_layout(scene_camera=camera, title=name)
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
fig.write_image("sparse umap plots/" + method + " Plot cool_camera1" + ".png")
fig = go.Figure()
method = "UMAP"
df = pd.DataFrame(mapper_arc_cm, columns=[method + '_1', method + '_2', method + '_3'])
for i in range(5):
    fig.add_trace(
        go.Scatter3d(x=mapper_arc_cm[:, 0], y=mapper_arc_cm[:, 1], z=mapper_arc_cm[:, 2], visible=False,
                     name="Distance from Archetype:" + str(i), mode='markers',
                     marker=dict(
                         size=2,
                         color=dist[:, i],  # set color to an array/list of desired values
                         colorscale='Viridis',  # choose a colorscale
                         opacity=0.8
                     )))
    fig.update_layout(
        title=method + " Plot ",
        template="plotly_dark",
        showgrid=FALSE
    )
    name = 'eye = (x:2, y:2, z:0.1)'
    camera = dict(
        eye=dict(x=2, y=2, z=0.1)
    )
    fig.update_layout(scene_camera=camera, title=name)
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
fig.write_html("sparse umap plots/" + method + " Plot cool_camera" + ".html")
fig = go.Figure()
method = "UMAP"
df = pd.DataFrame(mapper_arc_cm, columns=[method + '_1', method + '_2', method + '_3'])
for i in range(5):
    fig.add_trace(
        go.Scatter3d(x=mapper_arc_cm[:, 0], y=mapper_arc_cm[:, 1], z=mapper_arc_cm[:, 2], visible=False,
                     name="Distance from Archetype:" + str(i), mode='markers',
                     marker=dict(
                         size=2,
                         color=dist[:, i],  # set color to an array/list of desired values
                         colorscale='Viridis',  # choose a colorscale
                         opacity=0.8
                     )))
    fig.update_layout(
        title=method + " Plot ",
        template="plotly_dark",
        showgrid=False
    )
    name = 'eye = (x:2, y:2, z:0.1)'
    camera = dict(
        eye=dict(x=2, y=2, z=0.1)
    )
    fig.update_layout(scene_camera=camera, title=name)
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
fig.write_html("sparse umap plots/" + method + " Plot cool_camera" + ".html")
fig = go.Figure()
method = "UMAP"
df = pd.DataFrame(mapper_arc_cm, columns=[method + '_1', method + '_2', method + '_3'])
for i in range(5):
    fig.add_trace(
        go.Scatter3d(x=mapper_arc_cm[:, 0], y=mapper_arc_cm[:, 1], z=mapper_arc_cm[:, 2], visible=False,
                     name="Distance from Archetype:" + str(i), mode='markers',
                     marker=dict(
                         size=2,
                         color=dist[:, i],  # set color to an array/list of desired values
                         colorscale='Viridis',  # choose a colorscale
                         opacity=0.8
                     )))
    fig.update_layout(
        title=method + " Plot ",
        template="plotly_dark",
        grid=False
    )
    name = 'eye = (x:2, y:2, z:0.1)'
    camera = dict(
        eye=dict(x=2, y=2, z=0.1)
    )
    fig.update_layout(scene_camera=camera, title=name)
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
fig.write_html("sparse umap plots/" + method + " Plot cool_camera" + ".html")
fig = go.Figure()
method = "UMAP"
df = pd.DataFrame(mapper_arc_cm, columns=[method + '_1', method + '_2', method + '_3'])
for i in range(5):
    fig.add_trace(
        go.Scatter3d(x=mapper_arc_cm[:, 0], y=mapper_arc_cm[:, 1], z=mapper_arc_cm[:, 2], visible=False,
                     name="Distance from Archetype:" + str(i), mode='markers',
                     marker=dict(
                         size=2,
                         color=dist[:, i],  # set color to an array/list of desired values
                         colorscale='Viridis',  # choose a colorscale
                         opacity=0.8
                     )))
    fig.update_layout(
        title=method + " Plot ",
        template="plotly_dark",
        scene=dict(
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            zaxis=dict(showgrid=False))
    )

    name = 'eye = (x:2, y:2, z:0.1)'
    camera = dict(
        eye=dict(x=2, y=2, z=0.1)
    )
    fig.update_layout(scene_camera=camera, title=name)
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
fig.write_html("sparse umap plots/" + method + " Plot cool_camera" + ".html")
fig = go.Figure()
method = "UMAP"
df = pd.DataFrame(mapper_arc_cm, columns=[method + '_1', method + '_2', method + '_3'])
for i in range(5):
    fig.add_trace(
        go.Scatter3d(x=mapper_arc_cm[:, 0], y=mapper_arc_cm[:, 1], z=mapper_arc_cm[:, 2], visible=False,
                     name="Distance from Archetype:" + str(i), mode='markers',
                     marker=dict(
                         size=2,
                         color=dist[:, i],  # set color to an array/list of desired values
                         colorscale='Viridis',  # choose a colorscale
                         opacity=0.8
                     )))
    fig.update_layout(
        title=method + " Plot ",
        template="plotly_dark",
        scene=dict(
            xaxis=dict(showgrid=False, showaxis=False),
            yaxis=dict(showgrid=False),
            zaxis=dict(showgrid=False))
    )

    name = 'eye = (x:2, y:2, z:0.1)'
    camera = dict(
        eye=dict(x=2, y=2, z=0.1)
    )
    fig.update_layout(scene_camera=camera, title=name)
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
fig.write_html("sparse umap plots/" + method + " Plot cool_camera" + ".html")
fig = go.Figure()
method = "UMAP"
df = pd.DataFrame(mapper_arc_cm, columns=[method + '_1', method + '_2', method + '_3'])
for i in range(5):
    fig.add_trace(
        go.Scatter3d(x=mapper_arc_cm[:, 0], y=mapper_arc_cm[:, 1], z=mapper_arc_cm[:, 2], visible=False,
                     name="Distance from Archetype:" + str(i), mode='markers',
                     marker=dict(
                         size=2,
                         color=dist[:, i],  # set color to an array/list of desired values
                         colorscale='Viridis',  # choose a colorscale
                         opacity=0.8
                     )))
    fig.update_layout(
        title=method + " Plot ",
        template="plotly_dark",
        scene=dict(
            xaxis=dict(showgrid=False, axis=False),
            yaxis=dict(showgrid=False),
            zaxis=dict(showgrid=False))
    )

    name = 'eye = (x:2, y:2, z:0.1)'
    camera = dict(
        eye=dict(x=2, y=2, z=0.1)
    )
    fig.update_layout(scene_camera=camera, title=name)
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
fig.write_html("sparse umap plots/" + method + " Plot cool_camera" + ".html")
fig = go.Figure()
method = "UMAP"
df = pd.DataFrame(mapper_arc_cm, columns=[method + '_1', method + '_2', method + '_3'])
for i in range(5):
    fig.add_trace(
        go.Scatter3d(x=mapper_arc_cm[:, 0], y=mapper_arc_cm[:, 1], z=mapper_arc_cm[:, 2], visible=False,
                     name="Distance from Archetype:" + str(i), mode='markers',
                     marker=dict(
                         size=2,
                         color=dist[:, i],  # set color to an array/list of desired values
                         colorscale='Viridis',  # choose a colorscale
                         opacity=0.8
                     )))
    fig.update_layout(
        title=method + " Plot ",
        template="plotly_dark",
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            zaxis=dict(showgrid=False, showticklabels=False))
    )

    name = 'eye = (x:2, y:2, z:0.1)'
    camera = dict(
        eye=dict(x=2, y=2, z=0.1)
    )
    fig.update_layout(scene_camera=camera, title=name)
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
fig.write_html("sparse umap plots/" + method + " Plot cool_camera" + ".html")
fig = go.Figure()
method = "UMAP"
df = pd.DataFrame(mapper_arc_cm, columns=[method + '_1', method + '_2', method + '_3'])
for i in range(5):
    fig.add_trace(
        go.Scatter3d(x=mapper_arc_cm[:, 0], y=mapper_arc_cm[:, 1], z=mapper_arc_cm[:, 2], visible=False,
                     name="Distance from Archetype:" + str(i), mode='markers',
                     marker=dict(
                         size=2,
                         color=dist[:, i],  # set color to an array/list of desired values
                         colorscale='Viridis',  # choose a colorscale
                         opacity=0.8
                     )))
    fig.update_layout(
        title=method + " Plot ",
        template="plotly_dark",
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, ticks='', ),
            yaxis=dict(showgrid=False, showticklabels=False, ticks='', ),
            zaxis=dict(showgrid=False, showticklabels=False, ticks=''))
    )

    name = 'eye = (x:2, y:2, z:0.1)'
    camera = dict(
        eye=dict(x=2, y=2, z=0.1)
    )
    fig.update_layout(scene_camera=camera, title=name)
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
fig.write_html("sparse umap plots/" + method + " Plot cool_camera" + ".html")
fig = go.Figure()
method = "UMAP"
df = pd.DataFrame(mapper_arc_cm, columns=[method + '_1', method + '_2', method + '_3'])
for i in range(5):
    fig.add_trace(
        go.Scatter3d(x=mapper_arc_cm[:, 0], y=mapper_arc_cm[:, 1], z=mapper_arc_cm[:, 2], visible=False,
                     name="Distance from Archetype:" + str(i), mode='markers',
                     marker=dict(
                         size=2,
                         color=dist[:, i],  # set color to an array/list of desired values
                         colorscale='Viridis',  # choose a colorscale
                         opacity=0.8
                     )))
    fig.update_layout(
        title=method + " Plot ",
        template="plotly_dark",
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, showticklabels=False, title=''),
            zaxis=dict(showgrid=False, showticklabels=False, title=''))
    )

    name = 'eye = (x:2, y:2, z:0.1)'
    camera = dict(
        eye=dict(x=2, y=2, z=0.1)
    )
    fig.update_layout(scene_camera=camera, title=name)
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
fig.write_html("sparse umap plots/" + method + " Plot cool_camera" + ".html")
fig = go.Figure()
method = "UMAP"
df = pd.DataFrame(mapper_arc_cm, columns=[method + '_1', method + '_2', method + '_3'])
for i in range(5):
    fig.add_trace(
        go.Scatter3d(x=mapper_arc_cm[:, 0], y=mapper_arc_cm[:, 1], z=mapper_arc_cm[:, 2], visible=False,
                     name="Distance from Archetype:" + str(i), mode='markers',
                     marker=dict(
                         size=2,
                         color=dist[:, i],  # set color to an array/list of desired values
                         colorscale='Viridis',  # choose a colorscale
                         opacity=0.8
                     )))
    fig.update_layout(
        title=method + " Plot ",
        template="plotly_dark",
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, showticklabels=False, title=''),
            zaxis=dict(showgrid=False, showticklabels=False, title=''))
    )

    name = 'Showing Distance From Archetype:' + str(i)
    camera = dict(
        eye=dict(x=2, y=2, z=0.1)
    )
    fig.update_layout(scene_camera=camera, title=name)
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
fig.write_html("sparse umap plots/" + method + " Plot cool_camera" + ".html")
import plotly.graph_objects as go

fig = go.Figure()
method = "UMAP"
df = pd.DataFrame(mapper_arc_cm, columns=[method + '_1', method + '_2', method + '_3'])
for i in range(5):
    fig.add_trace(
        go.Scatter3d(x=mapper_arc_cm[:, 0], y=mapper_arc_cm[:, 1], z=mapper_arc_cm[:, 2], visible=False,
                     name="Distance from Archetype:" + str(i), mode='markers',
                     marker=dict(
                         size=2,
                         color=dist[:, i],  # set color to an array/list of desired values
                         colorscale='Viridis',  # choose a colorscale
                         opacity=0.8
                     )))
    fig.update_layout(
        title=method + " Plot ",
        template="plotly_dark",
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, showticklabels=False, title=''),
            zaxis=dict(showgrid=False, showticklabels=False, title=''))
    )

    name = 'Showing Distance From Archetype:' + str(i)
    camera = dict(
        eye=dict(x=2, y=2, z=0.1)
    )
    fig.update_layout(scene_camera=camera, title=name)
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
fig.write_image("sparse umap plots/" + method + " Plot cool_camera im" + ".png")
fig = go.Figure()
method = "UMAP"
df = pd.DataFrame(mapper_arc_cm, columns=[method + '_1', method + '_2', method + '_3'])
for i in range(5):
    fig.add_trace(
        go.Scatter3d(x=mapper_arc_cm[:, 0], y=mapper_arc_cm[:, 1], z=mapper_arc_cm[:, 2], visible=False,
                     name="Distance from Archetype:" + str(i), mode='markers',
                     marker=dict(
                         size=2,
                         color=dist[:, i],  # set color to an array/list of desired values
                         colorscale='Viridis',  # choose a colorscale
                         opacity=0.8
                     )))
    fig.update_layout(
        title=method + " Plot ",
        template="plotly_dark",
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, showticklabels=False, title=''),
            zaxis=dict(showgrid=False, showticklabels=False, title=''))
    )

    name = 'Showing Distance From Archetype:' + str(i)
    camera = dict(
        eye=dict(x=2, y=2, z=0.1)
    )
    fig.update_layout(scene_camera=camera, title=name)
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
fig.write_image("sparse umap plots/" + method + " Plot cool_camera im" + ".png")
fig = go.Figure()
method = "UMAP"
df = pd.DataFrame(mapper_arc_cm, columns=[method + '_1', method + '_2', method + '_3'])
for i in range(5):
    fig.add_trace(
        go.Scatter3d(x=mapper_arc_cm[:, 0], y=mapper_arc_cm[:, 1], z=mapper_arc_cm[:, 2], visible=False,
                     name="Distance from Archetype:" + str(i), mode='markers',
                     marker=dict(
                         size=2,
                         color=dist[:, i],  # set color to an array/list of desired values
                         colorscale='Viridis',  # choose a colorscale
                         opacity=0.8
                     )))
    fig.update_layout(
        title=method + " Plot ",
        template="plotly_dark",
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, showticklabels=False, title=''),
            zaxis=dict(showgrid=False, showticklabels=False, title=''))
    )

    name = 'Showing Distance From Archetype:' + str(i)
    camera = dict(
        eye=dict(x=2, y=2, z=0.1)
    )
    fig.update_layout(scene_camera=camera, title=name)
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
fig.data[0].visible = True
fig.write_image("sparse umap plots/" + method + " Plot cool_camera im" + ".png")
fig = go.Figure()
method = "UMAP"
df = pd.DataFrame(mapper_arc_cm, columns=[method + '_1', method + '_2', method + '_3'])
for i in range(5):
    fig.add_trace(
        go.Scatter3d(x=mapper_arc_cm[:, 0], y=mapper_arc_cm[:, 1], z=mapper_arc_cm[:, 2], visible=False,
                     name="Distance from Archetype:" + str(i), mode='markers',
                     marker=dict(
                         size=2,
                         color=dist[:, i],  # set color to an array/list of desired values
                         colorscale='Viridis',  # choose a colorscale
                         opacity=0.8
                     )))
    fig.update_layout(
        title=method + " Plot ",
        template="plotly_dark",
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, showticklabels=False, title=''),
            zaxis=dict(showgrid=False, showticklabels=False, title=''))
    )

    name = 'Showing Distance From Archetype:' + str(i)
    camera = dict(
        eye=dict(x=2, y=2, z=0.1)
    )
    fig.update_layout(scene_camera=camera, title=name)
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
fig.data[1].visible = True
fig.write_image("sparse umap plots/" + method + " Plot cool_camera im2" + ".png")
fig = go.Figure()
method = "UMAP"
df = pd.DataFrame(mapper_arc_cm, columns=[method + '_1', method + '_2', method + '_3'])
for i in range(5):
    fig.add_trace(
        go.Scatter3d(x=mapper_arc_cm[:, 0], y=mapper_arc_cm[:, 1], z=mapper_arc_cm[:, 2], visible=False,
                     name="Distance from Archetype:" + str(i), mode='markers',
                     marker=dict(
                         size=2,
                         color=dist[:, i],  # set color to an array/list of desired values
                         colorscale='Viridis',  # choose a colorscale
                         opacity=0.8
                     )))
    fig.update_layout(
        title=method + " Plot ",
        template="plotly_dark",
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, showticklabels=False, title=''),
            zaxis=dict(showgrid=False, showticklabels=False, title=''))
    )

    name = 'Showing Distance From Archetype:' + str(i)
    camera = dict(
        eye=dict(x=2, y=2, z=0.1)
    )
    fig.update_layout(scene_camera=camera, title=name)
# Make 10th trace visible
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
fig.data[1].visible = True
fig.write_image("sparse umap plots/" + method + " Plot cool_camera im2" + ".png")
fig = go.Figure()
method = "UMAP"
df = pd.DataFrame(mapper_arc_cm, columns=[method + '_1', method + '_2', method + '_3'])
for i in range(5):
    fig.add_trace(
        go.Scatter3d(x=mapper_arc_cm[:, 0], y=mapper_arc_cm[:, 1], z=mapper_arc_cm[:, 2], visible=False,
                     name="Distance from Archetype:" + str(i), mode='markers',
                     marker=dict(
                         size=2,
                         color=dist[:, i],  # set color to an array/list of desired values
                         colorscale='Viridis',  # choose a colorscale
                         opacity=0.8
                     )))
    fig.update_layout(
        title=method + " Plot ",
        template="plotly_dark",
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, showticklabels=False, title=''),
            zaxis=dict(showgrid=False, showticklabels=False, title=''))
    )

    name = 'Showing Distance From Archetype:' + str(i)
    camera = dict(
        eye=dict(x=1, y=1, z=0.05)
    )
    fig.update_layout(scene_camera=camera, title=name)
# Make 10th trace visible
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
fig.data[1].visible = True
fig.write_image("sparse umap plots/" + method + " Plot cool_camera im2" + ".png")
fig = go.Figure()
method = "UMAP"
df = pd.DataFrame(mapper_arc_cm, columns=[method + '_1', method + '_2', method + '_3'])
for i in range(5):
    fig.add_trace(
        go.Scatter3d(x=mapper_arc_cm[:, 0], y=mapper_arc_cm[:, 1], z=mapper_arc_cm[:, 2], visible=False,
                     name="Distance from Archetype:" + str(i), mode='markers',
                     marker=dict(
                         size=2,
                         color=dist[:, i],  # set color to an array/list of desired values
                         colorscale='Viridis',  # choose a colorscale
                         opacity=0.8
                     )))
    fig.update_layout(
        title=method + " Plot ",
        template="plotly_dark",
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, showticklabels=False, title=''),
            zaxis=dict(showgrid=False, showticklabels=False, title=''))
    )

    name = 'Showing Distance From Archetype:' + str(i)
    camera = dict(
        eye=dict(x=1 / 2, y=1 / 2, z=0.05 / 2)
    )
    fig.update_layout(scene_camera=camera, title=name)
# Make 10th trace visible
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
fig.data[1].visible = True
fig.write_image("sparse umap plots/" + method + " Plot cool_camera im2" + ".png")
fig = go.Figure()
method = "UMAP"
df = pd.DataFrame(mapper_arc_cm, columns=[method + '_1', method + '_2', method + '_3'])
for i in range(5):
    fig.add_trace(
        go.Scatter3d(x=mapper_arc_cm[:, 0], y=mapper_arc_cm[:, 1], z=mapper_arc_cm[:, 2], visible=False,
                     name="Distance from Archetype:" + str(i), mode='markers',
                     marker=dict(
                         size=2,
                         color=dist[:, i],  # set color to an array/list of desired values
                         colorscale='Viridis',  # choose a colorscale
                         opacity=0.8
                     )))
    fig.update_layout(
        title=method + " Plot ",
        template="plotly_dark",
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, showticklabels=False, title=''),
            zaxis=dict(showgrid=False, showticklabels=False, title=''))
    )

    name = 'Showing Distance From Archetype:' + str(i)
    camera = dict(
        eye=dict(x=1, y=1, z=0.05)
    )
    fig.update_layout(scene_camera=camera, title=name)
# Make 10th trace visible
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
fig.data[1].visible = True
fig.write_image("sparse umap plots/" + method + " Plot cool_camera im2" + ".png")
for k in range(0, 10, 0.1):
    fig = go.Figure()
    method = "UMAP"
    df = pd.DataFrame(mapper_arc_cm, columns=[method + '_1', method + '_2', method + '_3'])
    for i in range(5):
        fig.add_trace(
            go.Scatter3d(x=mapper_arc_cm[:, 0], y=mapper_arc_cm[:, 1], z=mapper_arc_cm[:, 2], visible=False,
                         name="Distance from Archetype:" + str(i), mode='markers',
                         marker=dict(
                             size=2,
                             color=dist[:, i],  # set color to an array/list of desired values
                             colorscale='Viridis',  # choose a colorscale
                             opacity=0.8
                         )))
        fig.update_layout(
            title=method + " Plot ",
            template="plotly_dark",
            scene=dict(
                xaxis=dict(showgrid=False, showticklabels=False, title=''),
                yaxis=dict(showgrid=False, showticklabels=False, title=''),
                zaxis=dict(showgrid=False, showticklabels=False, title=''))
        )
        name = 'Showing Distance From Archetype:' + str(i)
        camera = dict(
            eye=dict(x=1 + k, y=1, z=0.05)
        )
        fig.update_layout(scene_camera=camera, title=name)
    # Make 10th trace visible
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
    fig.data[1].visible = True
    fig.write_image("sparse umap plots/" + method + " Plot cool_camera im_" + str(k) + ".png")
range(0, 10, 0.1)
for k in np.arange(0.0, 10.0, 0.1):
    fig = go.Figure()
    method = "UMAP"
    df = pd.DataFrame(mapper_arc_cm, columns=[method + '_1', method + '_2', method + '_3'])
    for i in range(5):
        fig.add_trace(
            go.Scatter3d(x=mapper_arc_cm[:, 0], y=mapper_arc_cm[:, 1], z=mapper_arc_cm[:, 2], visible=False,
                         name="Distance from Archetype:" + str(i), mode='markers',
                         marker=dict(
                             size=2,
                             color=dist[:, i],  # set color to an array/list of desired values
                             colorscale='Viridis',  # choose a colorscale
                             opacity=0.8
                         )))
        fig.update_layout(
            title=method + " Plot ",
            template="plotly_dark",
            scene=dict(
                xaxis=dict(showgrid=False, showticklabels=False, title=''),
                yaxis=dict(showgrid=False, showticklabels=False, title=''),
                zaxis=dict(showgrid=False, showticklabels=False, title=''))
        )
        name = 'Showing Distance From Archetype:' + str(i)
        camera = dict(
            eye=dict(x=1 + k, y=1, z=0.05)
        )
        fig.update_layout(scene_camera=camera, title=name)
    # Make 10th trace visible
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
    fig.data[1].visible = True
    fig.write_image("sparse umap plots/" + method + " Plot cool_camera im_" + str(k) + ".png")
import plotly.graph_objects as go
from plotly.offline import iplot
import pandas as pd
import numpy as np

df = df['UMAP_1', 'UMAP_1']
sliders = []
frames = []
lon_range = np.arange(-180, 180, 2)
sliders.append(
    dict(
        active=0,
        currentvalue={"prefix": "Longitude: "},
        pad={"t": 0},
        steps=[{
            'method': 'relayout',
            'label': str(i),
            'args': ['geo.projection.rotation.lon', i]} for i in lon_range]
    )
)
# for i in lon_range:
#     frame = go.Frame(data=[go.Scattergeo(lon=['geo.projection.rotation.lon', i])], name=str(i))
#     frames.append(frame)
fig = go.Figure(
    data=go.Scattergeo(
        lon=df['Longitude'],
        lat=df['Latitude'],
        mode='markers',
        marker_color=df['Staff Required']),
    # frames=frames
)
fig.update_layout(
    title='Most trafficked US airports<br>(Hover for airport names)',
    geo=go.layout.Geo(
        projection_type='orthographic',
        showland=True,
        showcountries=True,
        landcolor='rgb(243, 243, 243)',
        countrycolor='rgb(204, 204, 204)'
    ),
    sliders=sliders,
    updatemenus=[dict(type='buttons',
                      showactive=True,
                      y=1,
                      x=0.8,
                      xanchor='left',
                      yanchor='bottom',
                      pad=dict(t=45, r=10),
                      buttons=[dict(label='Play',
                                    method='animate',
                                    args=[None, dict(frame=dict(duration=50, redraw=False),
                                                     transition=dict(duration=0),
                                                     fromcurrent=True,
                                                     mode='immediate'
                                                     )]
                                    )
                               ]
                      )
                 ]
)
fig.write_html("sparse umap plots/" + method + " Plot cool222" + ".html")
df2 = df['UMAP_1', 'UMAP_1']
df2 = df['UMAP_1']
df2 = df['UMAP_1':'UMAP_2']
df2 = df[:, 'UMAP_1':'UMAP_2']
df2 = df[['UMAP_1', 'UMAP_2']]
df = df2
sliders = []
frames = []
lon_range = np.arange(-180, 180, 2)
sliders.append(
    dict(
        active=0,
        currentvalue={"prefix": "Longitude: "},
        pad={"t": 0},
        steps=[{
            'method': 'relayout',
            'label': str(i),
            'args': ['geo.projection.rotation.lon', i]} for i in lon_range]
    )
)
# for i in lon_range:
#     frame = go.Frame(data=[go.Scattergeo(lon=['geo.projection.rotation.lon', i])], name=str(i))
#     frames.append(frame)
fig = go.Figure(
    data=go.Scattergeo(
        lon=df['Longitude'],
        lat=df['Latitude'],
        mode='markers',
        marker_color=df['Staff Required']),
    # frames=frames
)
fig.update_layout(
    title='Most trafficked US airports<br>(Hover for airport names)',
    geo=go.layout.Geo(
        projection_type='orthographic',
        showland=True,
        showcountries=True,
        landcolor='rgb(243, 243, 243)',
        countrycolor='rgb(204, 204, 204)'
    ),
    sliders=sliders,
    updatemenus=[dict(type='buttons',
                      showactive=True,
                      y=1,
                      x=0.8,
                      xanchor='left',
                      yanchor='bottom',
                      pad=dict(t=45, r=10),
                      buttons=[dict(label='Play',
                                    method='animate',
                                    args=[None, dict(frame=dict(duration=50, redraw=False),
                                                     transition=dict(duration=0),
                                                     fromcurrent=True,
                                                     mode='immediate'
                                                     )]
                                    )
                               ]
                      )
                 ]
)
fig.show()
sliders = []
frames = []
lon_range = np.arange(-180, 180, 2)
sliders.append(
    dict(
        active=0,
        currentvalue={"prefix": "Longitude: "},
        pad={"t": 0},
        steps=[{
            'method': 'relayout',
            'label': str(i),
            'args': ['geo.projection.rotation.lon', i]} for i in lon_range]
    )
)
# for i in lon_range:
#     frame = go.Frame(data=[go.Scattergeo(lon=['geo.projection.rotation.lon', i])], name=str(i))
#     frames.append(frame)
fig = go.Figure(
    data=go.Scattergeo(
        lon=df['UMAP_1'],
        lat=df['UMAP_2'],
        mode='markers',
        marker_color=df['Staff Required']),
    # frames=frames
)
fig.update_layout(
    title='Most trafficked US airports<br>(Hover for airport names)',
    geo=go.layout.Geo(
        projection_type='orthographic',
        showland=True,
        showcountries=True,
        landcolor='rgb(243, 243, 243)',
        countrycolor='rgb(204, 204, 204)'
    ),
    sliders=sliders,
    updatemenus=[dict(type='buttons',
                      showactive=True,
                      y=1,
                      x=0.8,
                      xanchor='left',
                      yanchor='bottom',
                      pad=dict(t=45, r=10),
                      buttons=[dict(label='Play',
                                    method='animate',
                                    args=[None, dict(frame=dict(duration=50, redraw=False),
                                                     transition=dict(duration=0),
                                                     fromcurrent=True,
                                                     mode='immediate'
                                                     )]
                                    )
                               ]
                      )
                 ]
)
fig.show()
history
for k in ["euclidean", "cosine", "correlation", "manhattan", "chebyshev", "minkowski", "wminkowski",
          "hamming",
          "hellinger"]:
    for i in [3, 7, 11, 13, 16, 20]:
        for j in [0.01, 0.1, 0.3, 0.5]:
            m = umap.UMAP(output_metric='gaussian_energy', metric=k,
                          random_state=42,
                          low_memory=True,
                          n_neighbors=i, min_dist=j, n_components=3).fit_transform(merged2)
            plot_in_plotly(m, "UMAP", g4,
                           addon="Gaussian, with n_neighbors: " + str(i) + " and with min_dist: " + str(
                               j) + "\nUnder the metric: " + str(k) + "\nAnd with marked Archetypes (A) and "
                                                                      "centers of masses (CM)",
                           save=True, s=2)
winsound.Beep(freq, duration)

for k in ["cosine", "euclidean", "correlation", "manhattan", "chebyshev", "minkowski", "wminkowski",
          "hamming",
          "hellinger"]:
    for i in [3, 7, 11, 13, 16, 20]:
        for j in [0.01, 0.1, 0.3, 0.5]:
            m = umap.UMAP(output_metric='gaussian_energy', metric=k,
                          random_state=42,
                          low_memory=True,
                          n_neighbors=i, min_dist=j, n_components=3).fit_transform(merged2)
            plot_in_plotly(m, "UMAP", g4,
                           addon="Gaussian, with n_neighbors: " + str(i) + " and with min_dist: " + str(
                               j) + "Under the metric: " + str(k) + "And with marked Archetypes (A) and "
                                                                    "centers of masses (CM)",
                           save=True, s=2)
winsound.Beep(freq, duration)
for k in ["cosine", "euclidean", "correlation", "manhattan", "chebyshev", "minkowski", "wminkowski",
          "hamming",
          "hellinger"]:
    for i in [3, 7, 11, 13, 16, 20]:
        for j in [0.01, 0.1, 0.3, 0.5]:
            m = umap.UMAP(output_metric='gaussian_energy', metric=k,
                          random_state=42,
                          low_memory=True,
                          n_neighbors=i, min_dist=j, n_components=3).fit_transform(merged2)
            plot_in_plotly(m, "UMAP", g4,
                           addon="Gaussian, with n_neighbors " + str(i) + " and with min_dist " + str(
                               j) + "Under the metric " + str(k) + "And with marked Archetypes (A) and "
                                                                   "centers of masses (CM)",
                           save=True, s=2)
winsound.Beep(freq, duration)
for k in ["cosine", "euclidean", "correlation", "manhattan", "chebyshev", "minkowski", "wminkowski",
          "hamming",
          "hellinger"]:
    for i in [3, 7, 11, 13, 16, 20]:
        for j in [0.01, 0.1, 0.3, 0.5]:
            m = umap.UMAP(output_metric='gaussian_energy', metric=k,
                          random_state=42,
                          low_memory=True,
                          n_neighbors=i, min_dist=j, n_components=3).fit_transform(merged2)
            plot_in_plotly(m, "UMAP", g4,
                           addon="metric " + str(k) + " Gaussian, n_neighbors " + str(i)
                                 + " and with min_dist " + str(j), save=True, s=2)
winsound.Beep(freq, duration)
for k in ["hamming", "hellinger"]:
    for i in [3, 7, 11, 13, 16, 20]:
        for j in [0.01, 0.1, 0.3, 0.5]:
            m = umap.UMAP(output_metric='gaussian_energy', metric=k,
                          random_state=42,
                          low_memory=True,
                          n_neighbors=i, min_dist=j, n_components=3).fit_transform(merged2)
            plot_in_plotly(m, "UMAP", g4,
                           addon="metric " + str(k) + " Gaussian, n_neighbors " + str(i)
                                 + " and with min_dist " + str(j), save=True, s=2)
winsound.Beep(freq, duration)
from sklearn.decomposition import LatentDirichletAllocation

from sklearn.preprocessing import normalize

matrix = numpy.arange(0,27,3).reshape(3,3).astype(numpy.float64)
# array([[  0.,   3.,   6.],
#        [  9.,  12.,  15.],
#        [ 18.,  21.,  24.]])

normed_matrix = normalize(matrix, axis=1, norm='l1')