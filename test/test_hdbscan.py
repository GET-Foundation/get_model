#%%
from caesar.io.zarr_io import SingleSparseZarrIO
import cupy
import numpy
import zarr
import kvikio
import kvikio.zarr

z = SingleSparseZarrIO('/home/xf2217/Projects/get_data/sample_64_cerebrum.hg38.fragments.zarr')
# %%
z.get_libsize()
# %%
z.metadata
# %%
import cupy as cp
# %%
z.get_libsize_for_chr_chunk('chr1', 1).shape
#%%
sorted([int(name.split('_')[1]) for name in z.dataset.chrs.chr1])
# %%
import cuml
cuml.set_global_output_type('cudf')
# %%
ary = z.dataset.peak_count.chr1[:]
ary = cp.asarray(ary)
ary = ary>0
ary = ary.astype('float32')

#%%
ary = ary[:, (ary>0).var(0).argsort()[-100:]]
#%%
# TF-IDF normalization
from cuml.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()
ary = tfidf.fit_transform(ary)
# %%
from cuml import TruncatedSVD
from cuml.decomposition import TruncatedSVD
# run TruncatedSVD on ary
svd = TruncatedSVD(n_components=20)
ary_svd = svd.fit_transform(ary.toarray())

# %%
dbscan_float = cuml.HDBSCAN(min_cluster_size=1, alpha=5, allow_single_cluster=True, cluster_selection_epsilon=0)
dbscan_float.fit(ary_svd)
dbscan_float.labels_.value_counts()

# %%
# plot ary_svd
import matplotlib.pyplot as plt
# umap 
umap = cuml.UMAP()
ary_umap = umap.fit_transform(ary_svd).values.get()
#%%
ary_svd_plot = ary_svd.values.get()
plt.scatter(ary_umap[:, 0], ary_umap[:, 1], s=1, c=dbscan_float.labels_.values.get(), cmap='Set3')
# %%
import seaborn as sns
sns.scatterplot(x=ary_umap[:, 0], y=ary_umap[:, 1], hue=z.metadata.labels, palette='Set3')
# %%
chunk_arr = []
for chr in z.chrs:
    for i in sorted([int(name.split('_')[1]) for name in z.dataset.chrs[chr]]):
        chunk_arr.append(z.get_libsize_for_chr_chunk(chr, i))
# %%
import numpy as np
chunk_arr = np.stack(chunk_arr).T
# %%
chunk_arr.shape
# %%
chunk_arr = cp.asarray(chunk_arr)
# %%
#tfidf
from cuml.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()
chunk_arr = tfidf.fit_transform(chunk_arr).toarray()
svd = TruncatedSVD(n_components=100)
chunk_arr_svd = svd.fit_transform(chunk_arr)
# %%
umap = cuml.UMAP(min_dist=0.1)
chunk_arr_umap = umap.fit_transform(chunk_arr_svd).values.get()
# %%
# plot iwthout legend
sns.scatterplot(x=chunk_arr_umap[:, 0], y=chunk_arr_umap[:, 1], hue=z.metadata.labels.str.split(' ').str[1], palette='tab10',s=10, legend=False)
# %%
# knn graph
from cuml import NearestNeighbors
knn = NearestNeighbors(n_neighbors=100)
knn.fit(chunk_arr_svd)
knn_graph = knn.kneighbors_graph(chunk_arr_svd).toarray()
# to numpy 
knn_graph = np.array(knn_graph.get())

#%%
# leiden
from cugraph import leiden, Graph 
G = Graph()
G.from_numpy_array(knn_graph)

leiden_float = leiden(G, resolution=0.1)


# %%
# plot with legend
sns.scatterplot(x=chunk_arr_umap[:, 0], y=chunk_arr_umap[:, 1], hue=z.metadata.labels.str.split(' ').str[1], palette='tab10',s=10, legend=False)
# %%
def plot_track(zio, chrom, start, end, conv_size=100, ax=None, color='blue', ):
    y = zio.get_track(np.where(z.metadata.labels.str.split(' ').str[1]=='Astrocyte')[0], chrom, start, end)#.sum(0)
    x = np.arange(len(y))
    y = np.convolve(y, np.ones(conv_size)/conv_size, mode='same')
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 5))
    # fill area
    ax.plot(x, y, color=color)
    ax.fill_between(x, y, color=color)
    # remove top and right spines
    ax.spines['top'].set_visible(False) 
    ax.spines['right'].set_visible(False)
    # rotate 90 degree y-axis label
    # ax.set_ylabel(traj_key, fontsize=15, rotation=0, labelpad=40, ha='right')
    return ax

plot_track(z, 'chr1', 1000000, 1100000)
# %%
def plot_track(zio, data_query, chrom, start, end, conv_size=20, ax=None, color='blue', libsize=None):
    y = zio.get_track(data_query, chrom, start, end)
    x = np.arange(len(y))
    y = np.convolve(y, np.ones(conv_size)/conv_size, mode='same')
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 5))
    # if libsize is not None:
    #     y = y / libsize * 10000000
    # y = y**2
    # fill area
    # calculate signal to noise ratio
    
    ax.plot(x, y, color=color)
    ax.fill_between(x, y, color=color)
    # remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax


def plot_trajectory(z, chrom, start, end, conv_size=100, palette='mako_r', figsize=(10, 3)):
    color_gradient = sns.color_palette(palette, as_cmap=True)
    m = z.metadata.copy()
    m['labels_broad'] = m.labels.str.split(' ').str[1]
    traj_query = m.groupby('labels_broad').cell_idx.agg(list).to_dict()
    traj_lib_size = m.groupby('labels_broad').per_cell_libsize.sum().to_dict()
    # remove query with length < 100
    traj_query = {k: v for k, v in traj_query.items() if len(v) > 100}
    fig, axes = plt.subplots(
        len(traj_query), 1, figsize=figsize, sharex=True, sharey=True)
    for i, traj_key in enumerate(traj_query.keys()):
        axes[i] = plot_track(z, traj_query[traj_key], 
                             chrom, start, end, conv_size, ax=axes[i], color=color_gradient(i/len(traj_query)),
                             libsize=traj_lib_size[traj_key])
    return fig, axes

plot_trajectory(z, 'chr1', 16000000, 18000000)

# %%
a = z.get_track(np.where(z.metadata.labels.str.split(' ').str[1]=='Excitatory')[0], 'chr1', 0, 10000000)
a = np.convolve(a, np.ones(100), mode='same')
sns.histplot(a)

# %%
# %%
