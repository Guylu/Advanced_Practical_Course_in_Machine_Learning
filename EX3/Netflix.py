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
import math
import re
from scipy.sparse import csr_matrix
import seaborn as sns
from collections import deque
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import pydiffmap


# movie id:
# CustomerID,Rating,Date

# - MovieIDs range from 1 to 17770 sequentially.
# - CustomerIDs range from 1 to 2649429, with gaps. There are 480189 users.
# - Ratings are on a five star (integral) scale from 1 to 5.
# - Dates have the format YYYY-MM-DD.

# movie id mapping is in movie_titles.csv

def method_1():
    df = pd.read_csv("./netflix/combined_data_1.txt", header=None, names=['Cust_Id', 'Rating'],
                     usecols=[0, 1])
    df['Rating'] = df['Rating'].astype(float)

    df.index = np.arange(0, len(df))
    p = df.groupby('Rating')['Rating'].agg(['count'])
    # get movie count
    movie_count = df.isnull().sum()[1]
    # get customer count
    cust_count = df['Cust_Id'].nunique() - movie_count
    # get rating count
    rating_count = df['Cust_Id'].count() - movie_count

    # ax = p.plot(kind='barh', legend=False, figsize=(15, 10))
    # plt.title('Total pool: {:,} Movies, {:,} customers, {:,} ratings given'.format(movie_count, cust_count,
    #                                                                                rating_count), fontsize=20)
    # plt.axis('off')
    # for i in range(1, 6):
    #     ax.text(p.iloc[i - 1][0] / 4, i - 1,
    #             'Rating {}: {:.0f}%'.format(i, p.iloc[i - 1][0] * 100 / p.sum()[0]), color='white',
    #             weight='bold')
    # plt.show()

    df_nan = pd.DataFrame(pd.isnull(df.Rating))
    df_nan = df_nan[df_nan['Rating'] == True]
    df_nan = df_nan.reset_index()
    movie_np = []
    movie_id = 1
    for i, j in zip(df_nan['index'][1:], df_nan['index'][:-1]):
        # numpy approach
        temp = np.full((1, i - j - 1), movie_id)
        movie_np = np.append(movie_np, temp)
        movie_id += 1
    # Account for last record and corresponding length
    # numpy approach
    last_record = np.full((1, len(df) - df_nan.iloc[-1, 0] - 1), movie_id)
    movie_np = np.append(movie_np, last_record)
    print('Movie numpy: {}'.format(movie_np))
    print('Length: {}'.format(len(movie_np)))
    # remove those Movie ID rows
    df = df[pd.notnull(df['Rating'])]
    df['Movie_Id'] = movie_np.astype(int)
    df['Cust_Id'] = df['Cust_Id'].astype(int)

    # df = pd.read_csv("nicer_data1.csv")

    f = ['count', 'mean']
    df_movie_summary = df.groupby('Movie_Id')['Rating'].agg(f)
    df_movie_summary.index = df_movie_summary.index.map(int)
    movie_benchmark = round(df_movie_summary['count'].quantile(0.5), 0)
    drop_movie_list = df_movie_summary[df_movie_summary['count'] < movie_benchmark].index
    print('Movie minimum times of review: {}'.format(movie_benchmark))
    df_cust_summary = df.groupby('Cust_Id')['Rating'].agg(f)
    df_cust_summary.index = df_cust_summary.index.map(int)
    cust_benchmark = round(df_cust_summary['count'].quantile(0.5), 0)
    drop_cust_list = df_cust_summary[df_cust_summary['count'] < cust_benchmark].index
    print('Customer minimum times of review: {}'.format(cust_benchmark))
    print('Original Shape: {}'.format(df.shape))
    df = df[~df['Movie_Id'].isin(drop_movie_list)]
    df = df[~df['Cust_Id'].isin(drop_cust_list)]

    df.to_csv("data1_trimmed.csv")

    print('After Trim Shape: {}'.format(df.shape))
    print('-Data Examples-')
    print(df.iloc[::5000000, :])
    df_p = pd.pivot_table(df, values='Rating', index='Cust_Id', columns='Movie_Id')
    df_p.to_csv("data1_trimmed_as_sparse_matrix.csv")
    print(df_p.shape)


##################################################################################
##################################################################################


def method_2():
    movie_titles = pd.read_csv("./netflix/combined_data_1.txt",
                               encoding='ISO-8859-1',
                               header=None,
                               names=['Id', 'Year', 'Name']).set_index('Id')

    print('Shape Movie-Titles:\t{}'.format(movie_titles.shape))
    movie_titles.sample(5)

    # Load a movie metadata dataset
    movie_metadata = pd.read_csv("./netflix/movie_titles.csv", low_memory=False)[
        ['original_title', 'overview', 'vote_count']].set_index('original_title').dropna()
    # Remove the long tail of rarly rated moves
    movie_metadata = movie_metadata[movie_metadata['vote_count'] > 10].drop('vote_count', axis=1)

    print('Shape Movie-Metadata:\t{}'.format(movie_metadata.shape))
    movie_metadata.sample(5)

    # Load single data-file
    df_raw = pd.read_csv("./netflix/combined_data_1.txt", header=None,
                         names=['User', 'Rating', 'Date'], usecols=[0, 1, 2])

    # Find empty rows to slice dataframe for each movie
    tmp_movies = df_raw[df_raw['Rating'].isna()]['User'].reset_index()
    movie_indices = [[index, int(movie[:-1])] for index, movie in tmp_movies.values]

    # Shift the movie_indices by one to get start and endpoints of all movies
    shifted_movie_indices = deque(movie_indices)
    shifted_movie_indices.rotate(-1)

    # Gather all dataframes
    user_data = []

    # Iterate over all movies
    for [df_id_1, movie_id], [df_id_2, next_movie_id] in zip(movie_indices, shifted_movie_indices):

        # Check if it is the last movie in the file
        if df_id_1 < df_id_2:
            tmp_df = df_raw.loc[df_id_1 + 1:df_id_2 - 1].copy()
        else:
            tmp_df = df_raw.loc[df_id_1 + 1:].copy()

        # Create movie_id column
        tmp_df['Movie'] = movie_id

        # Append dataframe to list
        user_data.append(tmp_df)

    # Combine all dataframes
    df = pd.concat(user_data)
    del user_data, df_raw, tmp_movies, tmp_df, shifted_movie_indices, movie_indices, df_id_1, movie_id, df_id_2, next_movie_id
    print('Shape User-Ratings:\t{}'.format(df.shape))
    df.sample(5)

    # Filter sparse movies
    min_movie_ratings = 10000
    filter_movies = (df['Movie'].value_counts() > min_movie_ratings)
    filter_movies = filter_movies[filter_movies].index.tolist()

    # Filter sparse users
    min_user_ratings = 200
    filter_users = (df['User'].value_counts() > min_user_ratings)
    filter_users = filter_users[filter_users].index.tolist()

    # Actual filtering
    df_filterd = df[(df['Movie'].isin(filter_movies)) & (df['User'].isin(filter_users))]
    del filter_movies, filter_users, min_movie_ratings, min_user_ratings
    print('Shape User-Ratings unfiltered:\t{}'.format(df.shape))
    print('Shape User-Ratings filtered:\t{}'.format(df_filterd.shape))

    # Create a user-movie matrix with empty values
    df_p = df_filterd.pivot_table(index='User', columns='Movie', values='Rating')
    print('Shape User-Movie-Matrix:\t{}'.format(df_p.shape))
    df_p.sample(3)


def topics():
    from imdb import IMDb
    # create an instance of the IMDb class
    ia = IMDb()

    mov = ia.search_movie("Jade")


if __name__ == '__main__':
    df = pd.read_csv("data1_trimmed_as_sparse_matrix.csv")
    df = df.set_index("Cust_Id")
    df = df.fillna(0)

    pca = PCA(n_components=50)
    p = pca.fit_transform(df)
    fig = plt.figure()
    plt.title("PCA")
    plt.scatter(p[:, 0], p[:, 1], cmap=plt.cm.Spectral)
    plt.show()

    # embedding = MDS(n_components=2)
    # mds = embedding.fit_transform(df)
    # fig = plt.figure()
    # plt.title("MDS")
    # plt.scatter(mds[:, 0], mds[:, 1], cmap=plt.cm.Spectral)
    # plt.show()

    umap = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='correlation').fit_transform(df)
    fig = plt.figure()
    plt.title("UMAP")
    plt.scatter(umap[:, 0], umap[:, 1], cmap=plt.cm.Spectral)
    plt.show()
    #
    # mydmap = pydiffmap.diffusion_map.DiffusionMap.from_sklearn(n_evecs=2, alpha=0.5, epsilon='bgh', k=5)
    # dmap = mydmap.fit_transform(df)
    # fig = plt.figure()
    # plt.title("DM")
    # plt.scatter(dmap[:, 0], dmap[:, 1], cmap=plt.cm.Spectral)
    # plt.show()
