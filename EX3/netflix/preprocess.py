import os
import zipfile
import io
import pandas as pd
from dateutil import parser
import scipy.sparse
import numpy as np
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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
from sklearn.manifold import SpectralEmbedding
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import random as sparse_random
from sklearn.random_projection import sparse_random_matrix
import plotly
import plotly.express as px

PICKLE_FILE_NAME_NETFLIX_MATRIX = 'netflix_matrix'
PICKLE_FILE_NAME_MOVIES_INFO = 'movies_info.pkl'


def create_initial_data(remove_empty_cols=False, use_genres=False):
    """
    This function reads all the data in the zip folder named 'archive.zip'.
    The file was downloaded from here: https://www.kaggle.com/netflix-inc/netflix-prize-data
    The folder contains files named 'combined_data_X', where in each text file there's info about ratings of users to
    many different movies. Every time a new movie starts, there's a line with its id and ':', and then after this line
    there are many lines with users' rating to this movie, in this format: "'customer_id, rating, date'", where rating is
    between 1 and 5, and date is in the format 'YYYY-MM-DD'.
    Movies' ids span the range 1 to 17,770 and customers' ids span the range 1 to 2,649,429, but many customer ids
    are not in use. Only 480,189 are in use.
    Except for these files, the 'archive.zip' contains a file named 'movie_titles.csv' with information about the
    movies. Each row is in the format: 'movie_id, year_of_release, title'.
    The function also reads another file with the genres of the movies if the 'use_genres' parameter is set to True
    (needs to be downloaded from 'https://github.com/bmxitalia/netflix-prize-with-genres/blob/master/netflix_genres.csv')
    , and adds it to the given information about the movies (joins dataframes).
    This function iterates through all the files mentioned above and builds a sparse matrix (Scipy lil matrix)
    whose rows represent movies' ids and columns represent users' ids.
    Important note - the indices start from 0, whereas the true ids start from one, so you should take it into
    account when relating to a specific movie or user in the matrix.
    After creating the files, the function saves the information abouts the movies to a pickle file, using joblib,
    and saves the large sparse matrix to '.npz' file.
    The function receives a default argument - whether to remove empty columns or not (user ids which have no use).
    Last note - if the files were already created, the function just reads them from the saved files (pickle and npz).
    :param remove_empty_cols:
    :param use_genres
    :return:
    """
    if not os.path.exists(PICKLE_FILE_NAME_NETFLIX_MATRIX + '.npz') or not os.path.exists(
            PICKLE_FILE_NAME_MOVIES_INFO):
        print("Started processing the data from scratch")
        # this matrix has movies indices as rows and user ids as columns, and inside it there's the rating
        mat_of_movies_and_users = scipy.sparse.lil_matrix((17_770, 2_649_429))
        with zipfile.ZipFile('archive.zip', 'r') as z:
            with tqdm(total=17_770, position=0, leave=True) as pbar:
                for filename in z.namelist():
                    if 'combined_data' in filename:
                        with z.open(filename, 'r') as f:
                            f = io.TextIOWrapper(f)
                            parse_single_ratings_file(f, mat_of_movies_and_users, pbar)
                    if 'movie_titles.csv' in filename:
                        with z.open(filename, 'r') as f:
                            df_of_movies_info = pd.read_csv(f, error_bad_lines=False, encoding='latin-1',
                                                            index_col=0,
                                                            names=['year_of_release', 'title'])
                            if use_genres:
                                df_of_movies_genres = get_genres_of_movies()
                                df_of_movies_info = df_of_movies_info.join(df_of_movies_genres)
                                df_of_movies_info.fillna(0, inplace=True)
            if remove_empty_cols:
                mat_of_movies_and_users = remove_empty_cols_of_sparse_matrix(mat_of_movies_and_users)
            save_created_files(df_of_movies_info, mat_of_movies_and_users)
    else:
        df_of_movies_info, mat_of_movies_and_users = load_files_from_disk()
    return mat_of_movies_and_users, df_of_movies_info


def load_files_from_disk():
    """
    This function loads both files from disk if they exist (one file is the ratings matrix and the other is the
    dataframe with the information about the movies)
    :return:
    """
    print("Started loading data from disk")
    mat_of_movies_and_users = scipy.sparse.load_npz(PICKLE_FILE_NAME_NETFLIX_MATRIX + '.npz').tolil()
    df_of_movies_info = joblib.load(PICKLE_FILE_NAME_MOVIES_INFO)
    print("Finished loading data from disk")
    return df_of_movies_info, mat_of_movies_and_users


def save_created_files(df_of_movies_info, mat_of_movies_and_users):
    """
    This function saves the files which were created
    :param df_of_movies_info:
    :param mat_of_movies_and_users:
    :return:
    """
    try:
        print("Started saving pickle files")
        scipy.sparse.save_npz(PICKLE_FILE_NAME_NETFLIX_MATRIX, mat_of_movies_and_users.tocsr(),
                              compressed=True)
        joblib.dump(df_of_movies_info, PICKLE_FILE_NAME_MOVIES_INFO)
        print("Finished saving pickle files")
    except Exception as e:
        print("failed to save files")
        print(e)


def parse_single_ratings_file(f, mat_of_movies_and_users, pbar):
    """
    This function handles a single ratings' file - parses the file and saves its data in the sparse matrix
    :param f:
    :param mat_of_movies_and_users:
    :param pbar:
    :return:
    """
    for line in f:
        if ',' in line:
            customer_id, rating, date = line.split(',')
            date = parser.parse(date)
            rating = int(rating)
            customer_id = int(customer_id)
            mat_of_movies_and_users[movie_id - 1, customer_id - 1] = rating
        else:
            movie_id = int(line.split(':')[0])
            pbar.update()


def remove_empty_cols_of_sparse_matrix(mat_of_movies_and_users):
    """
    This function receives the original matrix of movies and users' ratings and removes empty columns
    (ids of users who have no ratings for any movie)
    :param mat_of_movies_and_users:
    :return:
    """
    print("Started removing empty cols of matrix")
    indices = np.nonzero(mat_of_movies_and_users)
    columns_non_unique = indices[1]
    unique_columns = sorted(set(columns_non_unique))
    mat_of_movies_and_users = mat_of_movies_and_users.tocsc()[:, unique_columns]
    print("Finished removing empty cols of matrix")
    return mat_of_movies_and_users


def get_genres_of_movies():
    """
    This function reads the file named 'netflix_genres.csv' which has a mapping between movie id and its genre.
    It prints all unique genres and return a dataframe which is a one-hot encoding - the dataframe contains all
    genres as columns, and 1 if this movie is from this genre, and 0 otherwise (because a movie usually corresponds
    to more than one genre)
    :return:
    """
    df_of_genres = pd.read_csv('netflix_genres.csv')
    all_genres = set()
    for movie_genre in df_of_genres['genres'].to_list():
        all_genres.update(movie_genre.split('|'))
    print("all genres are:")
    print(all_genres)
    print("Number of genres is: ")
    print(len(all_genres))

    df_of_movies_and_all_genres = pd.DataFrame(columns=all_genres)
    for idx, row in df_of_genres.iterrows():
        movie_id = row[0]
        movie_genres = row[1].split('|')
        for movie_genre in movie_genres:
            df_of_movies_and_all_genres.loc[movie_id, movie_genre] = 1
    df_of_movies_and_all_genres.fillna(0, inplace=True)
    return df_of_movies_and_all_genres


def plot_in_plotly(res, method, labels, s=6, addon="", discrite=True):
    df = pd.DataFrame(res, columns=[method + '_1', method + '_2', method + '_3'])
    if discrite:
        df['labels'] = labels.astype(str)
    else:
        df['labels'] = labels
    fig = px.scatter_3d(df, x=method + '_1', y=method + '_2', z=method + '_3', color='labels')
    fig.update_layout(
        title=method + " Plot " + addon
    )
    fig.update_traces(marker=dict(size=s))
    fig.show()


if __name__ == '__main__':
    res = create_initial_data(True, True)
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
    fin2 = np.zeros_like(fin)
    gen = res[1]
    my = gen.columns[2:]
    my = np.array(my)
    my = np.append(my, ["No"])
    fin -= 2
    fin[np.where(fin == -2)] = 27
    fin2 = np.zeros_like(fin, dtype='object')
    for i in range(len(fin)):
        fin2[i] = my[int(fin[i])]

    little_batch_of_poeple = res[0][:, 0:1000].todense()
    # pca = PCA(n_components=50)
    # p = pca.fit_transform(little_batch_of_poeple)
    # fig = plt.figure()
    # plt.title("PCA")
    # color = list(map(str, res[1]["year_of_release"]))
    # from sklearn.preprocessing import LabelEncoder
    #
    # le = LabelEncoder()
    # color = le.fit_transform(color)
    #
    # plt.scatter(p[:, 0], p[:, 1],
    #             cmap=plt.cm.Spectral)
    # plt.legend()
    # plt.show()
    #
    # um = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='correlation').fit_transform(p)
    # fig = plt.figure()
    # plt.title("UMAP")
    # plt.scatter(um[:, 0], um[:, 1], label=np.array(res[1].columns[2:]), c=fin, cmap=plt.cm.Spectral)
    # plt.legend()
    # plt.show()
    #
    # fig, ax = plt.subplots()
    #
    # scatter = ax.scatter(um[:, 0], um[:, 1], c=fin)
    #
    # # produce a legend with the unique colors from the scatter
    # legend1 = ax.legend(*scatter.legend_elements(),
    #                     loc="lower left", title="Classes")
    # ax.add_artist(legend1)
    #
    # plt.show()

    p_new = TruncatedSVD(n_components=10).fit_transform(little_batch_of_poeple)
    # plt.title("PCA_new")
    # plt.scatter(p_new[:, 0], p_new[:, 1], label=np.array(res[1].columns[2:]), cmap=plt.cm.Spectral)
    # plt.legend()
    # plt.show()

    a = np.array(little_batch_of_poeple)
    hmr = np.count_nonzero(a, axis=1)
    summ = np.sum(a, axis=1)
    rev = np.divide(summ, hmr, where=hmr != 0)
    # dm_embedding = SpectralEmbedding(n_components=2)
    # dm = dm_embedding.fit_transform(p_new)
    # plt.title("DM")
    # plt.xlim([dm.min(), dm.max()])
    # plt.ylim([dm.min(), dm.max()])
    # plt.scatter(dm[:, 0], dm[:, 1], label=np.array(res[1].columns[2:]), cmap=plt.cm.Spectral)
    # plt.legend()
    # plt.show()
    # dm_embedding = SpectralEmbedding(n_components=2)
    # dm = dm_embedding.fit_transform(p_new)
    # plt.title("DM")
    # plt.xlim([dm.min(), dm.max()])
    # plt.ylim([dm.min(), dm.max()])
    # plt.scatter(dm[:, 0], dm[:, 1], c=summ, cmap=plt.cm.Spectral)
    # plt.legend()
    # plt.show()
    #
    # plt.title("DM")
    # plt.xlim([dm.min(), dm.max()])
    # plt.ylim([dm.min(), dm.max()])
    # plt.scatter(dm[:, 0], dm[:, 1], c=np.log(summ), cmap=plt.cm.Spectral)
    # plt.legend()
    # plt.show()

    dm_embedding3 = SpectralEmbedding(n_components=3)
    dm3 = dm_embedding3.fit_transform(p_new)

    # plot_in_plotly(dm3, "DM", summ, addon="on Netflix data set in 3D", discrite=False)



    pass
