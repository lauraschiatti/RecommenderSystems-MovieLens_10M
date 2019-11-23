#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

# data_manager.py: module for loading and preparing data. Also for displaying some statistics.

from urllib.request import urlretrieve
import zipfile, os
import matplotlib.pyplot as pyplot
import numpy as np
import scipy.sparse as sps
from sklearn import preprocessing

################################ LOADING DATA ...  ################################

def parse_data(file, is_URM):
    print("Loading original data ... ", end="\n")

    data_path = extract_data(file)
    matrix_file = open(data_path, 'r')  # read file's content

    if is_URM == True:
        global URM_file
        URM_file = matrix_file

    # Create a tuple for each interaction (line in the file)
    matrix_file.seek(0)  # start from beginning of the file
    matrix_tuples = []

    for line in matrix_file:
        matrix_tuples.append(row_split(line, is_URM))

    # Separate the four columns in different independent lists
    user_list, item_list, content_list, timestamp_list = zip(*matrix_tuples)  # join tuples together


    # Create lists of all users, items and contents (ratings)
    user_list = list(user_list)
    item_list = list(item_list)
    content_list = list(content_list)
    timestamp_list = list(timestamp_list)

    return user_list, item_list, content_list, timestamp_list


def extract_data(file):
    DATASET_URL = "http://files.grouplens.org/datasets/movielens/ml-10m.zip"
    DATASET_SUBFOLDER = "data/Movielens_10M"
    DATASET_FILE_NAME = "/movielens_10m.zip"

    try:

        data_file = zipfile.ZipFile(DATASET_SUBFOLDER + DATASET_FILE_NAME)  # open zip file

    except (FileNotFoundError, zipfile.BadZipFile):
        print("Unable to find data zip file. Downloading...")

        download_from_URL(DATASET_URL, DATASET_SUBFOLDER, DATASET_FILE_NAME)

        data_file = zipfile.ZipFile(DATASET_SUBFOLDER + DATASET_FILE_NAME)  # open zip file

    data_path = data_file.extract(file, path=DATASET_SUBFOLDER)  # extract data

    return data_path


def download_from_URL(URL, folder_path, file_name):
    # If directory does not exist, create
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    print("Downloading: {}".format(URL))
    print("In folder: {}".format(folder_path + file_name))

    try:
        urlretrieve(URL, folder_path + file_name)  # copy network object to a local file

    except urllib.request.URLError as urlerror:  # @TODO: handle network connection error
        print("Unable to complete automatic download, network error")
        raise urlerror


def download_data(file):
    DATASET_URL = "http://files.grouplens.org/datasets/movielens/ml-10m.zip"
    DATASET_SUBFOLDER = "data/Movielens_10M"
    DATASET_FILE_NAME = DATASET_SUBFOLDER + "/movielens_10m.zip"

    # If file exists, skip the download
    os.makedirs(DATASET_SUBFOLDER, exist_ok=True)  # create dir if not exists
    if not os.path.exists(DATASET_FILE_NAME):
        urlretrieve(DATASET_URL, DATASET_FILE_NAME)  # copy network object denoted by a URL to a local file
        print("Downloading: {}".format(DATASET_URL))

    data_file = zipfile.ZipFile(DATASET_FILE_NAME)  # open zip file
    data_path = data_file.extract(file, path=DATASET_SUBFOLDER)  # extract data

    return data_path

################################ PARSING DATA ...  ################################

# Separate user, item, rating (or tag) and timestamp
def row_split(row_string, is_URM):
    # file format: 1::364::5::838983707
    split = row_string.split("::")
    split[3] = split[3].replace("\n", "")

    split[0] = int(split[0])
    split[1] = int(split[1])

    if is_URM == True:
        split[2] = float(split[2])  # rating is a float
    elif is_URM == False:
        split[2] = str(split[2])  # tag is a string, not a float like the rating

    split[3] = int(split[3])

    result = tuple(split)

    return result

# Matrix Compressed Sparse Row format
def csr_sparse_matrix(data, row, col, shape=None):
    matrix = sps.coo_matrix((data, (row, col)), shape=shape)
    matrix = matrix.tocsr() # put in Compressed Sparse Row format for fast row access

    return matrix

# Remove duplicates from list by using a set
def remove_duplicates(list_o):
    list_unique = list(set(list_o))

    return list_unique

# Flatten single dimensional entries in an array
def flatten_array(array):
    flattened_array = (array > 0).sum(axis=0)
    flattened_array = np.array(flattened_array).squeeze()
    flattened_array = np.sort(flattened_array)

    return flattened_array

# Transforms matrix into a specific format
def check_matrix(X, format='csc', dtype=np.float32):
    """
        This function takes a matrix as input and transforms it into the specified format.
        The matrix in input can be either sparse or ndarray.
        If the matrix in input has already the desired format, it is returned as-is
        the dtype parameter is always applied and the default is np.float32
        :param X:
        :param format:
        :param dtype:
        :return:
    """

    if format == 'csc' and not isinstance(X, sps.csc_matrix):
        return X.tocsc().astype(dtype) # Compressed Sparse Column format
    elif format == 'csr' and not isinstance(X, sps.csr_matrix):
        return X.tocsr().astype(dtype) # Compressed Sparse Row format
    elif format == 'coo' and not isinstance(X, sps.coo_matrix):
        return X.tocoo().astype(dtype)
    elif format == 'dok' and not isinstance(X, sps.dok_matrix):
        return X.todok().astype(dtype)
    elif format == 'bsr' and not isinstance(X, sps.bsr_matrix):
        return X.tobsr().astype(dtype)
    elif format == 'dia' and not isinstance(X, sps.dia_matrix):
        return X.todia().astype(dtype)
    elif format == 'lil' and not isinstance(X, sps.lil_matrix):
        return X.tolil().astype(dtype)
    elif isinstance(X, np.ndarray):
        X = sps.csr_matrix(X, dtype=dtype)
        X.eliminate_zeros()
        return check_matrix(X, format=format, dtype=dtype)
    else:
        return X.astype(dtype)

# Encode labels with value between 0 and n_classes-1.
def label_encoder(list):
    le = preprocessing.LabelEncoder()
    le.fit(list)
    encoded_list = le.transform(list)
    # print("encoded encoded_list", encoded_list[0:10])

    return encoded_list



################################ STATISTICS ...  ################################

def plot_data(data, marker, title, y_label, x_label):
    pyplot.plot(data, marker)
    pyplot.title(title)
    pyplot.ylabel(y_label)
    pyplot.xlabel(x_label)
    pyplot.show()

def get_statistics_URM(user_list, item_list, URM):
    print("\nStatistics ... ")

    # Number of interactions in the URM
    URM_file.seek(0)
    number_interactions = 0

    for _ in URM_file:
        number_interactions += 1
    print("The number of interactions is {}".format(number_interactions))

    user_list_unique = remove_duplicates(user_list)
    item_list_unique = remove_duplicates(item_list)

    num_users = len(user_list_unique)
    num_items = len(item_list_unique)

    print("Number of items\t {}, Number of users\t {}".format(num_items, num_users))
    print("Max ID items\t {}, Max Id users\t {}\n".format(max(item_list_unique), max(user_list_unique)))
    print("Average interactions per user {:.2f}".format(number_interactions / num_users))
    print("Average interactions per item {:.2f}\n".format(number_interactions / num_items))

    print("Sparsity {:.2f} %\n".format((1 - float(number_interactions) / (num_items * num_users)) * 100))

    # Item popularity
    print("Item popularity ... \n")
    item_popularity = flatten_array(URM)

    # plot_data(item_popularity, 'ro', 'Item Popularity', 'Num Interactions', 'Item Index')

    ten_percent = int(num_items / 10)

    print("\nAverage per-item interactions over the whole dataset {:.2f}".
          format(item_popularity.mean()))

    print("Average per-item interactions for the top 10% popular items {:.2f}".
          format(item_popularity[-ten_percent].mean()))

    print("Average per-item interactions for the least 10% popular items {:.2f}".
          format(item_popularity[:ten_percent].mean()))

    print("Average per-item interactions for the median 10% popular items {:.2f}".
          format(item_popularity[int(num_items * 0.45):int(num_items * 0.55)].mean()))

    print("Number of items with zero interactions {}".
          format(np.sum(item_popularity == 0)))

    item_list_unique_nonzero = item_popularity[item_popularity > 0]

    ten_percent = int(len(item_list_unique_nonzero) / 10)

    print("\nAverage per-item interactions over the whole dataset {:.2f}".
          format(item_list_unique_nonzero.mean()))

    print("Average per-item interactions for the top 10% popular items {:.2f}".
          format(item_list_unique_nonzero[-ten_percent].mean()))

    print("Average per-item interactions for the least 10% popular items {:.2f}".
          format(item_list_unique_nonzero[:ten_percent].mean()))

    print("Average per-item interactions for the median 10% popular items {:.2f}".
          format(item_list_unique_nonzero[int(num_items * 0.45):int(num_items * 0.55)].mean()))

    # plot_data(item_list_unique_nonzero, 'ro', 'Item Popularity Nonzero' , 'Num Interactions', 'Item Index')

    # User Activity
    print("User activity ...\n")
    user_activity = flatten_array(URM)
    # plot_data(user_activity, 'ro', 'User Activity', 'Num Interactions', 'User Index')

def rating_distribution_over_time(timestamp_list):
    print("Rating distribution over time ... ", end="\n")
    # Clone the list to avoid changing the ordering of the original data
    timestamp_sorted = list(timestamp_list)
    timestamp_sorted.sort()

    plot_data(timestamp_sorted, 'ro', 'Timestamp Sorted', 'Timestamp', 'Item Index')

# Show how the recommendations are distributed among items
def recommendations_distribution(URM, recommender):
        x_tick = np.arange(URM.shape[1])
        counter = np.zeros(URM.shape[1])
        for user_id in range(URM.shape[0]):
            recs = recommender.recommend(user_id, at=5)
            counter[recs] += 1
            if user_id % 10000 == 0:
                print("Recommended to user {}/{}".format(user_id, URM.shape[0]))

        pyplot.plot(x_tick, np.sort(counter)[::-1])
        pyplot.ylabel('Number of recommendations')
        pyplot.xlabel('Items')
        pyplot.show()

def list_ID_stats(ID_list, label):
    min_val = min(ID_list)
    max_val = max(ID_list)
    unique_val = len(set(ID_list))
    missing_val = 1 - unique_val / (max_val - min_val)

    print("{} data, ID: min {}, max {}, unique {}, missig {:.2f} %".format(label, min_val, max_val, unique_val,
                                                                           missing_val * 100))

def item_feature_ratios(ICM):
    # Features per item
    ICM = sps.csr_matrix(ICM)
    features_per_item = np.ediff1d(ICM.indptr)  # differences between consecutive elements of an array.
    print("Features Per Item", features_per_item.shape)
    features_per_item = np.sort(features_per_item)
    # data.plot_data(features_per_item, 'ro', 'Features Per Item', 'Num Features', 'Item Index')

    # Items per feature
    ICM = sps.csc_matrix(ICM)
    items_per_feature = np.ediff1d(ICM.indptr)
    print("Items Per Feature", items_per_feature.shape)
    items_per_feature = np.sort(items_per_feature)
    # data.plot_data(items_per_feature, 'ro', 'Items Per Feature', 'Num Items', 'Feature Index')

# Train/test split
def train_test_holdout(URM, train_perc=0.8):

    number_interactions = URM.nnz  # number of nonzero values
    URM = URM.tocoo()  # Coordinate list matrix (COO)
    shape = URM.shape

    #  URM.row: user_list, URM.col: item_list, URM.data: rating_list

    # Sampling strategy: take random samples of data using a boolean mask
    train_mask = np.random.choice(
        [True, False],
        number_interactions,
        p=[train_perc, 1 - train_perc])  # train_perc for True, 1-train_perc for False

    URM_train = csr_sparse_matrix(URM.data[train_mask],
                                  URM.row[train_mask],
                                  URM.col[train_mask],
                                  shape=shape)

    test_mask = np.logical_not(train_mask)  # remaining samples
    URM_test = csr_sparse_matrix(URM.data[test_mask],
                                 URM.row[test_mask],
                                 URM.col[test_mask],
                                 shape=shape)

    return URM_train, URM_test

    # todo: check splitting strategy
    # There are different sampling strategies:
    # - random holdout: splits an URM in two matrices selecting the number of interactions one user at a time
    # - leave k out: splits an URM in two matrices selecting the k_out interactions one user at a time
    # - cold items: Selects a certain percentage of the URM_all WARM items and splits the URM in two
    # - sequential: only the final item in an item_list

    # The difference in MAP you see between your local test and the leaderboard
    # is due to the fact you are splitting by a random hold-out all the playlists.

