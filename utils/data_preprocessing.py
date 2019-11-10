#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

# data_preprocessing.py: module for loading and preparing data. And displaying some statistics.

from urllib.request import urlretrieve
import zipfile, os
import matplotlib.pyplot as pyplot
import numpy as np
import scipy.sparse as sps

def parse_data(file, is_URM):
    print("\nLoading data ... ", end="\n")

    matrix_path = download_data(file)
    matrix_file = open(matrix_path, 'r') # read file's content

    if is_URM == True:
        global URM_file
        URM_file = matrix_file

    # Create a tuple for each interaction (line in the file)
    matrix_file.seek(0)  # start from beginning of the file
    matrix_tuples = []

    for line in matrix_file:
        matrix_tuples.append(row_split(line, is_URM))

    # Separate the four columns in different independent lists
    user_list, item_list, content_list, timestamp_list = zip(*matrix_tuples)  # join tuples together (zip() to map values)

    # Convert values to list
    user_list = list(user_list)
    item_list = list(item_list)
    content_list = list(content_list)
    timestamp_list = list(timestamp_list)

    return user_list, item_list, content_list, timestamp_list

def download_data(file):
    data_url = "http://files.grouplens.org/datasets/movielens/ml-10m.zip"
    data_file_path = "data/Movielens_10M"
    data_file_name = data_file_path + "/movielens_10m.zip"

    # If file exists, skip the download
    os.makedirs(data_file_path, exist_ok=True)  # create dir if not exists
    if not os.path.exists(data_file_name):
        urlretrieve(data_url, data_file_name)  # copy network object denoted by a URL to a local file

    data_file = zipfile.ZipFile(data_file_name)  # open zip file
    data_path = data_file.extract(file, path=data_file_path)  # extract data

    return data_path

# Train/test split
def train_test_holdout(URM, train_perc=0.8):
    num_interactions = URM.nnz  # number of nonzero values

    URM = URM.tocoo()
    shape = URM.shape

    # URM.data: ratingList, URM.row: user_list, URM.col: item_list

    # Take random samples of data. Use random boolean mask
    train_mask = np.random.choice([True, False], num_interactions, p=[train_perc, 1 - train_perc]) # p train_perc for True and 1-train_perc for Fase
    URM_train = csr_sparse_matrix(URM.data[train_mask], URM.row[train_mask], URM.col[train_mask], shape=shape)

    test_mask = np.logical_not(train_mask) # Compute the truth value of NOT x element-wise.
    URM_test = csr_sparse_matrix(URM.data[test_mask], URM.row[test_mask], URM.col[test_mask], shape=shape)

    return URM_train, URM_test

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

# Matrix in COOrdinate format (fast format for constructing sparse matrices)
def csr_sparse_matrix(data, row, col, shape=None):
    matrix = sps.coo_matrix((data, (row, col)), shape=shape)
    matrix = matrix.tocsr() # put in Compressed Sparse Row format for fast row access

    return matrix

# Statistics on interactions
def display_statistics(user_list, item_list, URM):
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

    plot_data(item_popularity, 'ro', 'Item Popularity', 'Num Interactions', 'Item Index')

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

    plot_data(item_list_unique_nonzero, 'ro', 'Item Popularity Nonzero' , 'Num Interactions', 'Item Index')

    # User Activity
    print("User activity ...\n")
    user_activity = flatten_array(URM)
    plot_data(user_activity, 'ro', 'User Activity', 'Num Interactions', 'User Index')

def rating_distribution_over_time(timestamp_list):
    print("Rating distribution over time ... ", end="\n")
    # Clone the list to avoid changing the ordering of the original data
    timestamp_sorted = list(timestamp_list)
    timestamp_sorted.sort()

    plot_data(timestamp_sorted, 'ro', 'Timestamp Sorted', 'Timestamp', 'Item Index')

def plot_data(data, marker, title, y_label, x_label):
    pyplot.plot(data, marker)
    pyplot.title(title)
    pyplot.ylabel(y_label)
    pyplot.xlabel(x_label)
    pyplot.show()

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

def list_ID_stats(IDList, label):
    min_val = min(IDList)
    max_val = max(IDList)
    unique_val = len(set(IDList))
    missing_val = 1 - unique_val / (max_val - min_val)

    print("{} data, ID: min {}, max {}, unique {}, missing {:.2f} %".format(label, min_val, max_val, unique_val, missing_val))

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
        return X.tocsc().astype(dtype)
    elif format == 'csr' and not isinstance(X, sps.csr_matrix):
        return X.tocsr().astype(dtype)
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

