#!/usr/local/bin/python3.6
# -*- coding: utf-8 -*-

# data_preprocessing.py: module for loading and preparing data. And displaying some statistics.

from urllib.request import urlretrieve
import zipfile, os
import matplotlib.pyplot as pyplot
import numpy as np
import scipy.sparse as sps

URMFile = ""

def parse_data(file):
    print("\nLoading data ... ", end="\n")

    matrixPath = download_data(file)
    matrixFile = open(matrixPath, 'r') # read file's content

    global URMFile
    URMFile = matrixFile

    # Create a tuple for each interaction (line in the file)
    matrixFile.seek(0)  # start from beginning of the file
    matrixTuples = []

    for line in matrixFile:
        matrixTuples.append(row_split(line))

    # Separate the four columns in different independent lists
    userList, itemList, ratingList, timestampList = zip(*matrixTuples)  # join tuples together (zip() to map values)

    # Convert values to list
    userList = list(userList)
    itemList = list(itemList)
    ratingList = list(ratingList)
    timestampList = list(timestampList)

    return userList, itemList, ratingList, timestampList


def download_data(file):
    dataUrl = "http://files.grouplens.org/datasets/movielens/ml-10m.zip"
    dataFilePath = "data/Movielens_10M"
    dataFileName = dataFilePath + "/movielens_10m.zip"

    # If file exists, skip the download
    os.makedirs(dataFilePath, exist_ok=True)  # create dir if not exists
    if not os.path.exists(dataFileName):
        urlretrieve(dataUrl, dataFileName)  # copy network object denoted by a URL to a local file

    dataFile = zipfile.ZipFile(dataFileName)  # open zip file
    dataPath = dataFile.extract(file, path=dataFilePath)  # extract data

    return dataPath


# Split dataset into train and test sets
def data_splitting(userList, itemList, ratingList, URM, trainTestSplit):
    numInteractions = URM.nnz  # number of nonzero values

    # Take random samples of data.
    # Use random boolean mask. p trainTestSplit for True and 1-trainTestSplit for False
    userList = np.array(userList)
    itemList = np.array(itemList)
    ratingList = np.array(ratingList)

    trainMask = np.random.choice([True, False], numInteractions,p=[trainTestSplit, 1 - trainTestSplit])
    URMTrain = csr_sparse_matrix(ratingList[trainMask], userList[trainMask], itemList[trainMask])

    testMask = np.logical_not(trainMask) # Compute the truth value of NOT x element-wise.
    URMTest = csr_sparse_matrix(ratingList[testMask], userList[testMask], itemList[testMask])

    return URMTrain, URMTest


# Separate user, item, rating and timestamp
def row_split(rowString):
    # file format: 1::364::5::838983707
    split = rowString.split("::")
    split[3] = split[3].replace("\n", "")

    split[0] = int(split[0])
    split[1] = int(split[1])
    split[2] = float(split[2])
    split[3] = int(split[3])

    result = tuple(split)
    return result

# Matrix in COOrdinate format (fast format for constructing sparse matrices)
def csr_sparse_matrix(data, row, col):
    matrix = sps.coo_matrix((data, (row, col)))
    matrix = matrix.tocsr() # put in Compressed Sparse Row format for fast row access
    return matrix


# Statistics on interactions
def display_statistics(userList, itemList, URM):
    print("\nStatistics ... ")

    # Number of interactions in the URM
    URMFile.seek(0)
    numberInteractions = 0

    for _ in URMFile:
        numberInteractions += 1
    print("The number of interactions is {}".format(numberInteractions))

    userListUnique, itemListUnique = remove_duplicates(userList, itemList)
    numUsers = len(userListUnique)
    numItems = len(itemListUnique)

    print("Number of items\t {}, Number of users\t {}".format(numItems, numUsers))
    print("Max ID items\t {}, Max Id users\t {}\n".format(max(itemListUnique), max(userListUnique)))
    print("Average interactions per user {:.2f}".format(numberInteractions / numUsers))
    print("Average interactions per item {:.2f}\n".format(numberInteractions / numItems))

    print("Sparsity {:.2f} %\n".format((1 - float(numberInteractions) / (numItems * numUsers)) * 100))

    # Item popularity
    print("Item popularity ... \n")
    itemPopularity = flatten_array(URM)

    plot_data(itemPopularity, 'ro', 'Item Popularity', 'Num Interactions', 'Item Index')

    tenPercent = int(numItems / 10)

    print("\nAverage per-item interactions over the whole dataset {:.2f}".
          format(itemPopularity.mean()))

    print("Average per-item interactions for the top 10% popular items {:.2f}".
          format(itemPopularity[-tenPercent].mean()))

    print("Average per-item interactions for the least 10% popular items {:.2f}".
          format(itemPopularity[:tenPercent].mean()))

    print("Average per-item interactions for the median 10% popular items {:.2f}".
          format(itemPopularity[int(numItems * 0.45):int(numItems * 0.55)].mean()))

    print("Number of items with zero interactions {}".
          format(np.sum(itemPopularity == 0)))

    itemPopularityNonzero = itemPopularity[itemPopularity > 0]

    tenPercent = int(len(itemPopularityNonzero) / 10)

    print("\nAverage per-item interactions over the whole dataset {:.2f}".
          format(itemPopularityNonzero.mean()))

    print("Average per-item interactions for the top 10% popular items {:.2f}".
          format(itemPopularityNonzero[-tenPercent].mean()))

    print("Average per-item interactions for the least 10% popular items {:.2f}".
          format(itemPopularityNonzero[:tenPercent].mean()))

    print("Average per-item interactions for the median 10% popular items {:.2f}".
          format(itemPopularityNonzero[int(numItems * 0.45):int(numItems * 0.55)].mean()))

    plot_data(itemPopularityNonzero, 'ro', 'Item Popularity Nonzero' , 'Num Interactions', 'Item Index')

    # User Activity
    print("User activity ...\n")
    userActivity = flatten_array(URM)
    plot_data(userActivity, 'ro', 'User Activity', 'Num Interactions', 'User Index')


def rating_distribution_over_time(timestampList):
    print("Rating distribution over time ... ", end="\n")
    # Clone the list to avoid changing the ordering of the original data
    timestampSorted = list(timestampList)
    timestampSorted.sort()

    plot_data(timestampSorted, 'ro', 'Timestamp Sorted', 'Timestamp', 'Item Index')


def plot_data(data, marker, title, yLabel, xLabel):
    pyplot.plot(data, marker)
    pyplot.title(title)
    pyplot.ylabel(yLabel)
    pyplot.xlabel(xLabel)
    pyplot.show()


# Remove duplicates from list by using a set
def remove_duplicates(userList, itemList):
    userListUnique = list(set(userList))
    itemListUnique = list(set(itemList))

    return userListUnique, itemListUnique

# Flatten single dimensional entries in an array
def flatten_array(array):
    flattenedArray = (array > 0).sum(axis=0)
    flattenedArray = np.array(flattenedArray).squeeze()
    flattenedArray = np.sort(flattenedArray)

    return flattenedArray

def list_ID_stats(IDList, label):
    minVal = min(IDList)
    maxVal = max(IDList)
    uniqueVal = len(set(IDList))
    missingVal = 1 - uniqueVal / (maxVal - minVal)

    print("{} data, ID: min {}, max {}, unique {}, missig {:.2f} %".format(label, minVal, maxVal, uniqueVal,
                                                                           missingVal * 100))