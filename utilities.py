# -*- coding: utf-8 -*-
# -*- python 3.7.3 -*-

# Utilities Module
from urllib.request import urlretrieve
import zipfile, os
import matplotlib.pyplot as pyplot
import scipy.sparse as sps
import numpy as np

# ------------------------------------------------------------------ #
# Load and prepare data
# ------------------------------------------------------------------ #
def loadAndPrepareData():
    print("\nLoading data ... ", end="\n")

    # Download data
    dataFilePath = "data/Movielens_10M"
    dataFileName = dataFilePath + "/movielens_10m.zip"

    # If file exists, skip the download
    os.makedirs(dataFilePath, exist_ok=True) # create dir if not exists
    if not os.path.exists(dataFileName):
        urlretrieve("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dataFileName) # copy network object denoted by a URL to a local file

    dataFile = zipfile.ZipFile(dataFileName) # open zip file
    URMPath = dataFile.extract("ml-10M100K/ratings.dat", path="data/Movielens_10M") # extract data

    # Format data
    global URMFile
    URMFile = open(URMPath, 'r') # read file's content

    # Create a tuple for each interaction (line in the file)
    URMFile.seek(0)  # start from beginning of the file
    URMTuples = []

    for line in URMFile:
        URMTuples.append(rowSplit(line))

    # Separate the four columns in different independent lists
    userList, itemList, ratingList, timestampList = zip(*URMTuples)  # join tuples together (zip() to map values)

    # Convert values to list
    userList = list(userList)
    itemList = list(itemList)
    ratingList = list(ratingList)
    timestampList = list(timestampList)

    # To store the data we use a sparse matrix.
    # Build the matrix in COOrdinate format (fast format for constructing sparse matrices)
    URMSparse = sps.coo_matrix((ratingList, (userList, itemList))) # (data, (row, column))

    # Put the matrix in Compressed Sparse Row format for fast arithmetic and matrix vector operations
    URMSparse.tocsr()

    return userList, itemList, ratingList, timestampList, URMSparse

def dataSplitting(userList, itemList, ratingList, URMSparse, trainTestSplit):
    numInteractions = URMSparse.nnz  # number of nonzero values

    # take random samples of data.
    # Use random boolean mask. p trainTestSplit for True and 1-trainTestSplit for False
    trainMask = np.random.choice([True, False], numInteractions,
                                 p=[trainTestSplit, 1 - trainTestSplit]) # p: probabs associated with each entry in a

    userList = np.array(userList)
    itemList = np.array(itemList)
    ratingList = np.array(ratingList)

    URMTrain = sps.coo_matrix((ratingList[trainMask], (userList[trainMask], itemList[trainMask])))
    URMTrain = URMTrain.tocsr()

    testMask = np.logical_not(trainMask) # Compute the truth value of NOT x element-wise.

    URMTest = sps.coo_matrix((ratingList[testMask], (userList[testMask], itemList[testMask])))
    URMTest = URMTest.tocsr()

    return URMTrain, URMTest


# ------------------------------------------------------------------ #
# Display some statistics
# ------------------------------------------------------------------ #
def displayStatistics(userList, itemList, URMSparse):
    print("\nDisplay statistics ... ")

    # Number of interactions in the URM
    URMFile.seek(0)
    numberInteractions = 0

    for _ in URMFile:
        numberInteractions += 1
    print("The number of interactions is {} \n".format(numberInteractions))

    userListUnique, itemListUnique = getUserItemUnique(userList, itemList) # to convert set into a list

    numUsers = len(userListUnique)
    numItems = len(itemListUnique)

    print("Number of items\t {}, Number of users\t {}".format(numItems, numUsers))
    print("Max ID items\t {}, Max Id users\t {}\n".format(max(itemListUnique), max(userListUnique)))
    print("Average interactions per user {:.2f}".format(numberInteractions / numUsers))
    print("Average interactions per item {:.2f}\n".format(numberInteractions / numItems))

    print("Sparsity {:.2f} %\n".format((1 - float(numberInteractions) / (numItems * numUsers)) * 100))

    # Item Popularity
    itemPopularity = (URMSparse > 0).sum(axis=0)
    itemPopularity = np.array(itemPopularity).squeeze()
    itemPopularity = np.sort(itemPopularity)

    pyplot.plot(itemPopularity, 'ro')
    pyplot.ylabel('Num Interactions ')
    pyplot.xlabel('Item Index')
    pyplot.show()

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

    pyplot.plot(itemPopularityNonzero, 'ro')
    pyplot.ylabel('Num Interactions ')
    pyplot.xlabel('Item Index')
    pyplot.show()

    # User Activity
    userActivity = (URMSparse > 0).sum(axis=1)
    userActivity = np.array(userActivity).squeeze()
    userActivity = np.sort(userActivity)

    pyplot.plot(userActivity, 'ro')
    pyplot.ylabel('Num Interactions ')
    pyplot.xlabel('User Index')
    pyplot.show()

def ratingDistributionOverTime(timestampList):
    print("\nRating distribution over time ... ", end="\n")
    # Clone the list to avoid changing the ordering of the original data
    timestamp_sorted = list(timestampList)
    timestamp_sorted.sort()

    pyplot.plot(timestamp_sorted, 'ro')
    pyplot.ylabel('Timestamp ')
    pyplot.xlabel('Item Index')
    pyplot.show()

# Function to separate user, item, rating and timestamp ### file format: 1::364::5::838983707
def rowSplit(rowString):
    split = rowString.split("::")
    split[3] = split[3].replace("\n", "")

    split[0] = int(split[0])
    split[1] = int(split[1])
    split[2] = float(split[2])
    split[3] = int(split[3])

    result = tuple(split)
    return result

def getUserItemUnique(userList, itemList):
    userListUnique = list(set(userList))  # to convert set into a list
    itemListUnique = list(set(itemList))
    
    return userListUnique, itemListUnique