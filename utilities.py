# Utilities Module

from urllib.request import urlretrieve
import zipfile, os
import matplotlib.pyplot as pyplot
import scipy.sparse as sps


# Movielens 10 million dataset
# global variables
URM_file = "" # User Rating Matrix
userList = ""
itemList = ""
ratingList = ""
timeStampList = ""

# ------------------------------------------------------------------ #
# Load and prepare data
# ------------------------------------------------------------------ #

def loadData():
    print("\nLoading data ... ")

    # If file exists, skip the download
    data_file_path = "data/Movielens_10M"
    data_file_name = data_file_path + "/movielens_10m.zip"

    os.makedirs(data_file_path, exist_ok=True) # create dir if not exists
    if not os.path.exists(data_file_name):
        # Copy a network object denoted by a URL to a local file
        urlretrieve("http://files.grouplens.org/datasets/movielens/ml-10m.zip", data_file_name)

    dataFile = zipfile.ZipFile(data_file_name) # open zip file
    URM_path = dataFile.extract("ml-10M100K/ratings.dat", path="data/Movielens_10M") # extract data

    # read file's content
    global URM_file
    URM_file = open(URM_path, 'r')

    # Create a tuple for each interaction (line in the file)
    URM_file.seek(0)  # start from beginning of the file
    URM_tuples = []

    for line in URM_file:
        URM_tuples.append(rowSplit(line))

    # Separate the four columns in different independent lists
    global userList, itemList, ratingList, timestampList
    userList, itemList, ratingList, timestampList = zip(*URM_tuples)  # join tuples together (zip() to map values)

    # Convert values to list
    userList = list(userList)
    itemList = list(itemList)
    ratingList = list(ratingList)
    timestampList = list(timestampList)

    # Print first 10 tuples of the resulting lists
    print("The user list is : ", userList[0:10])
    print("The item list is : ", itemList[0:10])
    print("The rating list is : ", ratingList[0:10])
    print("The timestamp list is : ", timestampList[0:10], end=' \n')

    return userList, itemList, ratingList, timestampList

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

def constructSparseMatrix():
    # Build the matrix in COOrdinate format (fast format for constructing sparse matrices)
    # The COO constructor expects (data, (row, column))
    URM_all = sps.coo_matrix((ratingList, (userList, itemList)))

    # Put the matrix in Compressed Sparse Row format for fast arithmetic and matrix vector operations
    URM_all.tocsr()

    print(URM_all)

    return URM_all


# ------------------------------------------------------------------ #
# Display some statistics
# ------------------------------------------------------------------ #
def displayStatistics():
    print("\nDisplay statistics ... ")

    # get number of interactions in the URM
    URM_file.seek(0)
    numberInteractions = 0

    for _ in URM_file:
        numberInteractions += 1
    print("The number of interactions is {} \n".format(numberInteractions))

    userList_unique = list(set(userList))  # to convert set into a list
    itemList_unique = list(set(itemList))

    numUsers = len(userList_unique)
    numItems = len(itemList_unique)

    print("Number of items\t {}, Number of users\t {}".format(numItems, numUsers))
    print("Max ID items\t {}, Max Id users\t {}\n".format(max(itemList_unique), max(userList_unique)))
    print("Average interactions per user {:.2f}".format(numberInteractions / numUsers))
    print("Average interactions per item {:.2f}\n".format(numberInteractions / numItems))

    print("Sparsity {:.2f} %".format((1 - float(numberInteractions) / (numItems * numUsers)) * 100))


def ratingDistributionOverTime():
    print("\nRating distribution over time ... ")
    # Clone the list to avoid changing the ordering of the original data
    timestamp_sorted = list(timestampList)
    timestamp_sorted.sort()

    pyplot.plot(timestamp_sorted, 'ro')
    pyplot.ylabel('Timestamp ')
    pyplot.xlabel('Item Index')
    pyplot.show()


