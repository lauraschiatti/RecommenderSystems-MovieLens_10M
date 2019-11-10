#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

from sklearn import preprocessing
import numpy as np
import scipy.sparse as sps

from utils import data_preprocessing as data, evaluation as eval, IR_feature_weighting as fw
from recommenders import item_CBF_KNN_recommender as CBF

# Load ratings data
print("Ratings data ... ")
user_list, item_list, rating_list, timestamp_list = data.parse_data("ml-10M100K/ratings.dat", is_URM=True)

# Some stats
data.list_ID_stats(user_list, "User")
data.list_ID_stats(item_list, "Item")

# Build URM
URM = data.csr_sparse_matrix(rating_list, user_list, item_list)

# ---------- ---------- ---------- ---------- ---------- ---------- #
# Load tags data
print("Tags data ... ")
user_list_ICM, item_list_ICM, tag_list, timestamp_list_ICM = data.parse_data("ml-10M100K/tags.dat", is_URM=False)

# Some stats
data.list_ID_stats(user_list_ICM, "Users ICM")
data.list_ID_stats(item_list_ICM, "Items ICM")

# We can see that most users and items have no data associated to them
num_tags = len(set(tag_list))

print ("Number of tags\t {}, Number of item-tag tuples {}".format(num_tags, len(tag_list)))

print("\nData example:")
print("user_list_ICM", user_list_ICM[0:10])
print("item_list_ICM", item_list_ICM[0:10])
print("tag_list", tag_list[0:10])

# Build ICM
# The tags are string, we should translate them into numbers so we can use them as indices in the ICM¶
le = preprocessing.LabelEncoder() # encodes labels with value between 0 and n_classes-1.
le.fit(tag_list)
tag_list = le.transform(tag_list)
# print("encoded tag_list", tag_list[0:10])

n_items = URM.shape[1]
n_tags = max(tag_list) + 1

ICM_shape = (n_items, n_tags)

ones = np.ones(len(tag_list))
ICM = data.csr_sparse_matrix(ones, item_list_ICM, tag_list, ICM_shape)

#---------- ---------- ---------- ---------- ---------- ---------- #
# We leverage CSR and CSC indptr data structure to compute the number of cells that have values for that row or column¶
# Features per item
ICM = sps.csr_matrix(ICM)
features_per_item = np.ediff1d(ICM.indptr) # differences between consecutive elements of an array.
# print("Features Per Item", features_per_item.shape)
features_per_item = np.sort(features_per_item)
# data.plot_data(features_per_item, 'ro', 'Features Per Item', 'Num Features', 'Item Index')

# Items per feature
ICM = sps.csc_matrix(ICM)
items_per_feature = np.ediff1d(ICM.indptr)
# print("Items Per Feature", items_per_feature.shape)
items_per_feature = np.sort(items_per_feature)
# data.plot_data(items_per_feature, 'ro', 'Items Per Feature', 'Num Items', 'Feature Index')

ICM = sps.csr_matrix(ICM)

URM_train, URM_test = data.train_test_holdout(URM, train_perc = 0.8)

URM_train_csc = URM_train.tocsc()

num_tot_items = ICM.shape[0]
# print("num_tot_items", num_tot_items)

# ------------------------------------------------------------------ #
                ##### Content-based Recommender #####
# ------------------------------------------------------------------ #

# Train and test model

# Tuning model paramaters
tuning_param = input("Please choose tuning parameter: "
                     "1 - No tuning "
                     "2 - Shrinkage "
                     "3 - IDF weighted-ICM "
                     "4 - BM25 weighted-ICM "
                     "5 - Unnormalized similarity matrix ")

if tuning_param == "1":
    print("\nItem-CBFKNN recommender ... ", end="\n")
    ICBFKNNrecommender = CBF.ItemCBFKNNRecommender(URM_train, ICM)
    ICBFKNNrecommender.fit(shrink=0.0, topK=50)

    user_list_unique = data.remove_duplicates(user_list_ICM)
    for user_id in user_list_unique[0:10]:
        print("user", user_id)
        print(ICBFKNNrecommender.recommend(user_id, at=5))

    # Test model
    n_users_to_test = 1000
    print("Item-CBFKNN recommender response_time for 1000 users")
    eval.response_time(n_users_to_test, ICBFKNNrecommender)

    # Common mistake.... a CSC URM
    # URM_train_csc = URM_train.tocsc()
elif tuning_param == "2":
    # Shrinkage
    x_tick = [0, 10, 50, 100, 200, 500] # try different values to decide shrink factor
    MAP_per_shrinkage = []

    print("\nItem-CBFKNN recommender with shrinkage ... ", end="\n")
    for shrink in x_tick:
        ICBFKNNrecommender_shrinked = CBF.ItemCBFKNNRecommender(URM_train, ICM)
        ICBFKNNrecommender_shrinked.fit(shrink=shrink, topK=100)

        result_dict = eval.evaluate_algorithm(URM_test, ICBFKNNrecommender_shrinked)
        MAP_per_shrinkage.append(result_dict["MAP"])

    # The shrinkage value (i.e. support) have a much stronger impact.
    # Combine a parameter search with the two to ensure maximum recommendation quality
    data.plot_data({x_tick, MAP_per_shrinkage}, 'ro', '', 'MAP', 'Shrinkage')

elif tuning_param == "3": # IDF weighted-ICM
    # 1. IDF Feature weighting method
    items_per_feature = (ICM > 0).sum(axis=0)  # how many items have a certain feature

    idf = np.array(np.log(num_tot_items / items_per_feature))[0]  # calculate IDF

    # Compute weighted-ICM
    ICM_idf = ICM.copy()

    # NOTE: this works only if X is instance of sparse.csc_matrix
    col_nnz = np.diff(sps.csc_matrix(ICM_idf).indptr)  # number of non-zeros in each col
    ICM_idf.data *= np.repeat(idf, col_nnz)  # then normalize the values in each col by applying TF-IDF adjustment

    print("\nItem-CBFKNN recommender with IDF weighted-ICM ... ", end="\n")
    ICBFKNNrecommender_idf = CBF.ItemCBFKNNRecommender(URM_train, ICM_idf)
    ICBFKNNrecommender_idf.fit(shrink=0.0, topK=50)

    # There is a small gain over the non-weighted ICM.
    eval.evaluate_algorithm(URM_test, ICBFKNNrecommender_idf)

elif tuning_param == "4": # BM25 weighted-ICM
    # 2. BM25 Feature weighting method
    ICM_bm25 = ICM.copy().astype(np.float32)
    ICM_bm25 = fw.okapi_BM_25(ICM_bm25)
    ICM_bm25 = ICM_bm25.tocsr()

    print("\nItem-CBFKNN recommender with BM25 weighted-ICM ... ", end="\n")
    ICBFKNNrecommender_bm25 = CBF.ItemCBFKNNRecommender(URM_train, ICM_bm25)
    ICBFKNNrecommender_bm25.fit(shrink=0.0, topK=50)

    # Another small gain over TF-IDF
    eval.evaluate_algorithm(URM_test, ICBFKNNrecommender_bm25)

elif tuning_param == "5": # unnormalized similarity matrix
    recommender_dot = CBF.ItemCBFKNNRecommender(URM_train, ICM)
    recommender_dot.W_sparse = ICM * ICM.T  # .T: transposed array.

    eval.evaluate_algorithm(URM_test, recommender_dot)
