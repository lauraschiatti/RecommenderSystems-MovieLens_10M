#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

from sklearn import preprocessing
import numpy as np
import scipy.sparse as sps

from utils import data_preprocessing as data, evaluation as eval

# Load ratings data
user_list, item_list, rating_list, timestamp_list = data.parse_data("ml-10M100K/ratings.dat", is_URM=True)
data.list_ID_stats(user_list, "User")
data.list_ID_stats(item_list, "Item")

# Build URM
URM = data.csr_sparse_matrix(rating_list, user_list, item_list)

# Load tags data
user_list_ICM, item_list_ICM, tag_list, timestamp_list_ICM = data.parse_data("ml-10M100K/tags.dat", is_URM=False)
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
le = preprocessing.LabelEncoder() # Encodes labels with value between 0 and n_classes-1.
le.fit(tag_list)
tag_list = le.transform(tag_list)

print("encoded tag_list", tag_list[0:10])

n_items = URM.shape[1]
n_tags = max(tag_list) + 1

ICM_shape = (n_items, n_tags)

ones = np.ones(len(tag_list))
ICM = data.csr_sparse_matrix(ones, item_list_ICM, tag_list, ICM_shape)

# We leverage CSR and CSC indptr data structure to compute the number of cells that have values for that row or column¶
# Features per item
ICM = sps.csr_matrix(ICM)
features_per_item = np.ediff1d(ICM.indptr)
print("Features Per Item", features_per_item.shape)
features_per_item = np.sort(features_per_item)
data.plot_data(features_per_item, 'ro', 'Features Per Item', 'Num Features', 'Item Index')

# Items per feature
ICM = sps.csc_matrix(ICM)
items_per_feature = np.ediff1d(ICM.indptr)
print("Items Per Feature", items_per_feature.shape)
items_per_feature = np.sort(items_per_feature)
data.plot_data(items_per_feature, 'ro', 'Items Per Feature', 'Num Items', 'Feature Index')

ICM = sps.csr_matrix(ICM)

URM_train, URM_test = data.train_test_holdout(URM, train_perc = 0.8)




# ------------------------------------------------------------------ #
                ##### Content-based Recommender #####
# ------------------------------------------------------------------ #

# Train model

# Test model