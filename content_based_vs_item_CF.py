#!/usr/bin/env python3
#  -*- coding: utf-8 -*-


import numpy as np

from utils import data_preprocessing as data, evaluation as eval
from recommenders import item_CBF_KNN_recommender as item_CBF
from recommenders.collaborative import item_CF_KNN_recommender as item_CF


# ------------------------------------------------------------------ #
        ##### Content-based vs Item-based CF Recommenders #####
# ------------------------------------------------------------------ #

# Load ratings data
print("Ratings data ... ")
user_list, item_list, rating_list, timestamp_list = data.parse_data("ml-10M100K/ratings.dat", is_URM=True)

# Load content information
print("Tags data ... ")
user_list_ICM, item_list_ICM, tag_list_ICM, timestamp_list_ICM = data.parse_data("ml-10M100K/tags.dat", is_URM=False)
# Encode tags
tag_list_ICM = data.label_encoder(tag_list_ICM)


# Build URM
URM = data.csr_sparse_matrix(rating_list, user_list, item_list)

# Remove cold items and users
warm_items_mask = np.ediff1d(URM.tocsc().indptr) > 0
warm_items = np.arange(URM.shape[1])[warm_items_mask]
URM = URM[:, warm_items]

# repeat the same process for users
warm_users_mask = np.ediff1d(URM.tocsr().indptr) > 0
warm_users = np.arange(URM.shape[0])[warm_users_mask]
URM = URM[warm_users, :]


# Build ICM
n_items = len(warm_items_mask)
n_tags = max(tag_list_ICM) + 1
ICM_shape = (n_items, n_tags)

ones = np.ones(len(tag_list_ICM))
ICM = data.csr_sparse_matrix(ones, item_list_ICM, tag_list_ICM, ICM_shape)

# keep only warm items in the ICM
ICM = ICM[warm_items, :]
ICM = ICM.tocsr()

# Also remove the features that have no occurrencies: cold features
warm_features_mask = np.ediff1d(ICM.tocsc().indptr) > 0
warm_features = np.arange(ICM.shape[1])[warm_features_mask]
# Don't forget to keep the mapping
ICM = ICM[:, warm_features]
ICM = ICM.tocsr()

# There could be items without features
nofeatures_items_mask = np.ediff1d(ICM.tocsr().indptr) <= 0
nofeatures_items_mask.sum()

# We might not remove them in some cases, but we will do it for our comparison
warm_items_mask_2 = np.ediff1d(ICM.tocsr().indptr) > 0
warm_items_2 = np.arange(ICM.shape[0])[warm_items_mask_2]

ICM = ICM[warm_items_2, :]
ICM = ICM.tocsr()

# Now we have to remove cold items and users from the URM
URM = URM[:, warm_items_2]
URM = URM.tocsr()

warm_users_mask_2 = np.ediff1d(URM.tocsr().indptr) > 0
warm_users_2 = np.arange(URM.shape[0])[warm_users_mask_2]

URM = URM[warm_users_2, :]
URM = URM.tocsr()


# --------- --------- --------- --------- --------- --------- --------- #

# Split URM for train and test
URM_train, URM_test = data.train_test_holdout(URM, train_perc = 0.8)

# Train and test
content_recommender = item_CBF.ItemCBFKNNRecommender(URM_train, ICM)
content_recommender.fit(shrink=100, topK=200)
collaborative_recommender = item_CF.ItemCFKNNRecommender(URM_train)
collaborative_recommender.fit(shrink=100, topK=200)

# MAP_per_k = []
# for topK in [50, 100, 200]:
#     print("topK = ", topK)
#     for shrink in [10, 50, 100]:
#         print("shrink = ", shrink)
#
#         content_recommender.fit(shrink=shrink, topK=topK)
#         collaborative_recommender.fit(shrink=shrink, topK=topK)
#
#         print("content_recommender")
#         result_dict = eval.evaluate_algorithm(URM_test, content_recommender)
#         MAP_per_k.append(result_dict["MAP"])
#
#         print("collaborative_recommender")
#         result_dict = eval.evaluate_algorithm(URM_test, collaborative_recommender)
#         MAP_per_k.append(result_dict["MAP"])
#     print("### ### ### ### ###")

###  Remark : Collaborative is outperforming content-based by a large margin, as we could expect

# Recommendations distribution among items
data.recommendations_distribution(URM, collaborative_recommender)

data.recommendations_distribution(URM, collaborative_recommender)