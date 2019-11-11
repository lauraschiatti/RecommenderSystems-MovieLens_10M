#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

from sklearn import preprocessing
import numpy as np

from utils import data_preprocessing as data, evaluation as eval
from recommenders.collaborative import item_CF_KNN_recommender as item_CF, user_CF_KNN_recommender as user_CF


# ------------------------------------------------------------------ #
##### Collaborative Recommenders: item-item and user-user CF #####
# ------------------------------------------------------------------ #

# Load ratings data
print("Ratings data ... ")
user_list, item_list, rating_list, timestamp_list = data.parse_data("ml-10M100K/ratings.dat", is_URM=True)

# Build URM
URM = data.csr_sparse_matrix(rating_list, user_list, item_list)

# Remove cold items and users (which have no interactions) because they have no impact in the evaluation
# Moreover, considering how item-item and user-user CF are defined, they are not relevant.

# keep only the warm items also in the ICM
warm_items_mask = np.ediff1d(URM.tocsc().indptr) > 0
warm_items = np.arange(URM.shape[1])[warm_items_mask]
URM = URM[:, warm_items]

# repeat the same process for users
warm_users_mask = np.ediff1d(URM.tocsr().indptr) > 0
warm_users = np.arange(URM.shape[0])[warm_users_mask]
URM = URM[warm_users, :]

# Be careful! With this operation we lost the original mapping with item and user IDs!
# Keep the warm_items and warm_users array, we might need them in future...

# Train/test split
URM_train, URM_test = data.train_test_holdout(URM, train_perc = 0.8)

# Train/test model
algorithm = input("Please choose a collaborative approach: "
                     "1 - Item-based CF "
                     "2 - User-based CF ")

recomm_type = item_CF.ItemCFKNNRecommender(URM_train)
label = "Item-based"
if algorithm == "2":
    recomm_type = user_CF.UserCFKNNRecommender(URM_train)
    label = "User-based"


# Parameters tuning
tuning_param = input("Please choose tuning parameter: "
                     "1 - No tuning "
                     "2 - Number of neighbors "
                     "3 - Shrinkage ")

if tuning_param == "1":
    print("\n{} CF recommender ...".format(label))

    # Train model
    recommender = recomm_type
    recommender.fit(shrink=0.0, topK=50)

    # Test model
    n_users_to_test = 1000
    print("{} CF recommender response_time for {} users".format(label, n_users_to_test))
    eval.response_time(n_users_to_test, recommender)


elif tuning_param == "2": # Number of neighbors
    print("\n{} CF recommender with neighbors tuning...".format(label))
    x_tick = [10, 50, 100, 200, 500] # try different values to decide k-neighbors
    MAP_per_k = []

    for topK in x_tick:
        recommender = recomm_type
        recommender.fit(shrink=0.0, topK=topK)

        result_dict = eval.evaluate_algorithm(URM_test, recommender)
        MAP_per_k.append(result_dict["MAP"])

    # On this dataset the number of neighbors has a great impact on MAP. Different datasets will behave in different ways
    data.plot_data(x_tick, 'ro', '', 'MAP', 'TopK')


elif tuning_param == "3": # Shrinkage
    print("\n{} CF recommender with shrinking factor...".format(label))
    x_tick = [0, 10, 50, 100, 200, 500]
    MAP_per_shrinkage = []

    for shrink in x_tick:
        recommender = recomm_type
        recommender.fit(shrink=shrink, topK=100)

        result_dict = eval.evaluate_algorithm(URM_test, recommender)
        MAP_per_shrinkage.append(result_dict["MAP"])

    data.plot_data({x_tick, MAP_per_shrinkage}, 'ro', '', 'MAP', 'Shrinkage')
