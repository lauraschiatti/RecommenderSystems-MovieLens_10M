#!/usr/bin/env python3
#  -*- coding: utf-8 -*-


from utils import data_manager as data, evaluation as eval

from recommenders.machine_learning import SLIM_BPR_Recommender

# ------------------------------------------------------------------ #
        			##### SLIM BPR #####
# ------------------------------------------------------------------ #

# Load ratings data
user_list, item_list, rating_list, timestamp_list = data.parse_data("ml-10M100K/ratings.dat", is_URM=True)

# Build URM
URM = data.csr_sparse_matrix(rating_list, user_list, item_list)

# Train/test split
URM_train, URM_test = data.train_test_holdout(URM, train_perc = 0.8)

URM_train = URM_train[:,0:5000]
URM_test = URM_test[:,0:5000]


# Train and test model
print("training ... ")
recommender = SLIM_BPR_Recommender.SLIM_BPR_Recommender(URM_train)
recommender.fit(epochs=1)

eval.evaluate_algorithm(URM_test, recommender)





