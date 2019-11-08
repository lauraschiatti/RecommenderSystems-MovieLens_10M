# -*- coding: utf-8 -*-
# ------------------------------------------------------------------ #
                ##### Recommenders Systems #####
# ------------------------------------------------------------------ #

import utilities as util
import random_recommender as rr

# Data: movielens 10 million dataset
userList, itemList, ratingList, timestampList, URMSparse = util.loadAndPrepareData()

# Statistics on user-item interactions
util.displayStatistics(userList, itemList, URMSparse)
util.ratingDistributionOverTime(timestampList)


# ------------------------------------------------------------------ #
                ##### Random Recommender #####
# ------------------------------------------------------------------ #
# Data splitting
URMTrain, URMTest = util.dataSplitting(userList, itemList, ratingList, URMSparse, 0.80) # trainTestSplit = 0.80

# # Evaluation metrics
userListUnique, itemListUnique = util.getUserItemUnique(userList, itemList)
userId = userListUnique[1]

randomRecommender = rr.RandomRecommender()

randomRecommender.fit(URMTrain)
recommendedItems = randomRecommender.recommend(userId, at=5)
print("\nRandom recommender ... ", end="\n")
print("Recommended items", recommendedItems)


# Evaluation metrics