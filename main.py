#!/usr/local/bin/python3.6
# -*- coding: utf-8 -*-

## Non-personalized Recommenders Systems ##

from utilities import data_preprocessing as data, evaluation_metrics as eval
import random_recommender as rr

# Load and prepare ata: movielens 10 million dataset
userList, itemList, ratingList, timestampList, URMSparse = data.load_and_prepare_data()

# Statistics on interactions
# data.display_statistics(userList, itemList, URMSparse)
# data.rating_distribution_over_time(timestampList)


# ------------------------------------------------------------------ #
                ##### Random Recommender #####
# ------------------------------------------------------------------ #

URMTrain, URMTest = data.data_splitting(userList, itemList, ratingList, URMSparse, 0.80) # trainTestSplit = 0.80

userListUnique, itemListUnique = data.get_user_item_unique(userList, itemList)
userId = userListUnique[1]

randomRecommender = rr.RandomRecommender()
randomRecommender.fit(URMTrain)
recommendedItems = randomRecommender.recommend(userId, at=5)

print("\nRandom recommender ... ", end="\n")
print("Recommended items", recommendedItems, end='\n')


# Evaluation metrics
relevantItems = eval.get_relevant_items(userId, URMTest) # relevant items for a given user
print("Relevant items", relevantItems)
isRelevant = eval.is_relevant(recommendedItems, relevantItems)
print("Are recommended items relevant?", isRelevant)

eval.evaluate_algorithm(URMTest, randomRecommender, userListUnique)

