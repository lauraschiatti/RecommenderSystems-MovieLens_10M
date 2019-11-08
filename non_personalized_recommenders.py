#!/usr/local/bin/python3.6
# -*- coding: utf-8 -*-

## Non-personalized Recommenders Systems ##

from utilities import data_preprocessing as data, evaluation_metrics as eval
import random_recommender as rr
import top_popular_recommender as tp

# Load and prepare ata: movielens 10 million dataset
userList, itemList, ratingList, timestampList, URMSparse = data.load_and_prepare_data()

# Statistics on interactions
# data.display_statistics(userList, itemList, URMSparse)
# data.rating_distribution_over_time(timestampList)

# Split dataset into train and test sets
URMTrain, URMTest = data.data_splitting(userList, itemList, ratingList, URMSparse, 0.80) # trainTestSplit = 0.80

# Get users ids
userListUnique, itemListUnique = data.get_user_item_unique(userList, itemList)


# ------------------------------------------------------------------ #
                ##### Random Recommender #####
# ------------------------------------------------------------------ #

# Train model
print("\nRandom recommender ... ", end="\n")
randomRecommender = rr.RandomRecommender()
randomRecommender.fit(URMTrain)

# Recommendations for a user
userId = userListUnique[1]
recommendedItems = randomRecommender.recommend(userId, at=5)


print("Recommended items", recommendedItems, end='\n')
relevantItems = eval.get_relevant_items(userId, URMTest) # relevant items for a given user
print("Relevant items", relevantItems)
isRelevant = eval.is_relevant(recommendedItems, relevantItems)
print("Are recommended items relevant?", isRelevant)

# Test model
eval.evaluate_algorithm(URMTest, randomRecommender, userListUnique)


# ------------------------------------------------------------------ #
                ##### Top Popular Recommender #####
# ------------------------------------------------------------------ #

# Train model
print("\nTop popular recommender ... ", end="\n")
topPopRecommender = tp.TopPopRecommender()
topPopRecommender.fit(URMTrain)

# Make k recommendations to 10 users
for id in userListUnique[0:10]:
    print(topPopRecommender.recommend(id, at=5))  # at = # items to recommended

# Test model
eval.evaluate_algorithm(URMTest, topPopRecommender, userListUnique, at=5)