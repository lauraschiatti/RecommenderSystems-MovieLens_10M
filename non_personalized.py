#!/usr/local/bin/python3.6
# -*- coding: utf-8 -*-

## Non-personalized Recommenders Systems ##

from utilities import data_preprocessing as data, evaluation_metrics as eval
import RandomRecommender as rr
import TopPopRecommender as tp
import GlobalEffectsRecommender as ge

# Load and prepare ata: movielens 10 million dataset
userList, itemList, ratingList, timestampList, URMSparse = data.load_and_prepare_data()

# Statistics on interactions
# data.display_statistics(userList, itemList, URMSparse)
# data.rating_distribution_over_time(timestampList)

# Split dataset into train and test sets
TRAIN_TEST = 0.80
URMTrain, URMTest = data.data_splitting(userList, itemList, ratingList, URMSparse, TRAIN_TEST)

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


# ------------------------------------------------------------------ #
                ##### Global Effects Recommender #####
# ------------------------------------------------------------------ #

# Train model
print("\nGlobal effects recommender ... ", end="\n")
globalEffectsRecommender = ge.GlobalEffectsRecommender()
globalEffectsRecommender.fit(URMTrain)

# Test model
# Remark: why is GlobalEffect performing worse than TopPop even if we are taking into account
# more information about the interaction?

# The test data contains a lot of low rating interactions...
# We are testing against those as well, but GlobalEffects is penalizing interactions with low rating
# In reality we want to recommend items rated in a positive way, so let's build a new Test set with positive interactions only
URMTestPositiveOnly = URMTest.copy()
URMTestPositiveOnly.data[URMTest.data<=2] = 0
URMTestPositiveOnly.eliminate_zeros()

print("Deleted {} negative interactions".format(URMTest.nnz - URMTestPositiveOnly.nnz))

print("evaluation of TopPopRecommender with URMTestPositiveOnly: ")
eval.evaluate_algorithm(URMTestPositiveOnly, topPopRecommender, userListUnique)

print("evaluation of globalEffectsRecommender with URMTestPositiveOnly: ")
# Sometimes ratings are not really more informative than interactions, depends on their quality
eval.evaluate_algorithm(URMTestPositiveOnly, globalEffectsRecommender, userListUnique)

# but GlobalEffects performs worse again... why?
# Sometimes ratings are not really more informative than interactions, depends on their quality




