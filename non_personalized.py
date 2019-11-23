#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

from utils import data_manager as data, evaluation as eval, masks
from recommenders import RandomRecommender, TopPopularRecommender, GlobalEffectsRecommender

# Build URM
user_list, item_list, rating_list, timestamp_list = data.parse_data("ml-10M100K/ratings.dat", True)

# For items in particular most have no interactions.
# Sometimes it may be better to remove them to avoid creating big data structures with no need.
# In this case empty columns will nave no impact and we leave them as is.
URM = data.csr_sparse_matrix(rating_list, user_list, item_list)

# Statistics on interactions
data.get_statistics_URM(user_list, item_list, URM)
# data.rating_distribution_over_time(timestamp_list)

# Train/test split
URM_train, URM_test = data.train_test_holdout(URM, train_perc = 0.8)
user_list_unique = data.remove_duplicates(user_list)



# ------------------------------------------------------------------ #
                ##### Random Recommender #####
# ------------------------------------------------------------------ #

# Train model
print("\nRandom recommender ... ", end="\n")
randomRecommender = RandomRecommender.RandomRecommender()
randomRecommender.fit(URM_train)

# Test model
eval.evaluate_algorithm(URM_test, randomRecommender)

# Recommendations for a user
userId = user_list_unique[1]
recommendedItems = randomRecommender.recommend(userId, at=5)

print("Recommended items", recommendedItems, end='\n')
relevantItems = eval.get_relevant_items(userId, URM_test) # relevant items for a given user
print("Relevant items", relevantItems)
isRelevant = eval.get_is_relevant(recommendedItems, relevantItems)
print("Are recommended items relevant?", isRelevant)


# ------------------------------------------------------------------ #
                ##### Top Popular Recommender #####
# ------------------------------------------------------------------ #

# Train
print("\nTop popular recommender ... ", end="\n")
topPopRecommender = TopPopularRecommender.TopPopRecommender()
topPopRecommender.fit(URM_train)

# Make k recommendations to 10 users
for id in user_list_unique[0:10]:
    print(topPopRecommender.recommend(id, at=5, remove_seen=False))  # at = # items to recommended

# Test model
eval.evaluate_algorithm(URM_test, topPopRecommender, at=5)

# Train removing seen items
print("\nTop popular recommender removing seen items... ", end="\n")
topPopRecommender_remove_seen = TopPopularRecommender.TopPopRecommender()
topPopRecommender_remove_seen.fit(URM_train)

for user_id in user_list_unique[0:10]:
    print(topPopRecommender_remove_seen.recommend(user_id, at=5, remove_seen=True))

eval.evaluate_algorithm(URM_test, topPopRecommender_remove_seen)


# ------------------------------------------------------------------ #
                ##### Global Effects Recommender #####
# ------------------------------------------------------------------ #

# Train model
print("\nGlobal effects recommender ... ", end="\n")
globalEffectsRecommender = GlobalEffectsRecommender.GlobalEffectsRecommender()
globalEffectsRecommender.fit(URM_train)
eval.evaluate_algorithm(URM_test, globalEffectsRecommender)

# GlobalEffects vs TopPop
# why is GlobalEffect performing worse than TopPop even if we are taking into account
# more information about the interaction?

# NOTE: Remove low rating interactions (<=2 values) from the URM_test,
# low interactions are penalized by GlobalEffects
URM_test_positive_only = masks.get_positive_interations_only(URM_test, threshold=2)

print("Deleted {} negative interactions".format(URM_test.nnz - URM_test_positive_only.nnz))

print("evaluation of TopPopRecommender with URM_test_positive_only: ")
eval.evaluate_algorithm(URM_test_positive_only, topPopRecommender)

print("evaluation of globalEffectsRecommender with URM_test_positive_only: ")
eval.evaluate_algorithm(URM_test_positive_only, globalEffectsRecommender)

# but GlobalEffects performs worse again... why?
# Sometimes ratings are not really more informative than interactions, depends on their quality

