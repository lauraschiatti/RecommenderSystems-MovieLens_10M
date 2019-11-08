#!/usr/local/bin/python3.6
# -*- coding: utf-8 -*-

# evaluation_metrics.py: module for evaluating performance of recommenders.

import numpy as np

# Get test set relevantItems
def get_relevant_items(userId, URMTest):
    relevantItems = URMTest[userId].indices
    return relevantItems


# Test whether recommended items are relevant
def is_relevant(recommendedItems, relevantItems):
    isRelevant = np.in1d(recommendedItems, relevantItems, assume_unique=True) # compare elements in both arrays

    return isRelevant


# Precision: how many of the recommended items are relevant?
def precision(recommendedItems, relevantItems):
    isRelevant = is_relevant(recommendedItems, relevantItems)
    precisionScore = np.sum(isRelevant, dtype=np.float32) / len(isRelevant)

    return precisionScore


# Recall: how many of the relevant items I was able to recommend?
def recall(recommendedItems, relevantItems):
    isRelevant = is_relevant(recommendedItems, relevantItems)
    recallScore = np.sum(isRelevant, dtype=np.float32) / relevantItems.shape[0]

    return recallScore


# Mean Average Precision
def MAP(recommendedItems, relevantItems):
    isRelevant = is_relevant(recommendedItems, relevantItems)

    # Cumulative sum: precision at k=1, at k=2, at k=3 ...
    pAtk = isRelevant * np.cumsum(isRelevant, dtype=np.float32) / (1 + np.arange(isRelevant.shape[0]))
    mapScore = np.sum(pAtk) / np.min([relevantItems.shape[0], isRelevant.shape[0]])

    return mapScore


def evaluate_algorithm(URMTest, recommenderObject, userListUnique, at=5):
    cumulativePrecision = 0.0
    cumulativeRecall = 0.0
    cumulativeMAP = 0.0
    numEval = 0

    for userId in userListUnique:
        relevantItems = URMTest[userId].indices

        if len(relevantItems) > 0:
            recommendedItems = recommenderObject.recommend(userId, at=at)
            numEval += 1

            cumulativePrecision += precision(recommendedItems, relevantItems)
            cumulativeRecall += recall(recommendedItems, relevantItems)
            cumulativeMAP += MAP(recommendedItems, relevantItems)

    cumulativePrecision /= numEval
    cumulativeRecall /= numEval
    cumulativeMAP /= numEval

    print("Recommender performance is: Precision = {:.4f}, Recall = {:.4f}, MAP = {:.4f}".format(
        cumulativePrecision, cumulativeRecall, cumulativeMAP))

