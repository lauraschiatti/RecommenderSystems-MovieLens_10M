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


