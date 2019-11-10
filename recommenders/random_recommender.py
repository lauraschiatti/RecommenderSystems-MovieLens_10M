#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

# Doesn't have anything to learn from the data
class RandomRecommender(object):

    def fit(self, URMTrain):
        self.numItems = URMTrain.shape[0]

    def recommend(self, userId, at=5):
        recommendedItems = np.random.choice(self.numItems, at)

        return recommendedItems