#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

# Doesn't have anything to learn from the data: recommend at random items to each user
class RandomRecommender(object):

    def fit(self, URMTrain):
        self.num_items = URMTrain.shape[0]

    def recommend(self, user_id, at=5):
        recommended_items = np.random.choice(self.num_items, at)

        return recommended_items
