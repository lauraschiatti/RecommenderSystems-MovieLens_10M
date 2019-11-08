# -*- coding: utf-8 -*-
import numpy as np

# In a random recommend we don't have anything to learn from the data
class RandomRecommender(object):

    def fit(self, URM_train):
        self.numItems = URM_train.shape[0]

    def recommend(self, user_id, at=5):
        recommendedItems = np.random.choice(self.numItems, at)

        return recommendedItems