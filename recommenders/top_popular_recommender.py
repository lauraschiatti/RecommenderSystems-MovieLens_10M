#!/usr/bin/env python3
#  -*- coding: utf-8 -*-
import numpy as np

# Recommends to all users the most popular items, those with the highest number of interactions (ratings)
class TopPopRecommender(object):
    # model is the item popularity
    def fit(self, URMTrain):
        self.URMTrain = URMTrain

        itemPopularity = (URMTrain > 0).sum(axis=0)
        itemPopularity = np.array(itemPopularity).squeeze()

        # We are not interested in sorting the popularity value,
        # but to order the items according to it
        self.popularItems = np.argsort(itemPopularity)
        self.popularItems = np.flip(self.popularItems, axis=0)

    def recommend(self, userId, at=5, removeSeen=True):
        # Simple but effective. Always remove seen items if your purpose is to recommend "new" ones
        if removeSeen:
            unseenItemsMask = np.in1d(self.popularItems, self.URMTrain[userId].indices,
                                      assume_unique=True, invert=True)

            unseenItems = self.popularItems[unseenItemsMask]
            recommendedItems = unseenItems[0:at]

        else:
            recommendedItems = self.popularItems[0:at]

        return recommendedItems

