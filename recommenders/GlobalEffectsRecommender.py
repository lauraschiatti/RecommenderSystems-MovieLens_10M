#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

# Recommends to all users the items with highest average rating

class GlobalEffectsRecommender(object):
    # works for explicit ratings
    def fit(self, URM_train):
        self.URM_train = URM_train

        # 1) global average: average of all ratings
        mu = np.mean(URM_train.data)

        # remove mu from the URM (subs mu to all ratings)
        URM_train_unbiased = URM_train.copy()
        URM_train_unbiased.data -= mu

        # 2) user average bias: average rating for each user
        # NOTE: user bias is essential in case of rating prediction, but not relevant in case of TopN recommendations.

        user_bias = URM_train_unbiased.mean(axis=1)
        user_bias = np.array(user_bias).squeeze()

        # remove usr bias from the URM (subs mu to all ratings)
        for user_id in range(len(user_bias)):
            start_position = URM_train_unbiased.indptr[user_id]
            end_position = URM_train_unbiased.indptr[user_id + 1]

            URM_train_unbiased.data[start_position:end_position] -= user_bias[user_id]


        # 3) item average bias: average rating for each item
        item_bias = URM_train_unbiased.mean(axis=0)
        item_bias = np.array(item_bias).squeeze()

        # 4) precompute the item ranking
        self.best_rated_items = np.argsort(item_bias)
        self.best_rated_items = np.flip(self.best_rated_items, axis=0)

        # NOTE: plotting
        # user_bias = np.sort(user_bias[user_bias != 0.0])
        # data.plot_data(user_bias, 'ro', 'User Mean Rating', 'User Bias', 'User Index')

        # NOTE: plotting
        # item_bias = np.sort(item_bias[item_bias != 0])
        # data.plot_data(item_bias, 'ro', 'Item Mean Rating', 'Item Bias', 'Item Index')


    def recommend(self, user_id, at=5, remove_seen=True):
        # Sort the items by their item_bias and use the same recommendation principle as in TopPop
        user_seen_items = self.URM_train[user_id].indices

        if remove_seen:
            unseen_items_mask = np.in1d(self.best_rated_items, user_seen_items,
                                        assume_unique=True, invert=True)

            unseen_items = self.best_rated_items[unseen_items_mask]
            recommended_items = unseen_items[0:at]

        else:
            recommended_items = self.best_rated_items[0:at]

        return recommended_items


