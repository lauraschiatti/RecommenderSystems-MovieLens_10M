# -*- coding: utf-8 -*-
import numpy as np

# Recommends to all users the highest rated items
class GlobalEffectsRecommender(object):

    def fit(self, URMTrain):
        self.URMTrain = URMTrain

        # Average of all ratings, or global average
        globalAverage = np.mean(URMTrain.data)
        # print("The global average is {:.2f}".format(globalAverage))

        # Subtract the average to all ratings
        URMTrainUnbiased = URMTrain.copy()
        URMTrainUnbiased.data -= globalAverage

        # User Bias: average rating for each user
        # Remark: user bias is essential in case of rating prediction but not relevant in case of TopK recommendations.
        userMeanRating = URMTrainUnbiased.mean(axis=1)
        userMeanRating = np.array(userMeanRating).squeeze()
        # userMeanRating = np.sort(userMeanRating[userMeanRating != 0.0])
        # data.plot_data(userMeanRating, 'ro', 'User Mean Rating', 'User Bias', 'User Index')


        # In order to apply the user bias we have to change the rating value
        # in the URM_train_unbiased inner data structures
        # If we were to write:
        # URM_train_unbiased[user_id].data -= user_mean_rating[user_id]
        # we would change the value of a new matrix with no effect on the original data structure
        for userId in range(len(userMeanRating)):
            start_position = URMTrainUnbiased.indptr[userId]
            end_position = URMTrainUnbiased.indptr[userId + 1]

            URMTrainUnbiased.data[start_position:end_position] -= userMeanRating[userId]


        # Item Bias: average rating for each item
        itemMeanRating = URMTrainUnbiased.mean(axis=0)
        itemMeanRating = np.array(itemMeanRating).squeeze()
        # itemMeanRating = np.sort(itemMeanRating[itemMeanRating != 0])
        # data.plot_data(itemMeanRating, 'ro', 'Item Mean Rating', 'Item Bias', 'Item Index')

        self.bestRatedItems = np.argsort(itemMeanRating)
        self.bestRatedItems = np.flip(self.bestRatedItems, axis=0)


    def recommend(self, userId, at=5, removeSeen=True):
        # Sort the items by their itemBias and use the same recommendation principle as in TopPop
        if removeSeen:
            unseenItemsMask = np.in1d(self.bestRatedItems, self.URMTrain[userId].indices,
                                        assume_unique=True, invert=True)

            unseenItems = self.bestRatedItems[unseenItemsMask]
            recommendedItems = unseenItems[0:at]

        else:
            recommendedItems = self.bestRatedItems[0:at]

        return recommendedItems


