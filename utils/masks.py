#!/usr/bin/env python3
#  -*- coding: utf-8 -*-


import numpy as np

# Get mask of users with interactions
def get_users_with_interactions_mask(URM_train):

	# create a mask of positive interactions (How to build it depends on the data)
	URM_mask = URM_train.copy()
	URM_mask.data[URM_mask.data <= 3] = 0  # set to 0 all elements that have a value less or equal to 3
	URM_mask.eliminate_zeros()  # remove those zero elements

	n_users = URM_mask.shape[0]
	n_items = URM_mask.shape[1]

	return URM_mask, n_users, n_items
