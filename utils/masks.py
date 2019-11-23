#!/usr/bin/env python3
#  -*- coding: utf-8 -*-


# Remove low rating interactions
# ------------------------------
def get_positive_interations_only(URM, threshold=2):
	# NOTE: The train (or test) data may contain a lot of low rating interactions (<= threshold),
	# In reality we want to recommend items rated in a positive way (> threshold ),
	# so let's build a new data set with positive interactions only

	# Remove low rating interactions (<=2 values) from the URM_train (or URM_test),
	# now .data contains only values > threshold (positive interactions)
	URM_mask = URM.copy()
	URM_mask.data[URM_mask.data <= threshold] = 0  # set to 0 all elements that have a value less or equal to 3
	URM_mask.eliminate_zeros()  # remove those zero elements

	return URM_mask


# todo: cold items and cold users

## use boolean arrays as masks, to select particular subsets of the data themselves.
#  simply index on this Boolean array; this is known as a masking operation: x[x < 5],
# returns array filled with all the values that meet the condition (positions in which mask array is True)
# array([0, 3, 3, 3, 2, 4])



# Remove cold items, users or features from URM and/or since they do not provide info and help to reduce computations
# Remark:  Be careful! With this operation we lost the original mapping with item and user IDs!
# the masks will have different indices w.r.t to the original matrix

# def warm_items_mask(matrix):
#     warm_items_mask = np.ediff1d(matrix.tocsc().indptr) > 0
#     warm_items = np.arange(matrix.shape[1])[warm_items_mask]
#     warm_matrix = matrix[:, warm_items]
#
#     return warm_items, warm_matrix


# def warm_masks(URM_all=None, ICM_all=None, remove_items_no_features=False):
#
#     URM = URM_all
#     ICM = ICM_all
#
#     # URM items and users that have no interactions (cold)
#     if URM_all != None:
#         warm_items_mask = np.ediff1d(URM.tocsc().indptr) > 0
#         warm_items = np.arange(URM.shape[1])[warm_items_mask]
#         URM = URM[:, warm_items]
#
#         warm_users_mask = np.ediff1d(URM.tocsr().indptr) > 0
#         warm_users = np.arange(URM.shape[0])[warm_users_mask]
#         URM = URM[warm_users, :]
#
#         return warm_items, warm_users, URM
#
#     # ICM items or features that have no occurrencies (cold)
#     if URM_all != None and ICM_all != None:
#         warm_items_mask = np.ediff1d(URM.tocsc().indptr) > 0
#         warm_items = np.arange(URM.shape[1])[warm_items_mask]
#         ICM = ICM[warm_items, :]
#         ICM = ICM.tocsr()
#
#         warm_features_mask = np.ediff1d(ICM.tocsc().indptr) > 0
#         warm_features = np.arange(ICM.shape[1])[warm_features_mask]
#         ICM = ICM[:, warm_features]
#         ICM = ICM.tocsr()
#
#         return warm_items_mask, warm_features_mask, ICM
#
#
#     # There could be items without features. We might not remove them in some cases
#     if URM != None and ICM != None and remove_items_no_features == True:
#         print("... ... ...")
        # nofeatures_items_mask = np.ediff1d(ICM.tocsr().indptr) <= 0
        # nofeatures_items_mask.sum()
        # warm_items_mask_2 = np.ediff1d(ICM_all.tocsr().indptr) > 0
        # warm_items_2 = np.arange(ICM_all.shape[0])[warm_items_mask_2]
        #
        # ICM_all = ICM_all[warm_items_2, :]
        # ICM_all = ICM_all.tocsr()
        #
        # URM_all = URM_all[:, warm_items_2]
        # URM_all = URM_all.tocsr()
        #
        # warm_users_mask_2 = np.ediff1d(URM_all.tocsr().indptr) > 0
        # warm_users_2 = np.arange(URM_all.shape[0])[warm_users_mask_2]
        #
        # URM_all = URM_all[warm_users_2, :]
        # URM_all = URM_all.tocsr()

