# ------------------------------------------------------------------ #
       ##### Practice 2 - Non personalized recommenders #####
# ------------------------------------------------------------------ #

import utilities as util

userList, itemList, ratingList, timestampList = util.loadData()

# To store the data we use a sparse matrix.
URM_all = util.constructSparseMatrix()

# Statistics on user-item interactions
util.displayStatistics()
util.ratingDistributionOverTime()