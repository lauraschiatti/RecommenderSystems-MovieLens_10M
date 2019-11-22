#!/usr/local/bin/python3.6
# -*- coding: utf-8 -*-

# gradient_descent.py: module for applying gradient descent algorithm.

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import random
from scipy import stats
from scipy.optimize import fmin

# Gradient descent to find local minimums
# def gradient_descent():


f = lambda x: x**3-2*x**2+2


x = np.linspace(start=-1, stop=2.5, num=1000) # num evenly spaced samples, calculated over the interval [start, stop].

# Plot function
# plt.plot(x,f(x))
# plt.xlim([-1,2.5])
# plt.ylim([0,3])
# plt.show()

# find_local_min()

x_old = 0
x_new = 2  # random starting point x_0=2
n_k = 0.1  # step size
precision = 0.0001

x_list, y_list = [x_new], [f(x_new)]


# returns the value of the derivative of our function
def f_gradient(x):
	return 3 * x ** 2 - 4 * x


while abs(x_new - x_old) > precision:
	x_old = x_new

	# Gradient descent step
	s_k = -f_gradient(x_old) # direction of negative gradient

	x_new = x_old + n_k * s_k # update rule

	x_list.append(x_new)
	y_list.append(f(x_new))

print("Local minimum occurs at: {:.2f}".format(x_new))
print("Number of steps:", len(x_list))

# The figures below show the route that was taken to find the local minimum.

plt.figure(figsize=[10,3])
plt.subplot(1,2,1)
plt.scatter(x_list,y_list,c="r")
plt.plot(x_list,y_list,c="r")
plt.plot(x,f(x), c="b")
plt.xlim([-1,2.5])
plt.ylim([0,3])
plt.title("Gradient descent")

plt.subplot(1,2,2)
plt.scatter(x_list,y_list,c="r")
plt.plot(x_list,y_list,c="r")
plt.plot(x,f(x), c="b")
plt.xlim([1.2,2.1])
plt.ylim([0,3])
plt.title("Gradient descent (zoomed in)")
plt.show()