import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn import linear_model

# random data
A = [2, 3, 4, 6, 7, 8, 10, 12, 14, 17, 18, 22, 25, 27, 32, 35]
b = [1, 4, 5, 7, 9, 10, 11, 19, 22, 23, 17, 28, 31, 38, 40, 41]

# Visualize data
plt.plot(A, b, "ro")

# Change row vector to column vector
A = np.array([A]).T
b = np.array([b]).T

lr = linear_model.LinearRegression()
lr.fit(A, b)

print(lr.intercept_)
print(lr.coef_)

plt.plot(A, b, "ro")

x0 = np.array([[1, 40]]).T
y0 = lr.coef_ * x0 + lr.intercept_

plt.plot(x0, y0)

plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn import linear_model

# # random data
# B = [
#     2,
#     5,
#     7,
#     9,
#     11,
#     16,
#     19,
#     23,
#     22,
#     29,
#     29,
#     35,
#     37,
#     40,
#     46,
#     42,
#     39,
#     31,
#     30,
#     28,
#     20,
#     15,
#     10,
#     6,
# ]
# A = [
#     2,
#     3,
#     4,
#     5,
#     6,
#     7,
#     8,
#     9,
#     10,
#     11,
#     12,
#     13,
#     14,
#     15,
#     16,
#     17,
#     18,
#     19,
#     20,
#     21,
#     22,
#     23,
#     24,
#     25,
# ]
# A = np.array([A]).T
# B = np.array([B]).T
# plt.plot(A, B, "ro")
# X_square = np.array([A[:, 0] ** 2]).T
# A = np.concatenate((X_square, A), axis=1)
# lr = linear_model.LinearRegression()
# lr.fit(A, B)
# print(lr.intercept_)
# print(lr.coef_)
# x_0 = np.linspace(1, 25, 10000)
# print(x_0.shape)
# y_0 = lr.coef_[0][0] * x_0 * x_0 + lr.coef_[0][1] * x_0 + lr.intercept_
# plt.plot(x_0, y_0)
# plt.show()