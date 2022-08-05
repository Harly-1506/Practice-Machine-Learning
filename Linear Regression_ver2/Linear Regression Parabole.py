import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Random data
A = [
    2,
    3,
    4,
    5,
    6,
    8,
    7,
    10,
    11,
    14,
    18,
    20,
    23,
    24,
    25,
    26,
    28,
    29,
    30,
    31,
    32,
    33,
    35,
    36,
    37,
]
b = [
    2,
    3,
    5,
    8,
    10,
    12,
    16,
    14,
    15,
    18,
    19,
    23,
    25,
    28,
    30,
    29,
    26,
    25,
    22,
    19,
    18,
    11,
    10,
    8,
    9,
]

# Visualize data
plt.plot(A, b, "ro")

# Change row vector to column vector
A = np.array([A]).T
b = np.array([b]).T

# Create A square
x_square = np.array([A[:, 0] ** 2]).T

# Create vector 1
ones = np.ones((A.shape[0], 1), dtype=np.int8)

# Combine x_square and A
A = np.concatenate((x_square, A), axis=1)

# Combine 1 and A
A = np.concatenate((A, ones), axis=1)
print(A)
print(A.shape)
# Use fomular
x = np.linalg.inv(A.transpose().dot(A)).dot(A.transpose()).dot(b)

# Test data to draw
x0 = np.linspace(1, 40, 10000)
y0 = x[0][0] * x0 * x0 + x[1][0] * x0 + x[2][0]

plt.plot(x0, y0)

# test
x_test = 12
y_test = x[0][0] * x_test * x_test + x[1][0] * x_test + x[2][0]
print(y_test)

plt.show()
