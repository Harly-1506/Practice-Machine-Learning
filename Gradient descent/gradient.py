import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import linear_model


def loss(x):
    m = A.shape[0]
    return 0.5 / m * np.linalg.norm(A.dot(x) - b, 2) ** 2


def grad(x):
    m = A.shape[0]
    return 1 / m * A.T.dot(A.dot(x) - b)


def gradient_descent(x_init, learning_rate, iteration):
    x_list = [x_init]
    m = A.shape[0]
    for i in range(iteration):
        x_new = x_list[-1] - learning_rate * grad(x_list[-1])
        if np.linalg.norm(grad(x_new)) / len(x_new) < 0.3:
            break

        x_list.append(x_new)

    return x_list


A = np.array([[2, 9, 7, 9, 11, 16, 25, 23, 22, 29, 28, 35, 37, 40, 46]]).T
b = np.array([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]).T

print(A)
fig1 = plt.figure("GD for Linear Regression")
ax = plt.axes(xlim=(-10, 60), ylim=(-1, 20))
plt.plot(A, b, "ro")

lr = linear_model.LinearRegression()
lr.fit(A, b)

x0_gd = np.linspace(1, 46, 2)
y0_sklearn = lr.intercept_[0] + lr.coef_[0][0] * x0_gd
plt.plot(x0_gd, y0_sklearn, color="green")
print(x0_gd)
# Add one to A
ones = np.ones((A.shape[0], 1), dtype=np.int8)
A = np.concatenate((ones, A), axis=1)

print(A)
# random initial line
x_init = np.array([[1], [2]])
y0_init = x_init[0] + x_init[1] * x0_gd

plt.plot(x0_gd, y0_init, color="black")

# Run gradient descent
iteration = 100
learning_rate = 0.0001
x_list = gradient_descent(x_init, learning_rate, iteration)

for i in range(len(x_list)):
    y0_x_list = x_list[i][0] + x_list[i][1] * x0_gd
    plt.plot(x0_gd, y0_x_list, color="black")

plt.show()
# Plot cost per iteration to determine when to stop
loss_list = []
iter_list = []
for i in range(len(x_list)):
    iter_list.append(i)
    loss_list.append(loss(x_list[i]))

plt.plot(iter_list, loss_list)
plt.xlabel("Iteration")
plt.ylabel("loss_list")
plt.show()
