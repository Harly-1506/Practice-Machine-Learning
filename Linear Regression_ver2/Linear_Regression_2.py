import pandas as pd
import matplotlib.pyplot as plt


def predict(new_radio, weight, bias):
    return weight * new_radio + bias


def cost_function(X, y, weight, bias):
    n = len(X)
    sum_error = 0
    for i in range(n):
        sum_error += (y[i] - (weight * X[i] + bias)) ** 2

    return sum_error / n


def update_weight(X, y, weight, bias, learning_rate):
    n = len(X)
    weight_temp = 0.0
    bias_temp = 0.0
    for i in range(n):
        weight_temp += -2 * X[i] * (y[i] - (X[i] * weight + bias))
        bias_temp += -2 * (y[i] - (X[i] * weight + bias))

    weight -= (weight_temp / n) * learning_rate
    bias -= (bias / n) * learning_rate

    return weight, bias


def training(X, y, weight, bias, learning_rate, iter):
    # cost_his = []
    list_w = []
    for i in range(iter):
        weight, bias = update_weight(X, y, weight, bias, learning_rate)
        # cost = cost_function(X, y, weight, bias)
        # cost_his.append(cost)
        list_w.append([weight, bias])
    return list_w


df = pd.read_csv(
    "H:\Python\Python Practice\Basic_ML\Linear Regression_ver2\Advertising.csv"
)

X = df.values[:, 2]
y = df.values[:, 4]

list_w = training(X, y, 0.03, 0.0014, 0.001, 40)
print(list_w)
plt.plot(X, y, "ro")
plt.show()