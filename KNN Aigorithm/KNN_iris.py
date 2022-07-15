from sklearn import datasets
import numpy as np
import math
import operator


def calculate_distances(p1, p2):
    dimension = len(p1)
    distance = 0

    for i in range(dimension):
        distance += (p1[i] - p2[i]) * (p1[i] - p2[i])

    return math.sqrt(distance)


def get_k_neighbors(training_X, lable_y, point, k):
    distances = []
    neighbors = []
    # tính khoảng cách từ point tới các điểm và lấy giá trị labels của các điểm đó
    for i in range(len(training_X)):
        distance = calculate_distances(training_X[i], point)
        distances.append((distance, lable_y[i]))

    # sắp xếp lại khoảng cách từ nhỏ tới lớn và lấy K labels của khoảng cách đó
    distances.sort(key=operator.itemgetter(0))
    for i in range(k):
        neighbors.append(distances[i][1])

    return neighbors


def highest_votes(lables):
    lables_count = [0, 0, 0]
    for lable in lables:
        lables_count[lable] += 1

    max_count = max(lables_count)

    return lables_count.index(max_count)


def predict(training_X, lable_y, point, k):
    # lấy labels của những điểm gần nhất
    neighbors_lables = get_k_neighbors(training_X, lable_y, point, k)
    # trả về label phổ biến nhất
    return highest_votes(neighbors_lables)


def accuracy_score(predicts, lables):
    total = len(predicts)
    correct_count = 0
    for i in range(total):
        if predicts[i] == lables[i]:
            correct_count += 1
            accuracy = correct_count / total * 100
    return accuracy


iris = datasets.load_iris()
iris_X = iris.data  # data (petal lenght, petal width, sepal lenght, senpal width)
iris_Y = iris.target  # lable

# random data
randIndex = np.arange(iris_X.shape[0])
np.random.shuffle(randIndex)
# create random dataset
iris_X = iris_X[randIndex]
iris_Y = iris_Y[randIndex]

# divide data to train and test
X_train = iris_X[:100, :]
X_test = iris_X[100:, :]
y_train = iris_Y[:100]
y_test = iris_Y[100:]

k = 5
y_predicts = []  # labels's model for predict X_test
# Use X-test to test model
for p in X_test:
    lable = predict(X_train, y_train, p, k)
    y_predicts.append(lable)

print(y_test)
print(y_predicts)

acc = accuracy_score(y_predicts, y_test)
print(acc)
