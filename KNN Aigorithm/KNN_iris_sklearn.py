from sklearn import datasets, neighbors
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
iris_X = iris.data  # data (petal lenght, petal width, sepal lenght, senpal width)
iris_Y = iris.target  # lable

# random data
randIndex = np.arange(iris_X.shape[0])
np.random.shuffle(randIndex)
# create random dataset
iris_X = iris_X[randIndex]
iris_Y = iris_Y[randIndex]


X_train, X_test, Y_train, Y_test = train_test_split(iris_X, iris_Y, test_size=50)

knn = neighbors.KNeighborsClassifier(n_neighbors=5)
train_model = knn.fit(X_train, Y_train)

y_predict = train_model.predict(X_test)

print(Y_test)
print(y_predict)
# print(train_model.score(X_test,Y_test))
accuracy = accuracy_score(y_predict, Y_test)
print(accuracy)