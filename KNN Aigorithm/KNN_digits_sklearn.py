from sklearn import datasets, neighbors
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

digits = datasets.load_digits()
digits_X = digits.data  # data
digits_Y = digits.target  # lable

# random data
randIndex = np.arange(digits_X.shape[0])
np.random.shuffle(randIndex)
# create random dataset
digits_X = digits_X[randIndex]
digits_Y = digits_Y[randIndex]


X_train, X_test, Y_train, Y_test = train_test_split(digits_X, digits_Y, test_size=360)

knn = neighbors.KNeighborsClassifier(n_neighbors=5)
train_model = knn.fit(X_train, Y_train)

y_predict = train_model.predict(X_test)

print(Y_test)
print(y_predict)
# print(train_model.score(X_test,Y_test))
accuracy = accuracy_score(y_predict, Y_test)
print(accuracy)

# test
plt.gray()
plt.imshow(X_test[0].reshape(8, 8))
print(knn.predict(X_test[0].reshape(1, -1)))
plt.show()