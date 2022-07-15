from sklearn import tree

my_tree = tree.DecisionTreeClassifier()

feature = [
    [1, 3, 3, 7],
    [5, 2, 4, 6],
    [1, 2, 3, 6],
    [5, 4, 4, 3],
    [1, 4, 4, 7],
    [3, 2, 3, 7],
    [3, 3, 3, 6],
    [5, 2, 2, 7],
]
labels = [0, 1, 1, 0, 0, 0, 0, 1]

result = my_tree.fit(feature,labels)

predict = result.predict([[1,4,3,6]])

print(predict)