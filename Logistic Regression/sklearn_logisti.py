from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessin

from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report


X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)


nor = preprocessing.Normalizer().fit(X_train)
X_train = nor.transform(X_train)

clf = LogisticRegression(fit_intercept=True).fit(X_train, y_train)

y_pred = clf.predict(X_test)


cmatrix = confusion_matrix(y_test, y_pred, )
precision = metrics.precision_score(y_test, y_pred, average='macro')
recall = metrics.recall_score(y_test, y_pred, average='macro')
f1 = metrics.f1_score(y_test, y_pred, average='macro')
accuracy = metrics.accuracy_score(y_test, y_pred, normalize=True)
sns.heatmap(cmatrix, cmap='afmhot', fmt="d", annot=True)

print("Confusion matrix ")
print(cmatrix)
print("Precision: {:7.2f}%".format(precision*100))
print("Recall: {:7.2f}%".format(recall*100))
print("F1-measure: {:7.2f}%".format(f1*100))
print("Accuracy: {:7.2f}%".format(accuracy*100))
