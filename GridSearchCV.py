import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_digits
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from pipelinehelper import PipelineHelper
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings("ignore")


data = pd.read_csv('.../sonar.all-data')

x_data = data.iloc[:, :-1].values
y_data = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(
    x_data, y_data, train_size=0.8)

lb = LabelEncoder()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

pipe = Pipeline([
    ('scaler', PipelineHelper([
        ('std', StandardScaler()),
        ('max', MaxAbsScaler()),
    ])),
    ('classifier', PipelineHelper([
        ('svm', SVC()),

        ('mlp', MLPClassifier()),
        ('dtree', DecisionTreeClassifier())

    ])),
])

parameters = {
    'scaler__selected_model': pipe.named_steps['scaler'].generate({
        'std__with_mean': [True, False],
        'std__with_std': [True, False],
        'max__copy': [True],
    }),
    'classifier__selected_model': pipe.named_steps['classifier'].generate({
        'mlp__hidden_layer_sizes': [(100,)],
        'mlp__activation': ['relu', 'tanh'],
        'mlp__solver': ['adam'],

        'dtree__criterion': ["gini", "entropy"],

        'svm__C': [10.0],
        'svm__kernel': ['sigmoid'],
        'svm__degree': [2, 5],


    })
}
# Use GirdSearchCv to choose best model and param and K-fold for X_train
grid = GridSearchCV(pipe, parameters, cv=5, scoring='accuracy', verbose=1)
grid.fit(X_train, y_train)

print("-"*80)
print("Cấu hình tốt nhất: ", grid.best_params_)
print("Độ chính xác: {:6.2f}".format(grid.score(X_test, y_test)))
