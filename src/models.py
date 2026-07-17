from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def get_models():
    models={
        'LogisticRegression':LogisticRegression(),
        'DecisionTree':DecisionTreeClassifier(),
        'RandomForest':RandomForestClassifier(),
        'SVC':SVC(),
        'KNNClassifier':KNeighborsClassifier()
    }
    return models


def train_evalute_models(model,X_train,X_test,y_train,y_test):
    results ={}
    for name_models,models in model.items():
        models.fit(X_train,y_train)
        score = models.score(X_test,y_test)
        results[name_models]=score
    return results

