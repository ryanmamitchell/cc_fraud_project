from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import os
os.chdir('/Users/Ryan/PycharmProjects/cc_fraud_project')
# Import data, create test/train splits
data_x = pd.read_csv("data/clean_data.csv")
data_y = pd.read_csv("data/clean_data.csv")

X = data_x
X.drop('Class', axis=1, inplace=True)
y = data_y['Class']


# Feature Selection

best_feat = SelectKBest(score_func = f_classif, k=10)
fit = best_feat.fit(X,y)

cols_idxs = best_feat.get_support(indices=True)
X_10 = X.iloc[:,cols_idxs]



X_train, X_test, y_train, y_test = train_test_split(X_10, y, test_size=.80, random_state=6969)


# Function for model training

def model_train(X_train: pd.DataFrame, y_train: pd.DataFrame,
                X_test: pd.DataFrame, y_test: pd.DataFrame) -> pd.DataFrame:
    '''
    Lightweight script to test many models and find winners
        :param X_train: training split
        :param y_train: training target vector
        :param X_test: test split
        :param y_test: test target vector
        :return: DataFrame of predictions
    '''

    dfs = []

    models = [
        ("LogReg", LogisticRegression(max_iter=10000)),
        ("RF", RandomForestClassifier()),
        ("KNN", KNeighborsClassifier()),
        ("SVM", SVC()),
        ("GNB", GaussianNB()),
        ("XGB", XGBClassifier())

    ]

    results = []

    names = []

    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    target_names = ['not fraud', 'fraud']

    for name, model in models:
        kfold = model_selection.KFold(n_splits=5,shuffle=True,random_state=69)
        cv_results = model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring)
        clf = model.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(name)
        print(classification_report(y_test,y_pred,target_names=target_names))

        results.append(cv_results)
        names.append(name)

        this_df = pd.DataFrame(cv_results)
        this_df['model'] = name
        dfs.append(this_df)
    final = pd.concat(dfs, ignore_index=True)
    return final

model_train(X_train, y_train, X_test, y_test)

