import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier



""" Setting Display Options"""
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)


df = pd.read_csv("train.csv")


"""Dropping unnecessary columns"""
df = df.drop(["Name", "Ticket", "Fare", "Cabin", "PassengerId"], axis=1)


"""Adding Alone column, then dropping SibSp and Parch """
df["Relatives"] = df["SibSp"] + df["Parch"]
df.loc[df["Relatives"] > 0, "Alone"] = 0
df.loc[df["Relatives"] == 0, "Alone"] = 1

df["Alone"] = df["Alone"].round(0).astype(int)

df = df.drop(["SibSp", "Parch", "Relatives"], axis=1)


""" Check for NA values """
df.isnull().sum()


""" NA values of Age column will be filled with mean of their respective Sex value """
df.groupby("Sex").Age.mean()
df["Age"] = df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean"))
df["Age"] = df["Age"].round(0).astype(int)


"""Replacing Embarked values with the most frequency"""
df["Embarked"].value_counts()
df["Embarked"] = df["Embarked"].fillna("S")
df.isnull().sum()


""" Getting dummies for categorical columns"""
df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)*1


""" Slicing Dataframe for Train/Test Purposes """
y = df[["Survived"]]
X = df.drop(["Survived"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


"Random Forest Hyperparameter Tuning"
parameters = {
    'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
    'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [1, 2, 4],
    "min_samples_split": [2, 5, 10]
}

rndmforest = RandomForestClassifier()
rndmforest_gs = GridSearchCV(rndmforest, parameters, cv=3, n_jobs=-1, verbose=10)
rndmforest_gs.fit(X, y)
best_parameters = rndmforest_gs.best_params_


"""Tuned model and prediction"""
rndmforest_tuned = RandomForestClassifier(n_estimators=400, max_depth=70, max_features="sqrt", min_samples_split=5, min_samples_leaf=4)
rndmforest_tuned.fit(X, y)
predictions = rndmforest_tuned.predict(X_test)


""" Performance Metrics """
accuracy_rf = accuracy_score(y_test, predictions)
precision_rf = precision_score(y_test, predictions)
f1_rf = f1_score(y_test, predictions)
roc_auc_rf = roc_auc_score(y_test, predictions)

cv = cross_validate(estimator=rndmforest_tuned, X=X, y=y, cv=10, scoring="accuracy", return_train_score=True)
cv["test_score"].mean()


"""XGB Hyperparameter Tuning"""
parameters = {
    'n_estimators': [100, 200, 500, 750, 1000],
    'max_depth': [3, 5, 7, 9],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 1, 2, 3, 5],
    'subsample': [0.1, 0.4, 0.8, 1],
    'colsample_bytree': [0.1, 0.4, 0.8, 1],
    'reg_alpha': [0, 0.005, 0.01, 0.05, 0.1, 1],
    'learning_rate': [0.01, 0.02, 0.05, 0.1]
}

xgboost = XGBClassifier()
xgboost_gs = GridSearchCV(xgboost, parameters, cv=3, n_jobs=-1, verbose=10)
xgboost_gs.fit(X, y)
best_parameters_xgb = xgboost_gs.best_params_

"""Tuned model and prediction"""
xgboost_tuned = XGBClassifier(colsample_bytree=0.8, gamma=2, learning_rate=0.05, max_depth=5, min_child_weight=3, n_estimaors=750, reg_alpha=1, subsample=0.4)

xgb_tuned = xgboost_tuned.fit(X, y)
xgb_pred = xgb_tuned.predict(X_test)


""" Performance Metrics """
accuracy_xgb = accuracy_score(y_test, xgb_pred)
precision_xgb = precision_score(y_test, xgb_pred)
f1_xgb = f1_score(y_test, xgb_pred)
roc_auc_xgb = roc_auc_score(y_test, xgb_pred)

cv = cross_validate(estimator=xgboost_tuned, X=X, y=y, cv=10, scoring="accuracy", return_train_score=True)
cv["test_score"].mean()


"""Preparing Test DF(Same steps as above)"""
test = pd.read_csv("test.csv")
pid = pd.read_csv("gender_submission.csv")


test = test.drop(["Name", "Ticket", "Fare", "Cabin", "PassengerId"], axis=1)


"""Adding Alone column, then dropping SibSp and Parch """
test["Relatives"] = test["SibSp"] + test["Parch"]
test.loc[test["Relatives"] > 0, "Alone"] = 0
test.loc[test["Relatives"] == 0, "Alone"] = 1

test["Alone"] = test["Alone"].round(0).astype(int)

test = test.drop(["SibSp", "Parch", "Relatives"], axis=1)


""" NA values of Age column will be filled with mean of their respective Sex value """
test.groupby("Sex").Age.mean()
test["Age"] = test["Age"].fillna(test.groupby("Sex")["Age"].transform("mean"))
test["Age"] = test["Age"].round(0).astype(int)


"""Replacing Embarked values with the most frequency"""
test["Embarked"].value_counts()
test["Embarked"] = test["Embarked"].fillna("S")
df.isnull().sum()


""" Getting dummies for categorical columns"""
test = pd.get_dummies(test, columns=["Sex", "Embarked"], drop_first=True)*1


"""Final Prediction"""

y_final = pid[["Survived"]]
rndmforest_tuned.fit(test, y_final)
test["Survived"] = rndmforest_tuned.predict(test)
pid["Survived"] = test["Survived"]

"""Exporting Predictions"""
pid = pid.set_index("PassengerId")
pid.to_csv("gender_submission.csv")
