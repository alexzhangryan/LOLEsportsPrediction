# %%
from cycler import V
from matplotlib import pyplot as plt
import pandas as pd
import textwrap
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.metrics import classification_report
import xgboost as xgb
import create_prediction_df

training_data = pd.read_csv("predict_train.csv")
training_data = training_data.drop(columns="Unnamed: 0")


#%%

encoded_data = pd.get_dummies(training_data, dtype=int)
#print(encoded_data.to_string())
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

X = encoded_data.iloc[:, 1:]
y = encoded_data.iloc[:, 0]

#print(training_data.to_string())
#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=17, test_size=0.2)
rf = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=2)
rf.fit(X_train, y_train, eval_set=[(X_test, y_test)])

y_pred = rf.predict(X_test)
#print(len(y))
#print(len(y_pred))
rf.score(X_test, y_test)


print(classification_report(y_test, y_pred))
feature_importance = pd.DataFrame({
    "importance": rf.feature_importances_, 
    "feature": X.columns
})

#second retrain

useful_features = feature_importance[feature_importance["importance"] > 0]["feature"]

X_reduced = X[useful_features]

split_index = int(len(X_reduced) * 0.8)

X_train, X_test = X_reduced.iloc[:split_index], X_reduced.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]


rf = xgb.XGBClassifier(
    tree_methods="hist",
    n_estimators = 1000,
    learning_rate = 0.1,
    max_depth= 3,
    min_child_weight = 7,
    gamma = 0.1,
    subsample = 0.8,
    colsample_bytree = 0.8,
    eta = 0.1,
    early_stopping_rounds = 10,
    eval_metrics = "logloss"
)

rf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

y_pred = rf.predict(X_test)

print(classification_report(y_test, y_pred))

match_row = create_prediction_df.build_data("FlyQuest", "T1")
#print(training_data.head(1).to_string())
#print(match_row.to_string())

encoded_data = pd.get_dummies(match_row, dtype=int)
X = encoded_data
X_predict = X[useful_features]

prediction = rf.predict(X_predict)
probability = rf.predict_proba(X_predict)

print("Fly Win:", prediction[0])
print("Confidence scores:", probability[0]) 


#print(sorted_features.tail(50).to_string(), len(sorted_features), len(feature_importance))

#%%
