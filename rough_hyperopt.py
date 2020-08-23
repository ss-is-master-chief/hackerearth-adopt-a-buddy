import pandas as pd
import numpy as np
import os
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.metrics import classification_report, roc_auc_score, f1_score, log_loss

from xgboost import XGBClassifier
import xgboost as xgb
from catboost import Pool, CatBoostClassifier

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

import pickle

pd.options.display.max_columns = None

data_folder = "./data"

train = pd.read_csv(os.path.join(data_folder, "train.csv"))
# test = pd.read_csv(os.path.join(data_folder, "test.csv"))

remove_labels = ["issue_date", "pet_id", "listing_date","length(m)", 
"height(cm)"]
target_labels = ["pet_category", "breed_category"]
predictor_labels = train.columns.difference(target_labels).difference(remove_labels).tolist()

train["pet_category"] = train["pet_category"].replace({4:3})

# print(train["pet_category"].value_counts())

color_type_le = LabelEncoder()
color_type_le.fit(train["color_type"])
train["color_type"] = color_type_le.transform(train["color_type"])

train["issue_date"] = pd.to_datetime(train["issue_date"])
train["listing_date"] = pd.to_datetime(train["listing_date"])

train["date_diff_months"] = (train["listing_date"] - train["issue_date"])
train["date_diff_months"] = train["date_diff_months"]/np.timedelta64(1,'M')

train["listing_year"] = train["listing_date"].dt.to_period('Y').astype(int)
train["issue_year"] = train["issue_date"].dt.to_period('Y').astype(int)
train["listing_month"] = train["listing_date"].dt.to_period('M').astype(int)
train["issue_month"] = train["issue_date"].dt.to_period('M').astype(int)

train["issue_quarter"] = train["issue_date"].dt.quarter
train["listing_quarter"] = train["listing_date"].dt.quarter

predictor_labels = predictor_labels + ["date_diff_months", "listing_year", 
"issue_year", "listing_month", "issue_month", "issue_quarter", "listing_quarter"]

X_train, X_test, y_train, y_test = train_test_split(train[predictor_labels], 
                                                    train[target_labels],
                                                    test_size=0.2, 
                                                    stratify=train[target_labels], 
                                                    random_state=47)

# lb = LabelBinarizer()
# lb.fit(y_train["pet_category"])
# y_train = lb.transform(y_train["pet_category"])
# y_test = lb.transform(y_test["pet_category"])

# condition_median = X_train["condition"].median()
X_train["condition"] = X_train["condition"].fillna(-1)
X_test["condition"] = X_test["condition"].fillna(-1)
X_train["condition"] = X_train["condition"].astype(int)
X_test["condition"] = X_test["condition"].astype(int)



# xgb_breed = XGBClassifier(n_jobs=-1, random_state=47)
# xgb_breed.fit(X_train, y_train["breed_category"], verbose=1)
# y_pred_breed = xgb_breed.predict(X_test)
# print(classification_report(y_test["breed_category"], y_pred_breed))

# xgb_pet = XGBClassifier(n_jobs=-1, random_state=47)
# xgb_pet.fit(X_train, y_train["pet_category"], verbose=1)
# y_pred_pet = xgb_pet.predict(X_test)
# print(classification_report(y_test["pet_category"], y_pred_pet))

def optimize(space,
             trials, 
             random_state=47):
    """
    This is the optimization function that given a space (space here) of 
    hyperparameters and a scoring function (score here), finds the best hyperparameters.
    """
    # To learn more about XGBoost parameters, head to this page: 
    # https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
    # space = {
    #     'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),
    #     'eta': hp.quniform('eta', 0.001, 0.51, 0.01),
    #     # A problem with max_depth casted to float instead of int with
    #     # the hp.quniform method.
    #     'max_depth':  hp.choice('max_depth', np.arange(1, 14, dtype=int)),
    #     'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
    #     'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
    #     'gamma': hp.quniform('gamma', 0.2, 1, 0.05),
    #     'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
    #     'eval_metric': 'mlogloss',
    #     'objective': 'multi:softmax',
    #     'num_class': 4,
    #     # Increase this number if you have more cores. Otherwise, remove it and it will default 
    #     # to the maxium number. 
    #     'nthread': 4,
    #     'booster': 'gbtree',
    #     'tree_method': 'exact',
    #     'silent': 1,
    #     'seed': random_state
    # }

    # Use the fmin function from Hyperopt to find the best hyperparameters
    best = fmin(score, space, algo=tpe.suggest, 
                trials=trials, 
                max_evals=250)
    return best

def get_catboost_params(space):
    params = dict()
    params['learning_rate'] = space['learning_rate']
    params['depth'] = int(space['depth'])
    params['l2_leaf_reg'] = space['l2_leaf_reg']
    params['border_count'] = space['border_count']
    #params['rsm'] = space['rsm']
    return params

def score(space):
    # print("Training with params: ")
    print(params)
    params = get_catboost_params(space)
    # num_round = int(params['n_estimators'])
    # del params['n_estimators']
    # dtrain = xgb.DMatrix(X_train, label=y_train["breed_category"])
    # dvalid = xgb.DMatrix(X_test, label=y_test["breed_category"])
    # watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
    # gbm_model = xgb.train(params, dtrain, num_round,
    #                       evals=watchlist,
    #                       verbose_eval=True)
    dtrain = Pool(X_train, label=y_train)
    dvalid = Pool(X_test, label=y_test)
    model = CatBoostClassifier(iterations=100000,
                                        learning_rate=params['learning_rate'],
                                        depth=int(params['depth']),
                                        loss_function='CrossEntropy',
                                        use_best_model=True,
                                        task_type="CPU",
                                        eval_metric='AUC',
                                        classes_count=4,
                                        l2_leaf_reg=params['l2_leaf_reg'],
                                        early_stopping_rounds=3000,
                                        od_type="Iter",
                                        border_count=int(params['border_count']),
                                        verbose=False
                                        )
    model.fit(dtrain, eval_set=dvalid, verbose=False)
    # prinpredictions.shape)
    predictions = model.predict(dvalid.get_features())
    # print(Counter(y_test["pet_category"]), Counter(predictions))
    # lb = LabelBinarizer()
    # lb.fit(y_test["breed_category"])
    # y_true_bin = lb.transform(y_test["breed_category"])
    # y_pred_bin = lb.transform(predictions)

    score = f1_score(y_test["pet_category"], predictions, average="weighted")
    # TODO: Add the importance for the selected features
    print("\tScore {0}\n\n".format(score))
    # The score function should return the loss (1-score)
    # since the optimize function looks for the minimum
    loss = 1 - score
    return {'loss': loss, 'status': STATUS_OK}

space = {
    'depth': hp.quniform("depth", 1, 6, 1),
    'border_count': hp.uniform('border_count', 32, 255),
    'learning_rate': hp.quniform('eta', 0.001, 0.51, 0.01),
    'l2_leaf_reg': hp.uniform('l2_leaf_reg', 3, 8),
    }

trials = Trials()
best_hyperparams = optimize(
                            space=space,
                            trials=trials
                            )
print("The best hyperparameters are: ", "\n")
print(best_hyperparams)

# print(Counter(y_test["breed_category"]))

# params_pet_category = {'colsample_bytree': 0.65, 
# 'eta': 0.025, 
# 'gamma': 0.55, 
# 'max_depth': 10, 
# 'min_child_weight': 3.0, 
# 'n_estimators': 254.0, 
# 'subsample': 0.75, 
# 'n_jobs': -1, 
# 'random_state': 47}

# xgb_pet = XGBClassifier(params_pet_category)
# xgb_pet.fit(X_train, y_train["pet_category"], verbose=1)
# y_pred_pet = xgb_pet.predict(X_test)
# print(classification_report(y_test["pet_category"], y_pred_pet))


# params_breed_category = {
# 'n_jobs': -1, 
# 'random_state': 47}

# xgb_breed = XGBClassifier(params_breed_category)
# xgb_breed.fit(X_train, y_train["pet_category"], verbose=1)
# y_pred_breed = xgb_breed.predict(X_test)
# print(classification_report(y_test["breed_category"], y_pred_breed))

# pickle.dump(xgb_breed, open("xgb_breed_category.pkl", "w"))
# pickle.dump(xgb_pet, open("xgb_pet_category.pkl", "w"))