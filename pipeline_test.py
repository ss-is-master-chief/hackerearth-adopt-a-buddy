import pandas as pd
import numpy as np
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

pd.options.display.max_columns = None


def pre_processing(df_train, df_test):

    remove_color_type = list(
        set(df_train["color_type"]).difference(set(df_test["color_type"]))
    )
    df_train = df_train[~df_train["color_type"].isin(remove_color_type)]

    remove_X1 = list(set(df_train["X1"]).difference(set(df_test["X1"])))
    df_train = df_train[~df_train["X1"].isin(remove_X1)]

    color_type_le = LabelEncoder()
    color_type_le.fit(df_train["color_type"])
    df_train["color_type"] = color_type_le.transform(df_train["color_type"])
    df_test["color_type"] = color_type_le.transform(df_test["color_type"])

    # condition_median = df_train[(df_train["breed_category"]== 2.0) & (~df_train["condition"].isnull())]["condition"].median()
    # print(condition_median)
    df_train["condition"] = df_train["condition"].fillna(-1)
    df_test["condition"] = df_test["condition"].fillna(-1)

    df_train["condition"] = df_train["condition"].astype(int)
    df_test["condition"] = df_test["condition"].astype(int)

    train_size = df_train.shape[0]

    # df_traincombined = df_train.append(df_test)
    df_train = pd.get_dummies(df_train, columns=["condition", "color_type"])
    df_test = pd.get_dummies(df_test, columns=["condition", "color_type"])
    
    # df_train = combined[:train_size]
    # df_test = combined[train_size:].drop(["pet_category", "breed_category"], axis=1)


    return df_train, df_test


def add_new_features(df):
    df["issue_date"] = pd.to_datetime(df["issue_date"])
    df["listing_date"] = pd.to_datetime(df["listing_date"])

    df["date_diff_months"] = df["listing_date"] - df["issue_date"]
    df["date_diff_months"] = df["date_diff_months"] / np.timedelta64(1, "M")
    df["date_diff_months"] = df["date_diff_months"].astype(int)

    df["date_diff_years"] = df["listing_date"] - df["issue_date"]
    df["date_diff_years"] = df["date_diff_years"] / np.timedelta64(1, "Y")
    df["date_diff_years"] = df["date_diff_years"].astype(int)

    df["listing_year"] = df["listing_date"].dt.to_period("Y").astype(int)
    df["listing_month"] = df["listing_date"].dt.to_period("M").astype(int)

    df["issue_year"] = df["issue_date"].dt.to_period("Y").astype(int)
    df["issue_month"] = df["issue_date"].dt.to_period("M").astype(int)

    df["issue_quarter"] = df["issue_date"].dt.quarter
    df["listing_quarter"] = df["listing_date"].dt.quarter

    df = pd.get_dummies(df, columns=["listing_year"])

    return df


def update_predictors(df_train, predictor_labels):
    predictor_labels = predictor_labels + [
        "date_diff_months",
        "date_diff_years",
        "listing_year",
        "issue_year",
        "listing_month",
        "issue_month",
        "issue_quarter",
        "listing_quarter",
    ]

    return predictor_labels


def train_model(model, X_train, y_train, params):
    model = model(params)
    model.fit(X_train, y_train, verbose=1)

    return model


def save_model(model, file_name):
    pickle.dump(model, open(file_name, "wb"))


if __name__ == "__main__":

    data_folder = "./data"

    train = pd.read_csv(os.path.join(data_folder, "train.csv"))
    test = pd.read_csv(os.path.join(data_folder, "test.csv"))

    submission_columns = ["pet_id", "breed_category", "pet_category"]

    remove_labels = [
        "issue_date",
        "pet_id",
        "listing_date",
        # "length(m)",
        # "height(cm)",
        "condition",
        "color_type",
        "listing_year",
        "issue_year"
    ]
    target_labels = ["pet_category", "breed_category"]

    train, test = pre_processing(train, test)
    train, test = add_new_features(train), add_new_features(test)

    predictor_labels = train.columns.difference(target_labels + remove_labels).tolist()
    color_type_cols = [col for col in train.columns if col.startswith("color_type")]
    condition_cols = [col for col in train.columns if col.startswith("condition")]
    listing_year_cols = [col for col in train.columns if col.startswith("listing_year")]
    # X1_cols = [col for col in train.columns if col.startswith("X1")]
    # X2_cols = [col for col in train.columns if col.startswith("X2")]
    # issue_year_cols = [col for col in train.columns if col.startswith("issue_year")]

    # print(issue_year_cols)

    # predictor_labels = predictor_labels + color_type_cols + condition_cols

    print(predictor_labels)

    params_pet_category = {
        "colsample_bytree": 0.65,
        "eta": 0.025,
        "gamma": 0.55,
        "max_depth": 10,
        "min_child_weight": 3.0,
        "n_estimators": 254.0,
        "subsample": 0.75,
        "objective": "multi:softmax",
        "n_jobs": -1,
        "random_state": 47,
    }

    params_breed_category = {
        "booster": "gbtree",
        "colsample_bytree": 0.7000000000000001,
        "eta": 0.14,
        "eval_metric": "mlogloss",
        "gamma": 0.4,
        "max_depth": 13,
        "min_child_weight": 6.0,
        "n_estimators": 221.0,
        "nthread": 4,
        "num_class": 3,
        "objective": "multi:softmax",
        "seed": 47,
        "silent": 1,
        "subsample": 0.5,
        "tree_method": "exact",
        "n_jobs": -1,
        "random_state": 47,
    }

    # pet_model = train_model(
    #     XGBClassifier,
    #     train[predictor_labels],
    #     train["pet_category"].astype(int),
    #     params_pet_category,
    # )

    ########### STACKING ###########
    # lgbm = LGBMClassifier(objective="multiclassova", num_class=4, random_state=47)
    # xgb = XGBClassifier(params_pet_category)
    pet_model = CatBoostClassifier(
        cat_features=[
            "issue_quarter",
            "issue_month",
            "listing_quarter",
            "listing_month",
            "X1",
            "X2",
        ]
        + color_type_cols
        + condition_cols
        + listing_year_cols,
        random_seed=47,
    )
    # estimators = [
    #     ('lgbm', lgbm),
    #     ('xgb', xgb),
    #     ('cat', cat)
    # ]

    # pet_model = StackingClassifier(
    # estimators=estimators, final_estimator=LogisticRegression(random_state=47))
    pet_model.fit(train[predictor_labels], train["pet_category"].astype(int))

    train["pet_prediction"] = pet_model.predict(train[predictor_labels])
    test["pet_prediction"] = pet_model.predict(test[predictor_labels])

    # lgbm = LGBMClassifier(random_state=47)
    # xgb = XGBClassifier(params_breed_category)
    # cat = CatBoostClassifier(cat_features=[
    #     "pet_prediction"
    #     'condition_-1',
    #     'condition_0',
    #     'condition_1',
    #     'condition_2',
    #     "color_type",
    #     "issue_quarter",
    #     "issue_month",
    #     "listing_quarter",
    #     "listing_month",
    #     "X1", "X2"], random_seed=47)
    # estimators = [
    #     ('lgbm', lgbm),
    #     ('xgb', xgb),
    #     # ('cat', cat)
    # ]

    # breed_model = StackingClassifier(
    # estimators=estimators, final_estimator=LogisticRegression(random_state=47))
    # breed_model.fit(train[predictor_labels + ["pet_prediction"]], train["breed_category"].astype(int))

    # print(classification_report(y_test["pet_category"].astype(int), clf.predict(X_test)))

    # print(predictor_labels)
    # pet_model = LGBMClassifier(random_state=47)
    # pet_model.fit(train[predictor_labels],
    # train["pet_category"].astype(int),
    # categorical_feature=[2,3,6,7,10,11])

    # y_pred_pet = breed_model.predict(X_test)
    # print("Breed Category Results")
    # print(classification_report(y_test["breed_category"], y_pred_breed))

    # breed_model = train_model(
    #     XGBClassifier,
    #     train[predictor_labels],
    #     train["breed_category"].astype(int),
    #     params_breed_category,
    # )
    # y_pred_breed = breed_model.predict(X_test)
    # print("Breed Category Results")
    # print(classification_report(y_test["breed_category"], y_pred_breed))

    #  = XGBClassifier(params_breed_category)
    breed_model = CatBoostClassifier(
        cat_features=[
            "issue_quarter",
            "issue_month",
            "listing_quarter",
            "listing_month",
            "pet_prediction",
            # "X1", 
            # "X2"
        ]
        + color_type_cols
        + condition_cols
        + listing_year_cols,
        random_state=47,
    )
    breed_model.fit(
        train[predictor_labels + ["pet_prediction"]],
        train["breed_category"].astype(int),
    )

    test["pet_category"] = test["pet_prediction"]
    # test["pet_category"] = pet_model.predict(test[predictor_labels])
    # test["pet_prediction"] = test["pet_category"]
    test["breed_category"] = breed_model.predict(
        test[predictor_labels + ["pet_prediction"]]
    )

    # save_model(pet_model, "pet_model_v14.pkl")
    # save_model(breed_model, "breed_model_v14.pkl")
    test[submission_columns].to_csv("submission_15.csv", index=False)
