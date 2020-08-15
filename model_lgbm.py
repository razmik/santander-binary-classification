"""
Author: Rashmika Nawaratne
Date: 15-Aug-20 at 12:45 PM

Bootstrapped from Senkin in Kaggle Notebook.
https://www.kaggle.com/senkin13/lstm-starter/code
"""
import pandas as pd
import numpy as np
import time
import pickle
import os
import gc
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score

from load_data import Data
from evaluation import Evaluator

# Algo config
num_folds = 7

# File config
VERSION = 4
MODEL_NAME = 'lgbm'
OUTPUT_FOLDER = 'model_outputs/{}_{}/'.format(MODEL_NAME, VERSION)
OUTPUT_FILENAME = OUTPUT_FOLDER + MODEL_NAME

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

if __name__ == "__main__":

    print('Loading Data...')
    df = Data().read_data()
    df_x = df.drop(['target'], axis=1)
    df_y = df[['target']]
    features = list(df_x.columns.values)

    X_train, X_test, Y_train, Y_test = train_test_split(df_x, df_y, shuffle=False, train_size=0.8)

    del df
    gc.collect()

    folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=47)
    oof = np.zeros(X_train.shape[0])
    getVal = np.zeros(X_train.shape[0])
    predictions = np.zeros(X_test.shape[0])
    feature_importance_df = pd.DataFrame()

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X_train, Y_train)):

        train_x, train_y = X_train.iloc[train_idx], Y_train.iloc[train_idx]
        valid_x, valid_y = X_train.iloc[valid_idx], Y_train.iloc[valid_idx]

        train_data = lgb.Dataset(train_x, label=train_y)
        valid_data = lgb.Dataset(valid_x, label=valid_y)

        param = {
            "objective": "binary",
            "metric": "auc",
            "boosting": 'gbdt',
            "max_depth": -1,
            "num_leaves": 31,
            "learning_rate": 0.01,
            "bagging_freq": 5,
            "bagging_fraction": 0.4,
            "feature_fraction": 0.05,
            "min_data_in_leaf": 150,
            "min_sum_heassian_in_leaf": 10,
            "tree_learner": "serial",
            "boost_from_average": "false",
            "bagging_seed": 42,
            "verbosity": 1,
            "seed": 42}

        clf = lgb.train(param, train_data, 1000000, valid_sets=[train_data, valid_data], verbose_eval=500,
                        early_stopping_rounds=200)
        oof[valid_idx] = clf.predict(X_train.iloc[valid_idx], num_iteration=clf.best_iteration)
        getVal[valid_idx] += clf.predict(X_train.iloc[valid_idx], num_iteration=clf.best_iteration) / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = features
        fold_importance_df["importance"] = clf.feature_importance()
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        predictions += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits

    print("CV score (Validation): {:<8.5f}".format(roc_auc_score(Y_train, oof)))
    print("CV score (Test): {:<8.5f}".format(roc_auc_score(Y_test, predictions)))

    y_pred = np.zeros(predictions.shape[0])
    y_pred[predictions >= 0.1] = 1

    eval = Evaluator()
    eval.evaluate(Y_test, y_pred)

    cols = (feature_importance_df[["feature", "importance"]]
            .groupby("feature")
            .mean()
            .sort_values(by="importance", ascending=False)[:1000].index)
    best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

    plt.figure(figsize=(14, 26))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (averaged over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances.png')
