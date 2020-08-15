"""
Author: Rashmika Nawaratne
Date: 05-Aug-20 at 4:53 PM

Categorical features in LGBM:
https://www.kaggle.com/c/home-credit-default-risk/discussion/58950
"""

from datetime import datetime
import pickle
import time
import gc

import lightgbm as lgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV

from load_data import Data

MODEL_NAME = 'base_lgb_random_search'
OUTPUT_FOLDER = 'model_outputs/' + MODEL_NAME
SELECTED_STORES = [i for i in range(1, 11)]
ONLY_EVALUATE = False
TRAIN_TEST_SPLIT_DATE = datetime(2017, 7, 1)

if __name__ == "__main__":

    print('Loading Data...')
    df = Data().read_data()
    df_x = df.drop(['target'], axis=1)
    df_y = df[['target']]
    features = list(df_x.columns.values)

    X_train, X_test, Y_train, Y_test = train_test_split(df_x, df_y, shuffle=False, train_size=0.8)

    del df
    gc.collect()

    print('Training dataset: {}'.format(X_train.shape))
    print('Testing dataset: {}'.format(X_test.shape))

    if not ONLY_EVALUATE:

        # converting datasets into lgb format,
        # list of names of categorical variable has been provided to conduct One-hot encoding
        lgb_train = lgb.Dataset(data=X_train, label=Y_train)
        lgb_test = lgb.Dataset(data=X_test, label=Y_test, reference=lgb_train)

        # Parameters
        grid_params = {'learning_rate': [0.001, 0.01, 0.1, 0.2, 0, 3],
                       'num_leaves': [50, 60, 80, 100],
                       'metric': ['rmse'],
                       'boosting_type': ['gbdt'],
                       'bagging_fraction': [0.8, 0.85, 0.9],
                       'max_depth': [6, 8, 10, 12],
                       'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                       'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                       'objective': ['regression'],
                       'min_data_per_leaf': [80, 100, 120, 150, 180, 210, 250, 300],
                       'min_split_gain': [0.01, 0.05, 0.1]}

        # training the model using 100 iterations with early stopping if validation RMSE decreases
        eval_result = {}
        start_time = time.time()
        lgb_random = RandomizedSearchCV(estimator=lgb.LGBMRegressor(), param_distributions=grid_params, n_iter=10, cv=5, verbose=2,
                                    random_state=8, n_jobs=-1)
        lgb_random.fit(X_train, Y_train)
        end_time = time.time()
        print("Model Grid search time: {} mins.".format((end_time - start_time) / 60))

        # Save model
        with open(OUTPUT_FOLDER + '.pickle', 'wb') as file:
            pickle.dump(lgb_random, file)

    else:
        # Load from file
        with open(OUTPUT_FOLDER + '.pickle', 'rb') as file:
            lgb_random = pickle.load(file)

    # Evaluation

    best_random = lgb_random.best_estimator_
    best_random_params = lgb_random.best_params_

    print(best_random_params)