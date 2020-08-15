﻿# Santander Customer Transaction Prediction
 
This repository contains the codebase to reproduce the Customer Transaction Prediction challenges on Kaggle.

### Required packages.

* python 3.X (Tested with 3.7)
* pandas
* numpy
* scipy
* scikit-learn
* lightgbm
* tqdm
* matplotlib
* tensorflow 2.x

### Codebase for analytics lifecycle

1. Data preparation.  
Basic pre-processing steps can be found from the first half of the Jupyter notebook at `favorita/1_EDA_Cleaning.ipynb`.  

3. Data Exploration.  
Exploratory data analytics (EDA) is detailed in the second half of the same above notebook - `favorita/1_EDA_Cleaning.ipynb`.  

4. Modeling prototypes.  
Prior to the model development, a prototyping was conducted for LGBM and DNN using Google Colab. Notebooks are available at `favorita/2_Modeling_LGBM_Log_Scaled_Prototype.ipynb` and `favorita/3_Modeling_NN_Log_Scaled_Prototype.ipynb`. Base code for LGBM and XGBoost are available at `favorita/base_lgb_model.py` and `favorita/base_xgb_model.py`.

5. Utility scripts.   
Script to load data: `favorita/load_data.py`  
Script to engineer features: `favorita/feature_extractor.py`  
Script to evaluate: `favorita/evaluation.py`  
!Important: Please create a config.py file in your environment indicating the root folder for the dataset.  

6. Hyper-parameter search.  
Random Search and Grid Search scripts for LGBM can be found at `favorita/base_lgb_model_random_search.py`.  

7. Predictive models (general model for all stores).  
Script for LGBM: `favorita/model_lgbm.py`  
Script for DNN: `favorita/model_nn.py`   

8. Predictive models (per store model).  
Script for LGBM: `favorita/model_lgbm_per_store.py`  
Script for DNN: `favorita/model_nn_per_store.py`   

9. Ensemble.  
Prototype ensemble is avaialble here: `favorita/4_Modeling_LGBM_Ensemble.ipynb`

