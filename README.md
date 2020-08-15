# Santander Customer Transaction Prediction
 
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

1. Data preparation and Exploration.  
Basic pre-processing steps can be found from the Jupyter notebook at `1_EDA_Cleaning.ipynb`.  

2. Modeling prototypes.  
Prior to the model development, a prototyping was conducted for LGBM and DNN using Google Colab. Notebooks are available at the latter part of `1_EDA_Cleaning.ipynb`.

3. Utility scripts.   
Script to load data: `load_data.py`  
Script to evaluate: `evaluation.py`  

4. Hyper-parameter search.  
Random Search and Grid Search scripts for LGBM can be found at `lgbm_model_random_search.py`.  

5. Predictive models.  
Script for LGBM: `2_LGBM.ipynb` and  `model_lgbm.py`.
Script for LGBM with Up-Sampled data: `3_LGBM_Upsampled.ipynb`.
Script for DNN: `4_DNN.ipynb` 


