#!/usr/bin/env python
# coding: utf-8


import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors
calc = Calculator(descriptors, ignore_3D=True)

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score, matthews_corrcoef, average_precision_score, confusion_matrix
from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.metrics import roc_curve
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from scipy.stats import randint, uniform
from sklearn.model_selection import StratifiedKFold
import math

from pandarallel import pandarallel
pandarallel.initialize()

import sys
sys.path.append('/home/ss2686/03_DICTrank')
import argparse
from scripts.evaluation_functions import evaluate_classifier, optimize_threshold_j_statistic


# Path where your data is stored
data_path = '../data/processed_binarised__splits/'

results = {}
held_out_results = []

def get_Mordred_columns_to_use():
    
        datasets = {}
        directory='../data/processed/'
        # Load datasets from given directory
        for foldername in os.listdir(directory):

            if not foldername.startswith('.'):  # Ignore folders starting with a dot

                print(foldername)
                file_path = os.path.join(directory, foldername, f"{foldername}_processed.csv.gz")

                if os.path.exists(file_path):
                    datasets[foldername] = pd.read_csv(file_path, compression='gzip')
                else:
                    print(f"No matching file found for folder: {foldername}")
        
        smiles_list = []
        
        for featuresets in ["sider", "DICTrank"]: 
    
            smiles_list.extend(datasets[featuresets].Standardized_SMILES.to_list())
            print(len(smiles_list))
        smiles_list = list(set(smiles_list))
        
        print(len(smiles_list))
        data = pd.DataFrame(smiles_list, columns=["Standardized_SMILES"])
        
        Ser_Mol_train = data['Standardized_SMILES'].apply(Chem.MolFromSmiles)
        Mordred_table_data = calc.pandas(Ser_Mol_train)
        Mordred_table_data = Mordred_table_data.astype('float')
        Mordred_table_data = Mordred_table_data.dropna(axis='columns')
        data_columns = Mordred_table_data.columns
        
        return(data_columns)
    
data_columns = get_Mordred_columns_to_use()


# Assuming PK dataset is regression and others are classification
for dataset in os.listdir(data_path):
    
    # Exclude hidden files or directories like .ipynb_checkpoints
    if dataset.startswith('.'):
        continue
    print(dataset)

    # Get all the file names for this dataset
    all_files = os.listdir(os.path.join(data_path, dataset))

    # Extract activity names by removing the _train.csv.gz or _test.csv.gz from file names
    activity_names = list(set([f.replace("_train.csv.gz", "").replace("_test.csv.gz", "")  for f in all_files if not f.startswith(".ipynb_checkpoints")]))

    for activity in tqdm(activity_names, desc="Processing activities"):
        
        train_path = os.path.join(data_path, dataset, f"{activity}_train.csv.gz")
        test_path = os.path.join(data_path, dataset, f"{activity}_test.csv.gz")

        train_df = pd.read_csv(train_path, compression='gzip')
        test_df = pd.read_csv(test_path, compression='gzip')
        
        Ser_Mol_train = train_df['Standardized_SMILES'].apply(Chem.MolFromSmiles)
        Mordred_table_train = calc.pandas(Ser_Mol_train)
        Mordred_table_train = Mordred_table_train.astype('float')
               
        Ser_Mol_test = test_df['Standardized_SMILES'].apply(Chem.MolFromSmiles)
        Mordred_table_test = calc.pandas(Ser_Mol_test)
        Mordred_table_test = Mordred_table_test.astype('float')

        # Retain only those columns in the test dataset
        Mordred_table_train = Mordred_table_train[data_columns]
        Mordred_table_test = Mordred_table_test[data_columns]

        X_train = np.array(Mordred_table_train)
        X_test = np.array(Mordred_table_test)
        y_train = train_df[activity]
        y_test = test_df[activity]

      
        # Classification
        model = RandomForestClassifier(n_jobs=40)
            
        # Hyperparameter Optimization
        param_dist_classification = {'max_depth': randint(10, 20),
                          'max_features': randint(40, 50),
                          'min_samples_leaf': randint(5, 15),
                          'min_samples_split': randint(5, 15),
                          'n_estimators':[200, 300, 400, 500, 600],
                          'bootstrap': [True, False],
                          'oob_score': [False],
                          'random_state': [42],
                          'criterion': ['gini', 'entropy'],
                          'n_jobs': [40],
                          'class_weight' : [None, 'balanced']
                         }
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)   
            
        classification_search = HalvingRandomSearchCV(
                model,
                param_dist_classification,
                factor=3,
                cv=inner_cv,
                random_state=42,
                verbose=1,
                n_jobs=40)
            
        classification_search.fit(X_train, y_train)
        best_model = classification_search.best_estimator_
            
        # Random Over-sampling 
        sampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
            
        pipeline = Pipeline(steps=[('sampler', sampler), ('model', best_model)])
        pipeline.fit(X_train, y_train)
            
        # Predict using threshold-optimized model
        probs_test = pipeline.predict_proba(X_test)[:, 1]
            
        oof_predictions = np.zeros(X_train.shape[0])
        oof_probs = np.zeros(X_train.shape[0])

        cv_scores = []

        for train_idx, valid_idx in inner_cv.split(X_train, y_train):
            X_train_fold, y_train_fold = X_train[train_idx], y_train[train_idx]
            X_valid_fold, y_valid_fold = X_train[valid_idx], y_train[valid_idx]

            # Random Over-sampling
            X_resampled, y_resampled = sampler.fit_resample(X_train_fold, y_train_fold)

            # Train the model on the resampled data
            best_model.fit(X_resampled, y_resampled)

            # Store out-of-fold predictions
            oof_predictions[valid_idx] = best_model.predict(X_valid_fold)
            oof_probs[valid_idx] = best_model.predict_proba(X_valid_fold)[:, 1]

            # AUC for this fold
            fold_auc = roc_auc_score(y_valid_fold, oof_probs[valid_idx])
            cv_scores.append(fold_auc)

        # Optimize the threshold using out-of-fold predictions
        best_threshold = optimize_threshold_j_statistic(y_train, oof_probs)
        predictions_test = (probs_test >= best_threshold).astype(int)

        results[activity] = {
                'CV_AUC_mean': np.mean(cv_scores),
                'CV_AUC_std': np.std(cv_scores),
                **evaluate_classifier(y_test, predictions_test, probs_test)
            }
        
        held_out_data = {
            'Dataset': dataset,
            "Actviity": activity,
            'SMILES': test_df['Standardized_SMILES'],
            'True_Value': y_test,
            'Prediction': predictions_test,
            'Probability': probs_test,
            'Best_Threshold': best_threshold
        }
        
        held_out_results.append(pd.DataFrame(held_out_data))         
        # Save results at each step
            
            
    # Save results at each step
    pd.DataFrame(results).T.to_csv('./physicochemical_model_results.csv')
              

# Save results
results_df = pd.DataFrame(results).T.reset_index(drop=False)
results_df = results_df.rename(columns={'index': 'endpoint'})
results_df.to_csv('./physicochemical_model_results.csv', index=False)

# Concatenate and save held-out test set results
pd.concat(held_out_results).to_csv('./physicochemical_model_held_out_test_results.csv', index=False)
            
