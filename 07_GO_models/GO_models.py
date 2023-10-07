#!/usr/bin/env python
# coding: utf-8


import os
import numpy as np
import pandas as pd
from tqdm import tqdm
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
from pandarallel import pandarallel
from sklearn.model_selection import StratifiedKFold

import math
import sys
sys.path.append('/home/ss2686/03_DICTrank')

import argparse
from scripts.evaluation_functions import evaluate_classifier, optimize_threshold_j_statistic

# Initialize pandarallel for parallel processing
pandarallel.initialize()
import gzip

data_path = '../data/processed_binarised__splits/'

csv_file_path = '../data/GeneOntology/GeneOntology_processed.csv.gz'


def create_molecule_dict(csv_file_path):
    molecule_dict = {}

    with gzip.open(csv_file_path, 'rt') as f:
        next(f)  # Skip the first line (header)
        for line in f:
            data = line.strip().split(',')
            smiles = data[0]
            features = np.array(data[1:4439], dtype=float)
            molecule_dict[smiles] = features
    
    return molecule_dict

# Assuming you call create_molecule_dict once to create the dictionary
molecule_dict = create_molecule_dict(csv_file_path)

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

def calculate_tanimoto_similarity(molecule1, molecule2):
    molecule1_fp =  AllChem.GetMorganFingerprintAsBitVect(molecule1, 2, nBits=2048)
    molecule2_fp =  AllChem.GetMorganFingerprintAsBitVect(molecule2, 2, nBits=2048)
    return DataStructs.TanimotoSimilarity(molecule1_fp, molecule2_fp)

def fetch_similar_profiles(smiles, exclude_smiles_set=None):
    if exclude_smiles_set is None:
        exclude_smiles_set = set()

    query_mol = Chem.MolFromSmiles(smiles)
    similar_profiles = []

    for key_smiles in molecule_dict:
        if key_smiles in exclude_smiles_set:
            continue

        key_mol = Chem.MolFromSmiles(key_smiles)
        similarity = calculate_tanimoto_similarity(query_mol, key_mol)
        if similarity > 0.7:
            similar_profiles.append(molecule_dict[key_smiles])

    if len(similar_profiles) > 0:
        return np.mean(similar_profiles, axis=0)
    else:
        return None

# Modify the generate_cellpainting function
# Modify the generate_cellpainting function
def generate_go(smiles, behave="Train", exclude_smiles_set=None):
    profile = molecule_dict.get(smiles)
    if profile is not None:
        return profile
    elif(behave=="Train"):
        return fetch_similar_profiles(smiles, exclude_smiles_set)
    else:
        return molecule_dict.get(smiles, None)
    

#Exammple usage:

#smiles_list = [
#    'CCc1nccn1-c1cccc(C2CCC[NH+]2C(=O)c2ccc(OCC[NH+](C)C)cc2)n1',
#    'O=C1NCCC[NH+]1Cc1ccc(Cl)cc1',
#    'O=C1NC(=O)c2cc(Nc3ccccc3)c(Nc3ccccc3)cc21',
#    'CCCn1nccc1S(=O)(=O)[NH+]1CC2CCC1C[NH2+]C2',
#    'CCNC(=O)CC1N=C(c2ccc(Cl)cc2)c2cc(OC)ccc2-n2c(C)nnc21'
#]

# Create a DataFrame with the SMILES
#smiles_df = pd.DataFrame({'SMILES': smiles_list})

#X_train = smiles_df['SMILES'].parallel_apply(generate_cellpainting)
#X_train = np.array(X_train.to_list())
#X_train

# Assuming image-based dataset is regression and others are classification
results = {}
held_out_results = []

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

        train_df = pd.read_csv(train_path, compression='gzip')#.sample(20)
        test_df = pd.read_csv(test_path, compression='gzip')#.sample(20)

        train_smiles_set = set(train_df['Standardized_SMILES'].tolist())
        test_smiles_set = set(test_df['Standardized_SMILES'].tolist())
        
        X_train = train_df['Standardized_SMILES'].parallel_apply(lambda x: generate_go(x, "Train", test_smiles_set))
        X_train = np.array(X_train.to_list())
        
        X_test = test_df['Standardized_SMILES'].parallel_apply(lambda x: generate_go(x, "Test"))
        X_test = np.array(X_test.to_list())
        
        y_train = train_df[activity]
        y_test = test_df[activity]

        failed_train_indices = [i for i, geneont in enumerate(X_train) if geneont is None]
        failed_test_indices = [i for i, geneont in enumerate(X_test) if geneont is None]
        
        # Drop those indices from X_train, X_test, y_train, and y_test
        X_train = np.delete(X_train, failed_train_indices, axis=0)
        X_train = np.vstack(X_train)
        y_train = y_train.drop(failed_train_indices).reset_index(drop=True)

        X_test = np.delete(X_test, failed_test_indices, axis=0)
        X_test = np.vstack(X_test)
        y_test = y_test.drop(failed_test_indices).reset_index(drop=True)

        # If you want to drop the rows from train_df and test_df as well
        train_df = train_df.drop(failed_train_indices).reset_index(drop=True)
        test_df = test_df.drop(failed_test_indices).reset_index(drop=True)
        
        print(train_df[activity].value_counts())
        print(test_df[activity].value_counts())
        
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
            
        # Random Over-sampling and Threshold Optimization
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
        #break

        #Save results at each step
        pd.DataFrame(results).T.to_csv('./GO_model_results.csv')
            
        

# Save results
results_df = pd.DataFrame(results).T.reset_index(drop=False)
results_df = results_df.rename(columns={'index': 'endpoint'})
results_df.to_csv('./GO_model_results.csv', index=False)

# Concatenate and save held-out test set results
pd.concat(held_out_results).to_csv('./GO_model_held_out_test_results.csv', index=False)



