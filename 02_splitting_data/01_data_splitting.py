#!/usr/bin/env python
# coding: utf-8

import os
import sys
sys.path.append('/home/ss2686/03_DICTrank')

from scripts.stratified_split_helper import stratified_data_split

import pandas as pd
import pickle

# Loading the dictionary

pickle_file_path = '../01_standardise_datasets/activity_columns_mapping_selected.pkl'

# Loading the dictionary
with open(pickle_file_path, 'rb') as file:
    activity_columns_mapping = pickle.load(file)
    
def process_datasets(directory='../data/binarised/'):
    datasets = {}
    splits = {}
    dict_rank_test_smiles = []  # New variable to store DICTrank test_SMILES
    
    # Load datasets from given directory
    for foldername in os.listdir(directory):
        
        if not foldername.startswith('.'):  # Ignore folders starting with a dot
            
            #print(foldername)
            file_path = os.path.join(directory, foldername, f"{foldername}_binarised.csv.gz")

            if os.path.exists(file_path):
                datasets[foldername] = pd.read_csv(file_path, compression='gzip')
            else:
                print(f"No matching file found for folder: {foldername}")
                
            

    # First loop: Get the SMILES for test sets
    for name, df in datasets.items():
      
        if name == "DICTrank":
            activity_col = activity_columns_mapping.get(name, [])[0]  # Assuming only one activity col for DICTrank
            _, test_df = stratified_data_split(df, activity_col=activity_col, dataset_name=name)
            dict_rank_test_smiles.extend(test_df['Standardized_SMILES'].tolist())

    # Second loop: Stratified split for each dataset and each activity column
    for name, df in datasets.items():
        activity_cols = activity_columns_mapping.get(name, [])
        dataset_splits = {}
        
        for col in activity_cols:
            if name in ["cardiotox_with_sider_inactives", "cardiotox_with_sider_actives", 
                         "cardiotox_with_sider_all"] and dict_rank_test_smiles:
                
                print("Using DICTrank splits for this dataset ", name)
                test_df = df[df['Standardized_SMILES'].isin(dict_rank_test_smiles)]
                train_df = df[~df['Standardized_SMILES'].isin(dict_rank_test_smiles)]
                
            else:
                train_df, test_df = stratified_data_split(df, activity_col=col, dataset_name=name)
                
            # Filter the columns for saving
            cols_to_keep = ['Standardized_SMILES', 'Standardized_InChI', col]
            train_df = train_df[cols_to_keep]
            test_df = test_df[cols_to_keep]
            
            dataset_splits[col] = {'train': train_df, 'test': test_df}

        splits[name] = dataset_splits
    
    # Save the splits to new directories
    output_dir = "../data/processed_binarised__splits/"
    # Ensure the directory exists, if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through each dataset name, activities, and their corresponding splits
    for dataset_name, activities in splits.items():
        print(dataset_name)
        for activity, split_data in activities.items():
            for data_type, split_df in split_data.items():
                # Create a path for the dataset if it doesn't exist
                dataset_path = os.path.join(output_dir, dataset_name)
                if not os.path.exists(dataset_path):
                    os.makedirs(dataset_path)

                # Define the full output path for the split dataframe
                output_path = os.path.join(dataset_path, f"{activity}_{data_type}.csv.gz")

                # Save the dataframe
                split_df.to_csv(output_path, index=False, compression='gzip')

if __name__ == '__main__':
    process_datasets()