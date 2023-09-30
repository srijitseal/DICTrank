#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.model_selection import train_test_split

def stratified_data_split(df, activity_col, dataset_name, test_size=0.10, random_state=42):
    """
    Performs a stratified train/test split.
    
    If dataset_name is 'PK_Lombardo', values in activity_col are considered continuous and 
    the values will be binned to facilitate stratified splitting. Else, it's considered binary or categorical.
    
    Parameters:
    - df: DataFrame containing the data
    - activity_col: column name containing activity labels
    - dataset_name: name of the dataset (to handle 'PK_Lombardo' differently)
    - test_size: proportion of the data to be used as test set
    - random_state: random seed
    
    Returns:
    - train_df, test_df: train and test DataFrames after split
    """
    
    # drop NaN or is non-numeric
    df = df.dropna(subset=[activity_col])
        
    if dataset_name == "PK_Lombardo":
        # Convert continuous targets to bins for stratification
        bins = pd.qcut(df[activity_col], q=10, duplicates="drop")  # Quantile-based discretization
        stratify_col = bins
    else:
        stratify_col = df[activity_col]
    
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=stratify_col, random_state=random_state)
    
    return train_df, test_df