#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

# Define the path to your gzip-compressed image_features.csv.gz file
csv_file_path = '../data/CellPainting/CellPainting_processed.csv.gz'


def create_molecule_dict(csv_file_path):
    molecule_dict = {}

    with gzip.open(csv_file_path, 'rt') as f:
        next(f)  # Skip the first line (header)
        for line in f:
            data = line.strip().split(',')
            smiles = data[0]
            features = np.array(data[1:1784], dtype=float)
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

# Modify the generate_cellpainting function
def generate_cellpainting(smiles, behave="Train", exclude_smiles_set=None):
    
    profile = molecule_dict.get(smiles)
    if profile is not None:
        return profile
    else:
        return molecule_dict.get(smiles, None)
    
    
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
        
        X_train = train_df['Standardized_SMILES'].parallel_apply(lambda x: generate_cellpainting(x, "Train", test_smiles_set))
        X_train = np.array(X_train.to_list())
        
        X_test = test_df['Standardized_SMILES'].parallel_apply(lambda x: generate_cellpainting(x, "Test"))
        X_test = np.array(X_test.to_list())
        
        y_train = train_df[activity]
        y_test = test_df[activity]

        failed_train_indices = [i for i, cellpaint in enumerate(X_train) if cellpaint is None]
        failed_test_indices = [i for i, cellpaint in enumerate(X_test) if cellpaint is None]
        
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
        


# In[2]:


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

# Define the path to your gzip-compressed image_features.csv.gz file
csv_file_path = '../data/LINCSL1000/LINCSL1000_processed.csv.gz'


def create_molecule_dict(csv_file_path):
    molecule_dict = {}

    with gzip.open(csv_file_path, 'rt') as f:
        next(f)  # Skip the first line (header)
        for line in f:
            data = line.strip().split(',')
            smiles = data[0]
            features = np.array(data[1:979], dtype=float)
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

# Modify the generate_cellpainting function
def generate_lincs(smiles, behave="Train", exclude_smiles_set=None):
    profile = molecule_dict.get(smiles)
    if profile is not None:
        return profile
    else:
        return molecule_dict.get(smiles, None)
    
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
        
        X_train = train_df['Standardized_SMILES'].parallel_apply(lambda x: generate_lincs(x, "Train", test_smiles_set))
        X_train = np.array(X_train.to_list())
        
        X_test = test_df['Standardized_SMILES'].parallel_apply(lambda x: generate_lincs(x, "Test"))
        X_test = np.array(X_test.to_list())
        
        y_train = train_df[activity]
        y_test = test_df[activity]

        failed_train_indices = [i for i, lincs in enumerate(X_train) if lincs is None]
        failed_test_indices = [i for i, lincs in enumerate(X_test) if lincs is None]
        
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
        


# In[3]:


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

# Modify the generate_cellpainting function
# Modify the generate_cellpainting function
def generate_go(smiles, behave="Train", exclude_smiles_set=None):
    profile = molecule_dict.get(smiles)
    if profile is not None:
        return profile
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
        


# In[4]:


#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
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


import sys
sys.path.append('/home/ss2686/03_DICTrank')
import argparse
from scripts.evaluation_functions import evaluate_classifier, optimize_threshold_j_statistic


# Path where your data is stored
data_path = '../data/processed_binarised__splits/'

results = {}
held_out_results = []

def generate_fingerprints(smiles_list):
    fps = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
    return np.array(fps)


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

        print(train_df[activity].value_counts())
        print(test_df[activity].value_counts())

      
        


# In[ ]:





# In[ ]:





# In[ ]:




