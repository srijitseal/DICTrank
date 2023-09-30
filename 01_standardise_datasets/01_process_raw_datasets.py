import sys
sys.path.append('/home/ss2686/03_DICTrank')

import argparse
from scripts.standardise_smiles import process_data, save_data

def main(raw_path, smiles_variable, save_path):
    processed_data = process_data(raw_path, smiles_variable)
    save_data(processed_data, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Standardize SMILES in datasets")
    parser.add_argument('--raw_path', required=True, help="Path to the raw dataset")
    parser.add_argument('--smiles_variable', required=True, help="the name of the column where smiles are stored")
    parser.add_argument('--save_path', required=True, help="Path where the processed dataset should be saved")

    args = parser.parse_args()
    main(args.raw_path, args.smiles_variable, args.save_path)