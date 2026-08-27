import pandas as pd
import argparse
from utils.prepare_drug import prepare_data

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="BindingDB_cluster", help="Dataset name under dataset/ (e.g. BindingDB_cluster, Biosnap, Davis)")
args = parser.parse_args()

smiles_path = f"./dataset/{args.dataset}/full.csv"
selfies_path = f"./dataset/{args.dataset}/SEFLIES.csv"

prepare_data(path=smiles_path, save_to=selfies_path)
chembl_df = pd.read_csv(selfies_path)
print("SELFIES representation file is ready.")