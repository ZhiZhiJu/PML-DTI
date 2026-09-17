import json
import sys
import os
import torch
import logging
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

sys.path.append("../")

LOGGER = logging.getLogger(__name__)

class DatabaseProcessor:
    def __init__(self,args, data_dir="dataset/"):
        # Determine the dataset based on args.dataset_name
        if args.dataset == 'BindingDB':
            dataset_file = 'BindingDB.csv'
            self.name = "BindingDB"
        elif args.dataset == 'Human':
            dataset_file = 'Human.csv'
            self.name = "Human"
        elif args.dataset == 'Biosnap':
            dataset_file = 'Biosnap.csv'
            self.name = "Biosnap"
        elif args.dataset == 'Davis':
            dataset_file = 'davis.csv'
            self.name = "Davis"
        elif args.dataset == 'Drugbank':
            dataset_file = 'Drugbank.csv'
            self.name = "Drugbank"
        elif args.dataset == 'BindingDB_cluster':
            dataset_file = 'BindingDB_cluster.csv'
            self.name = "BindingDB_cluster"
        elif args.dataset == 'human':
            dataset_file = 'human.csv'
            self.name = "human"
        elif args.dataset == 'Biosnap_cluster':
            dataset_file = 'Biosnap_cluster.csv'
            self.name = "Biosnap_cluster"
        else:
            raise ValueError("Invalid dataset name provided. Please choose from 'BindingDB', 'Human', or 'Biosnap'.")
        
        
        # Setup dataset directory
        dataset_dir = f"{data_dir}{self.name}/"
        # self.dataset_df = pd.read_csv(dataset_dir + dataset_file)
        # shuffled_indices = np.random.permutation(len(self.dataset_df))
        # train_idx = shuffled_indices[:int(0.8 * len(shuffled_indices))]
        # val_idx = shuffled_indices[int(0.8 * len(shuffled_indices)):int(0.9 * len(shuffled_indices))]
        # test_idx = shuffled_indices[int(0.9 * len(shuffled_indices)):]
        #self.train_dataset_df =self.dataset_df.iloc[train_idx]
        #self.val_dataset_df = self.dataset_df.iloc[val_idx]
        #self.test_dataset_df = self.dataset_df.iloc[test_idx]
        
        # Load datasets
        if args.task == 'cluster':
            self.train_dataset_df = pd.read_csv(dataset_dir + "source_train_with_id.csv")
            self.val_dataset_df = pd.read_csv(dataset_dir + "target_test_with_id.csv")
            self.test_dataset_df = pd.read_csv(dataset_dir + "target_test_with_id.csv")
        else:
            self.train_dataset_df = pd.read_csv(dataset_dir + "train.csv")
            self.val_dataset_df = pd.read_csv(dataset_dir + "val.csv")
            self.test_dataset_df = pd.read_csv(dataset_dir + "test.csv")


        
        # Instantiate DTI_Dataset for training, validation, and testing
        self.train_data_loader = DTI_Dataset((
            self.train_dataset_df["SMILES"].values,
            self.train_dataset_df["uid"].values,
            # self.train_dataset_df["target_sequence_start"].values,
            # self.train_dataset_df["target_sequence_end"].values,
            # self.train_dataset_df["proteinSeq"].values,
            # self.train_dataset_df["selfies"].values,
            self.train_dataset_df["label"].values,
        ))
        self.val_data_loader = DTI_Dataset((
            self.val_dataset_df["SMILES"].values,
            self.val_dataset_df["uid"].values,
            # self.val_dataset_df["target_sequence_start"].values,
            # self.val_dataset_df["target_sequence_end"].values,
            # self.val_dataset_df["proteinSeq"].values,
            # self.val_dataset_df["selfies"].values,
            self.val_dataset_df["label"].values,
        ))
        self.test_data_loader = DTI_Dataset((
            self.test_dataset_df["SMILES"].values,
            self.test_dataset_df["uid"].values,
            # self.test_dataset_df["target_sequence_start"].values,
            # self.test_dataset_df["target_sequence_end"].values,
            # self.test_dataset_df["proteinSeq"].values,
            # self.test_dataset_df["selfies"].values,
            self.test_dataset_df["label"].values,
        ))

    def get_train_examples(self, test=False):
        if test == 1:  # Small testing set, to reduce the running time
            return (
                self.train_dataset_df["Seq"].values[:4096],
                self.train_dataset_df["selfies"].values[:4096],
                self.train_dataset_df["label"].values[:4096],
            )
        elif test > 1:
            return (
                self.train_dataset_df["Seq"].values[:test],
                self.train_dataset_df["selfies"].values[:test],
                self.train_dataset_df["label"].values[:test],
            )
        else:
            return self.train_data_loader
    
    def get_val_examples(self, test=False):
        if test == 1:
            return (
                self.val_dataset_df["Seq"].values[:1024],
                self.val_dataset_df["selfies"].values[:1024],
                self.val_dataset_df["label"].values[:1024],
            )
        elif test > 1:
            return (
                self.val_dataset_df["Seq"].values[:test],
                self.val_dataset_df["selfies"].values[:test],
                self.val_dataset_df["label"].values[:test],
            )
        else:
            return self.val_data_loader

    def get_test_examples(self, test=False):
        if test == 1:
            return (
                self.test_dataset_df["Seq"].values[:1024],
                self.test_dataset_df["selfies"].values[:1024],
                self.test_dataset_df["label"].values[:1024],
            )
        elif test > 1:
            return (
                self.test_dataset_df["Seq"].values[:test],
                self.test_dataset_df["selfies"].values[:test],
                self.test_dataset_df["label"].values[:test],
            )
        else:
            return self.test_data_loader

class DTI_Dataset(Dataset):
    """
    Candidate Dataset for:
        ALL drug-target interactions
    """

    def __init__(self, data_examples):
        self.smiles = data_examples[0]
        self.uniprot = data_examples[1]
        # self.start = data_examples[2]
        # self.end = data_examples[3]
        #
        # self.protein_seqs = data_examples[4]
        # self.drug_smil = data_examples[5]
        self.scores = data_examples[2]


    def __getitem__(self, query_idx):

        smile = self.smiles[query_idx]
        uniprot = self.uniprot[query_idx]
        # start = self.start[query_idx]
        # end = self.end[query_idx]
        #
        # protein_seq = self.protein_seqs[query_idx]
        # drug_smi = self.drug_smil[query_idx]
        score = self.scores[query_idx]

        # return smile, uniprot, start, end, protein_seq, drug_smi, score
        return smile, uniprot, score

    def __len__(self):
        return len(self.smiles)