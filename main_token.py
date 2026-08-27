import argparse
import os
import random
import string
import sys
import pandas as pd
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.append("../")
import numpy as np
# import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm
import sklearn.metrics as metrics
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.metrics import precision_recall_curve, f1_score, precision_recall_fscore_support
from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score, recall_score, precision_score, \
            accuracy_score, roc_auc_score, precision_recall_curve, roc_curve
from transformers import EsmForMaskedLM, AutoModel, EsmTokenizer, AutoTokenizer
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from utils.process_datasets import DatabaseProcessor
from utils.metric_learning_models import BatchFileDataset, PMLDTI, set_seed
from sklearn.metrics import auc as auc_fuc
from scheduler import CosineAnnealingWarmupRestarts

import time
from tqdm import tqdm
import yaml

import logging
import pickle
import dgl

from utils.loss_function_vMF import get_loss


CONFIG_PATH = "./config/config.yaml"

CHARPROTSET = {
    "A": 1,
    "C": 2,
    "B": 3,
    "E": 4,
    "D": 5,
    "G": 6,
    "F": 7,
    "I": 8,
    "H": 9,
    "K": 10,
    "M": 11,
    "L": 12,
    "O": 13,
    "N": 14,
    "Q": 15,
    "P": 16,
    "S": 17,
    "R": 18,
    "U": 19,
    "T": 20,
    "W": 21,
    "V": 22,
    "Y": 23,
    "X": 24,
    "Z": 25,
}

CHARPROTLEN = 25




# from bertviz import head_view
# import lightgbm as lgb

def parse_config():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser(
        description="Train and test a model."
    )
    # parser = argparse.ArgumentParser()
    parser.add_argument('-f')
    parser.add_argument(
        "--input_feature_save_path",
        type=str,
        default="dataset/processed_DTI_Token",
        help="path of tokenized training data",
    )
    parser.add_argument(
        "--agg_mode", default="mean", type=str, help="{cls|mean|mean_all_tok}"
    )
    parser.add_argument(
        "--fusion", default="CAN", type=str, help="{CAN|BAN}")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--group_size", type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.2, help="warmup ratio of total epochs for cosine warmup")
    parser.add_argument("--device", type   =str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--seed', default=1, help="which seed to use", type=int)
    parser.add_argument('--epoch', default=100, help="which epoch to use", type=int)
    parser.add_argument('--patience', default=25, type=int, help="early stopping patience (epochs without improvement)")
    parser.add_argument(
        "--task", default="cluster", type=str, help="{cluster|random}")
    parser.add_argument(
        "--save_path_prefix",
        type=str,
        default="save_model_ckp/",
        help="save the result in which directory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="BindingDB",
        help="Name of the dataset to use (e.g., 'BindingDB', 'Human', 'Biosnap')"
    )

    parser.add_argument('--num_heads', type=int, default=1, help="attention heads for fusion")
    parser.add_argument("--beta", type=float, default=0.2)
    parser.set_defaults(**config)
    args = parser.parse_args()

    # 加载 per-(dataset, task) 的 yaml，覆盖 batch_size / num_heads / mm_output_sizes
    cfg_file = os.path.join(os.path.dirname(CONFIG_PATH), f"{args.dataset}_{args.task}.yaml")
    if os.path.exists(cfg_file):
        with open(cfg_file, "r") as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            if k == 'mm_output_sizes':
                args.mmmamba['mm_output_sizes'] = v
            else:
                setattr(args, k, v)
    return args


def integer_label_protein(sequence, max_length=1024):
    """
    Integer encoding for protein string sequence.
    Args:
        sequence (str): Protein string sequence.
        max_length: Maximum encoding length of input protein string.
    """
    encoding = np.zeros(max_length)
    for idx, letter in enumerate(sequence[:max_length]):
        try:
            letter = letter.upper()
            encoding[idx] = CHARPROTSET[letter]
        except KeyError:
            logging.warning(
                f"character {letter} does not exists in sequence category encoding, skip and treat as " f"padding."
            )
    return encoding

def get_feature(model, dataloader, args, set_type):
    # Create a subdirectory within input_feature_save_path
    if args.task == "cluster":
        subdirectory = os.path.join(args.input_feature_save_path, args.dataset + '_' + args.task)
    else:
        subdirectory = os.path.join(args.input_feature_save_path, args.dataset)

    os.makedirs(subdirectory, exist_ok=True)

    batch_files = []
    batch_number = 0
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader)):
            # prot_input_ids, prot_attention_mask, drug_input_ids, drug_attention_mask, label = batch
            # prot_input_ids, prot_attention_mask, drug_input_ids, drug_attention_mask = prot_input_ids.to(
            #     args.device), prot_attention_mask.to(args.device), drug_input_ids.to(
            #     args.device), drug_attention_mask.to(args.device)
            #
            # prot_embed, drug_embed = model.encoding(prot_input_ids, prot_attention_mask, drug_input_ids,
            #                                         drug_attention_mask)

            prot_embed, prot_embed_esmc, prot_attention_mask, drug_embed, drug_attention_mask, label, drug_graph_list = batch
            #prot_embed = prot_embed.cpu().to(torch.float16)
            #prot_embed_esmc = prot_embed_esmc.cpu().to(torch.float16)
            prot_embed = prot_embed.cpu()
            prot_embed_esmc = prot_embed_esmc.cpu()
            drug_embed = drug_embed.cpu()

            prot_attention_mask = prot_attention_mask.cpu()
            drug_attention_mask = drug_attention_mask.cpu()
            batched_graph = dgl.batch(drug_graph_list)
            batched_graph.ndata['feat'] = batched_graph.ndata['feat'].to(torch.float16)
            batched_graph.edata['feat'] = batched_graph.edata['feat'].to(torch.float16)

            label = label.cpu()

            # Save each batch to a separate file in the subdirectory
            batch_file = os.path.join(
                subdirectory,
                f"{args.dataset}_{set_type}_batch_{batch_number}.pt"
            )
            torch.save({
                'prot': prot_embed,
                'prot_esmc': prot_embed_esmc,
                'drug': drug_embed,
                'drug_graph': batched_graph,
                'prot_mask': prot_attention_mask,
                'drug_mask': drug_attention_mask,
                'y': label
            }, batch_file,
            _use_new_zipfile_serialization=True)
            batch_files.append(batch_file)
            batch_number += 1
    return batch_files


def get_data_loader(file_list, batch_file, shuffle=False, num_workers=4):
    dataset = BatchFileDataset(file_list)
    return DataLoader(dataset, batch_file, shuffle=shuffle, num_workers=num_workers, collate_fn=lambda x: x[0])

def limit_graph_nodes(graph, max_nodes=100):
    """
    限制DGL图的节点数量，超过max_nodes则取前max_nodes个节点的子图
    并且 **彻底删除 _ID 防止图结构不一致**
    """
    num_nodes = graph.num_nodes()

    if num_nodes <= max_nodes:
        # 关键：即使不截断，也强制清理一遍 _ID，保证统一
        if '_ID' in graph.ndata:
            del graph.ndata['_ID']
        if '_ID' in graph.edata:
            del graph.edata['_ID']
        return graph

    # 截断子图
    selected_nodes = torch.arange(max_nodes)
    subgraph = graph.subgraph(selected_nodes)

    # 强制删除 DGL 自动加的 _ID（必须删！）
    if '_ID' in subgraph.ndata:
        del subgraph.ndata['_ID']
    if '_ID' in subgraph.edata:
        del subgraph.edata['_ID']

    return subgraph


def encode_pretrained_feature(args):
    # Define the path to check for existing batch files
    input_feat_path = os.path.join(args.input_feature_save_path, args.dataset)
    if args.task == 'cluster':
        input_feat_path = os.path.join(args.input_feature_save_path, args.dataset + '_' + args.task)
    else:
        input_feat_path = os.path.join(args.input_feature_save_path, args.dataset)


    # Check if the directory exists, if not, create it
    if not os.path.exists(input_feat_path):
        os.makedirs(input_feat_path)
        # Check if batch files are already saved
    # clear_folder(input_feat_path, verbose=False)
    train_files = sorted([os.path.join(input_feat_path, f) for f in os.listdir(input_feat_path) if
                          f.startswith(f"{args.dataset}_train_batch")])
    valid_files = sorted([os.path.join(input_feat_path, f) for f in os.listdir(input_feat_path) if
                          f.startswith(f"{args.dataset}_valid_batch")])
    test_files = sorted([os.path.join(input_feat_path, f) for f in os.listdir(input_feat_path) if
                         f.startswith(f"{args.dataset}_test_batch")])

    if train_files and valid_files and test_files:
        print("Batch files found and will be used.")
    else:

        protein_embeddings = torch.load(os.path.join(dataset_dir, "protein_embeddings.pth"))
        protein_embeddings_esmc = torch.load(os.path.join(dataset_dir, "protein_embeddings_esmc.pth"))
        drug_embeddings = torch.load(os.path.join(dataset_dir, "selfies_embeddings.pth"))
        with open(os.path.join(dataset_dir, "davis_graphs_dict.pkl"), 'rb') as f:
            drug_graph = pickle.load(f)

        def repeat_pad(x, maxsize):
            b = len(x)
            features = x[0].shape[-1]
            out = torch.zeros(b, maxsize, features)
            for i in range(b):
                a = x[i]
                if a.shape[-2] >= maxsize:
                    out[i, 0: maxsize, :] = a[0:maxsize, :]
                else:
                    out[i, 0:a.shape[-2], :] = a

            return out

        def repeat_pad_pro(x, maxsize):
            b = len(x)
            features = x[0].shape[-1]
            out = torch.zeros(b, maxsize, features)
            for i in range(b):
                a = x[i]
                quot = maxsize // a.shape[-2]
                for j in range(quot):
                    st = j * a.shape[-2]
                    out[i, st: st + a.shape[-2], :] = a
            xp = out.view(-1, 512, 9, out.size()[-1])
            xp = torch.mean(xp, dim=2)
            xp = xp[:, :256, :]
            return xp.to(a.device)

        def collate_fn_batch_encoding(batch):
            # smiles, uniprot, start, end,  query1, query2, scores = zip(*batch)
            smiles, uniprot, scores = zip(*batch)
            drug_graph_list = []
            for smi in smiles:
                # 从字典中获取当前 ID 的嵌入向量（确保 ID 存在于字典中）
                if smi in drug_graph:
                    emb = drug_graph[smi]
                    if emb.num_nodes() > 100:
                        print("1")
                    emb = limit_graph_nodes(emb)
                    drug_graph_list.append(emb)
                else:
                    # 处理不存在的 ID（根据需求选择报错或跳过）
                    raise ValueError(f"smile ID {pid} 不在 smile_embeddings 字典中")

            pro_embedding_list = []
            pro_embedding_list_esmc = []
            for pid in uniprot:
                # 从字典中获取当前 ID 的嵌入向量（确保 ID 存在于字典中）
                # pid = str(pid)
                lookup_pid = pid.upper() if str(pid).startswith('pp') else pid
                if lookup_pid in protein_embeddings:
                    emb = protein_embeddings[lookup_pid]['embeddings']
                    pro_embedding_list.append(emb)
                    pro_embedding_list_esmc.append(protein_embeddings_esmc[lookup_pid]['embeddings'])
                elif pid in protein_embeddings:
                    emb = protein_embeddings[pid]['embeddings']
                    pro_embedding_list.append(emb)
                    pro_embedding_list_esmc.append(protein_embeddings_esmc[pid]['embeddings'])
                else:
                    raise ValueError(f"UniProt ID {pid} 不在 protein_embeddings 字典中")

            drug_embedding_list = []
            for smi in smiles:
                # 从字典中获取当前 ID 的嵌入向量（确保 ID 存在于字典中）
                if smi in drug_embeddings:
                    emb = drug_embeddings[smi]['embeddings']
                    drug_embedding_list.append(emb)
                else:
                    # 处理不存在的 ID（根据需求选择报错或跳过）
                    raise ValueError(f"smile ID {pid} 不在 smile_embeddings 字典中")


            #pro_embedding = repeat_pad(pro_embedding_list, 512)
            #pro_embedding_esmc = repeat_pad(pro_embedding_list_esmc, 512)
            dru_embedding = repeat_pad(drug_embedding_list, 100)
            pro_embedding = repeat_pad_pro(pro_embedding_list, 9 * 512)
            pro_embedding_esmc = repeat_pad_pro(pro_embedding_list_esmc, 9 * 512)
            print(pro_embedding_esmc.shape)
            print(pro_embedding.shape)


            pro_sum_result = torch.sum(pro_embedding, dim=2)
            pro_embedding_mask = pro_sum_result != 0

            dru_sum_result = torch.sum(dru_embedding, dim=2)
            dru_embedding_mask = dru_sum_result != 0
            scores = torch.tensor(list(scores), dtype=torch.float32)
            return pro_embedding, pro_embedding_esmc, pro_embedding_mask, dru_embedding, dru_embedding_mask, scores, drug_graph_list

        Dataset = DatabaseProcessor(args)
        train_examples = Dataset.get_train_examples()
        valid_examples = Dataset.get_val_examples()
        test_examples = Dataset.get_test_examples()



        train_dataloader = DataLoader(
            train_examples,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn_batch_encoding,
        )
        valid_dataloader = DataLoader(
            valid_examples,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn_batch_encoding,
        )
        test_dataloader = DataLoader(
            test_examples,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn_batch_encoding,
        )
        print(f"dataset loaded: train-{len(train_examples)}; valid-{len(valid_examples)}; test-{len(test_examples)}")

        train_files = get_feature(model, train_dataloader, args, "train")
        valid_files = get_feature(model, valid_dataloader, args, "valid")
        test_files = get_feature(model, test_dataloader, args, "test")

    return train_files, valid_files, test_files


def train(model, train_loader, valid_loader, criterion, optimizer, scheduler, device, num_epochs=200, patience=50):
    best_auc = 0
    best_model = None
    epochs_without_improvement = 0  # Initialize counter for early stopping

    # 创建epoch进度条
    epoch_pbar = tqdm(range(num_epochs), desc="Training Progress", unit="epoch")
    maeloss = nn.L1Loss(reduction='mean')
    for epoch in epoch_pbar:
        start_time = time.time()

        # Training phase
        model.train()
        total_loss = 0

        # 创建batch进度条
        batch_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False, unit="batch")
        for step, batch in enumerate(batch_pbar):
            prot, prot_esmc, drug,drug_graph,  prot_mask, drug_mask, label = batch
            prot, prot_esmc, drug, prot_mask, drug_mask, label = prot.to(device),prot_esmc.to(device),  drug.to(device), prot_mask.to(
                device), drug_mask.to(device), label.to(device).long()
            drug_graph = drug_graph.to(device)
            optimizer.zero_grad()

            final_out, logit, tcp= model(prot,prot_esmc,  drug,drug_graph, prot_mask, drug_mask, 'pdf_train')
            loss = get_loss(final_out, logit, tcp, label, 2)
            total_loss += loss.item()


            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()

            # 更新batch进度条信息
            batch_pbar.set_postfix({
                'Batch Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss / len(batch_pbar):.4f}'
            })

        scheduler.step()
        batch_pbar.close()

        # Validation phase
        model.eval()
        val_start_time = time.time()

        with torch.no_grad():
            predictions, actuals = [], []
            # 验证阶段的进度条
            val_pbar = tqdm(valid_loader, desc='Validating', leave=False, unit="batch")

            for batch in val_pbar:
                prot, prot_esmc, drug,drug_graph,  prot_mask, drug_mask, label = batch
                prot, prot_esmc, drug, prot_mask, drug_mask, label = prot.to(device), prot_esmc.to(device), drug.to(
                    device), prot_mask.to(
                    device), drug_mask.to(device), label.to(device)
                drug_graph = drug_graph.to(device)
                final_out, preds, weights = model(prot, prot_esmc, drug, drug_graph, prot_mask, drug_mask, 'pdf_test')
                output = final_out.softmax(-1)[:, 1]
                predictions.extend(output.flatten().cpu().numpy())
                actuals.extend(label.cpu().numpy())

            val_pbar.close()

            auc = roc_auc_score(actuals, predictions)
            validation_time = time.time() - val_start_time
            epoch_time = time.time() - start_time

            # 更新epoch进度条信息
            epoch_pbar.set_postfix({
                'Loss': f'{total_loss / len(train_loader):.4f}',
                'Val AUC': f'{auc:.4f}',
                'Best AUC': f'{best_auc:.4f}',
                'Epoch Time': f'{epoch_time:.1f}s',
                'Val Time': f'{validation_time:.1f}s'
            })

            # 打印详细日志
            print(f'\nEpoch {epoch + 1}: '
                  f'Loss: {total_loss / len(train_loader):.4f}, '
                  f'Val AUC: {auc:.4f}, '
                  f'Best AUC: {best_auc:.4f}, '
                  f'Epoch Time: {epoch_time:.1f}s')

            # Log metrics to wandb
            # wandb.log({"epoch": epoch + 1, "loss": total_loss / len(train_loader), "val_auc": auc})

            if auc > best_auc:
                best_auc = auc
                best_model = model.state_dict()
                # Save the best model
                torch.save(best_model, f'{best_model_dir}/best_model.ckpt')
                epochs_without_improvement = 0
                print(f'🔥 New best model! AUC: {auc:.4f}')
                test(model, test_loader, device)
            else:
                epochs_without_improvement += 1
                print(f'⏳ No improvement for {epochs_without_improvement} epoch(s)')

            if epochs_without_improvement >= patience:
                print(f'🛑 Early stopping triggered after {epoch + 1} epochs.')
                epoch_pbar.close()
                break

    epoch_pbar.close()
    print(f'🏆 Training completed! Best AUC: {best_auc:.4f}')
    return best_model


def test(model, test_loader, device):
    model.eval()
    predictions, actuals = [], []
    fused_reliability = None
    with torch.no_grad():
        for batch in test_loader:
            prot, prot_esmc, drug, drug_graph, prot_mask, drug_mask, label = batch
            prot, prot_esmc, drug, prot_mask, drug_mask, label = prot.to(device), prot_esmc.to(device), drug.to(
                device), prot_mask.to(
                device), drug_mask.to(device), label.to(device)
            drug_graph = drug_graph.to(device)
            # out, _, _, _, _ = model(prot, prot_esmc, drug,drug_graph,  prot_mask, drug_mask, 'pdf_test')
            # output = F.softmax(out, dim=1)[:, 1]
            # output, _ = model(prot, prot_esmc, drug, drug_graph, prot_mask, drug_mask, label)
            # final_out, preds, weights = model(prot, prot_esmc, drug, drug_graph, prot_mask, drug_mask, 'pdf_test',
            #                                   ret=ret)
            final_out, preds, weights, kappas, self_entropy, cross_entropy, rel = model(prot, prot_esmc, drug, drug_graph, prot_mask, drug_mask, 'pdf_test',
                                               ret=True)

            # final_out, preds, weights = model(prot, prot_esmc, drug, drug_graph, prot_mask, drug_mask, 'pdf_test')

            if fused_reliability is None:
                fused_reliability = rel.cpu().numpy()
            else:
                fused_reliability = np.concatenate([fused_reliability, rel.cpu().numpy()])

            output = final_out.softmax(-1)[:, 1]
            predictions.extend(output.squeeze().cpu().numpy())
            # predictions.append(output.item())
            actuals.extend(label.cpu().numpy())


    print("=" * 60)
    print(fused_reliability[:, 0].mean().item())
    std_val = fused_reliability[:, 0].std()
    print(std_val)

    print(fused_reliability[:, 1].mean().item())
    std_val = fused_reliability[:, 1].std()
    print(std_val)
    y_label = actuals
    y_pred = predictions
    auroc = roc_auc_score(y_label, y_pred)
    auprc = average_precision_score(y_label, y_pred)
    fpr, tpr, thresholds = roc_curve(y_label, y_pred)
    prec, recall, _ = precision_recall_curve(y_label, y_pred)

    # Youden index for the optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    y_pred_bin = (y_pred >= optimal_threshold).astype(int)

    # confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_label, y_pred_bin).ravel()

    acc = accuracy_score(y_label, y_pred_bin)
    sensitivity = tp / (tp + fn)  # recall
    specificity = tn / (tn + fp)
    f1 = f1_score(y_label, y_pred_bin)

    # 格式化打印关键指标（保留4位小数，增强可读性）
    print("=" * 50)
    print("模型评估关键指标")
    print("=" * 50)
    print(f"AUROC（ROC曲线下面积）: {auroc:.4f}")
    print(f"AUPRC（PR曲线下面积）: {auprc:.4f}")
    print(f"Sensitivity（敏感度/召回率）: {sensitivity:.4f}")
    print(f"Specificity（特异度）: {specificity:.4f}")
    print(f"Accuracy（准确率）: {acc:.4f}")
    print("=" * 50)





if __name__ == "__main__":
    args = parse_config()
    if args.dataset == 'BindingDB':
        name = "BindingDB"
    elif args.dataset == 'Human':
        name = "Human"
    elif args.dataset == 'Biosnap':
        name = "Biosnap"
    elif args.dataset == 'Davis':
        name = "Davis"
    elif args.dataset == 'Drugbank':
        name = "Drugbank"
    elif args.dataset == 'BindingDB_cluster':
        name = "BindingDB_cluster"
    elif args.dataset == 'human':
        name = "human"
    elif args.dataset == 'Biosnap_cluster':
        name = "Biosnap_cluster"
    else:
        raise ValueError("Invalid dataset name provided. Please choose from 'BindingDB', 'Human', or 'Biosnap'.")

    set_seed(seed=args.seed)
    # Setup dataset directory
    dataset_dir = f"dataset/{name}/"

    device = torch.device(args.device)
    print(f"Current device: {args.device}.")
    # wandb.init(project="DTI_Prediction_with_Token-level_Fusion", config=args, save_code=True)

    # wandb.config.update(args)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(args)
    best_model_dir = (
        f"{args.save_path_prefix}{args.dataset}_{args.fusion}")
    os.makedirs(best_model_dir, exist_ok=True)

    model = PMLDTI(1280, 768, args).to(device)
    criterion = nn.BCELoss()

    epoch = args.epoch

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer=optimizer,
        first_cycle_steps=epoch,
        max_lr=1e-4,
        min_lr=1e-8,
        warmup_steps=int(epoch * args.warmup_ratio)
    )

    # Load features from the saved batch files
    train_files, valid_files, test_files = encode_pretrained_feature(args)
    train_loader = get_data_loader(train_files, batch_file=1, shuffle=True)
    valid_loader = get_data_loader(valid_files, batch_file=1, shuffle=False)
    test_loader = get_data_loader(test_files, batch_file=1, shuffle=False)
    best_model = train(model, train_loader, valid_loader, criterion, optimizer, scheduler, device, num_epochs=epoch, patience=args.patience)
    model.load_state_dict(torch.load(f'{best_model_dir}/best_model.ckpt', map_location=device))
    test(model, test_loader, device)

    # wandb.finish()