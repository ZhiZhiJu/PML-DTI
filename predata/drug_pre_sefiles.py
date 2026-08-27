import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
import os
import numpy as np
from collections import Counter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="BindingDB_cluster", help="Dataset name under dataset/")
args = parser.parse_args()

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载模型和tokenizer
model_path = "../language/SELFormer"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, ignore_mismatched_sizes=True)
# model = AutoModel.from_pretrained("/data/zhangzhijun/2025-10/FusionDTI-main/language/MoLFormer-XL-both-10pct",
#                                                           deterministic_eval=True,
#                                                           trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained(
#                 "/data/zhangzhijun/2025-10/FusionDTI-main/language/MoLFormer-XL-both-10pct", trust_remote_code=True)

# 将模型移动到GPU
model.to(device)
model.eval()  # 设置为评估模式

# 读取CSV文件
csv_path = f"../dataset/{args.dataset}/SEFLIES.csv"
df = pd.read_csv(csv_path)

# 创建字典：SMILES为键，selfies为值
selfies_dict = dict(zip(df['SMILES'], df['selfies']))
print(f"总共 {len(selfies_dict)} 个SELFIES序列")

# 存储所有序列的嵌入和token长度
embeddings_dict = {}
token_lengths = []  # 存储所有序列的token长度

max_drug_atoms = 512

# 遍历字典并提取嵌入
for smiles, selfies_str in selfies_dict.items():
    try:
        # 跳过空值
        if pd.isna(selfies_str) or selfies_str == "":
            continue

        # # 编码序列（包含attention_mask）
        encoded = tokenizer.encode(selfies_str, add_special_tokens=True, max_length=512, padding=True, truncation=True)
        token_length = len(encoded)  # 获取token长度
        token_lengths.append(token_length)

        token_tensor = torch.tensor([encoded]).to(device)

        # 获取模型输出（不计算梯度以节省内存）
        with torch.no_grad():
            outputs = model(
                token_tensor,
                output_hidden_states=True
            )
        # inputs = tokenizer(smiles[: max_drug_atoms - 2], padding=True, return_tensors="pt").to(device)
        # print(inputs.input_ids.shape)
        # print(inputs.attention_mask.shape)
        # print(inputs.input_ids.shape)
        #
        # with torch.no_grad():
        #     outputs = model(**inputs)
        embed = outputs.last_hidden_state[0][1:-1]
        print(embed.shape,  "embed")

        # 提取token嵌入（去掉[CLS]和[SEP]特殊token）
        # outputs.last_hidden_state形状: [batch_size, sequence_length, hidden_size]
        # sequence_out = outputs.last_hidden_state[0][1:-1]  # [sequence_length-2, hidden_size]

        # 存储到字典中
        embeddings_dict[smiles] = {
            'selfies': selfies_str,
            'embeddings': embed.cpu(),  # 移到CPU存储以节省GPU内存
        }

        if len(embeddings_dict) % 100 == 0:  # 每处理100个序列打印一次进度
            print(f"已处理 {len(embeddings_dict)} 个序列")

    except Exception as e:
        print(f"处理SMILES {smiles} 时出错: {str(e)}")
        continue



# 保存嵌入到文件
output_path = f"../dataset/{args.dataset}/selfies_embeddings.pth"
torch.save(embeddings_dict, output_path)
print(f"\n嵌入已保存到: {output_path}")

# 使用示例BindingDB_cluster
if embeddings_dict:
    sample_smiles = list(embeddings_dict.keys())[0]
    sample_data = embeddings_dict[sample_smiles]
    print(f"\n示例序列: {sample_smiles}")
    print(f"SELFIES: {sample_data['selfies']}")
    print(f"嵌入形状: {sample_data['embeddings'].shape}")
    print(f"Token长度: {sample_data['token_length']}")