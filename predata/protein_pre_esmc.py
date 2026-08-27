import pandas as pd
import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="BindingDB_cluster", help="Dataset name under dataset/")
args = parser.parse_args()

# 设置环境变量
os.environ["INFRA_PROVIDER"] = "True"

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# 加载ESMC模型
print("正在加载ESMC模型...")
client = ESMC.from_pretrained("esmc_300m").to(device)
print("模型加载完成")

# 读取CSV文件
csv_path = f"../dataset/{args.dataset}/Alpha_seq.csv"
df = pd.read_csv(csv_path, header=None)

# 创建字典：uid为键，Seq为值
seq_dict = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
print(f"总共 {len(seq_dict)} 个蛋白质序列")

# 存储所有序列的嵌入
embeddings_dict = {}

# ESMC模型最大序列长度
MAX_SEQ_LENGTH = 1022

# 遍历字典并提取嵌入
for uid, seq in seq_dict.items():
    try:
        # 检查序列长度，如果超过最大长度则截断
        if len(seq) > MAX_SEQ_LENGTH:
            print(f"警告: 序列 {uid} 长度 {len(seq)} 超过最大长度 {MAX_SEQ_LENGTH}，将被截断")
            seq = seq[:MAX_SEQ_LENGTH]

        # 创建ESMProtein对象
        protein = ESMProtein(sequence=seq)

        # 编码蛋白质序列
        protein_tensor = client.encode(protein)

        # 获取logits和嵌入
        logits_output = client.logits(
            protein_tensor,
            LogitsConfig(sequence=True, return_embeddings=True)
        )

        # 提取嵌入并去除特殊token（第一个和最后一个）
        embedding = logits_output.embeddings[0][:, :]
        print(embedding.shape)

        # 存储到字典中
        embeddings_dict[uid] = {
            'sequence': seq,
            'embeddings': embedding.cpu(),  # 移到CPU以节省GPU内存
        }

        print(f"UID: {uid}, 序列长度: {len(seq)}, 嵌入形状: {embedding.shape}")

    except Exception as e:
        print(f"处理序列 {uid} 时出错: {str(e)}")
        continue

# 统计截断情况


# 可选：保存嵌入到文件
output_path = f"../dataset/{args.dataset}/protein_embeddings_esmc.pth"
torch.save(embeddings_dict, output_path)
print(f"嵌入已保存到: {output_path}")