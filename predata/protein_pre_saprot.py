import pandas as pd
import torch
from transformers import EsmTokenizer, EsmForMaskedLM
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="BindingDB_cluster", help="Dataset name under dataset/")
args = parser.parse_args()

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# 加载模型和tokenizer
model_path = r"../language/SaProt_650M_AF2"
tokenizer = EsmTokenizer.from_pretrained(model_path)
model = EsmForMaskedLM.from_pretrained(model_path)
model.to(device)
model.eval()  # 设置为评估模式

# 读取CSV文件
csv_path = f"../dataset/{args.dataset}/Alpha_stru_seq.csv"
df = pd.read_csv(csv_path, header=None)

# 创建字典：uid为键，Seq为值
seq_dict = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
print(f"总共 {len(seq_dict)} 个蛋白质序列")

# 存储所有序列的嵌入
embeddings_dict = {}

# 遍历字典并提取嵌入
for uid, seq in seq_dict.items():
    try:
        # 对序列进行tokenize
        tokens = tokenizer.tokenize(seq)



        # 编码序列
        inputs = tokenizer(seq, return_tensors="pt", max_length=1024, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 获取模型输出（不计算梯度以节省内存）
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][0]


        # 提取token嵌入（去掉[CLS]和[SEP]特殊token）
        token_embeddings = outputs[:, :]

        # 存储到字典中
        embeddings_dict[uid] = {
            'sequence': seq,
            'embeddings': token_embeddings.cpu(),  # 移到CPU以节省GPU内存
        }

        print(f"UID: {uid}, 嵌入形状: {token_embeddings.shape}")

    except Exception as e:
        print(f"处理序列 {uid} 时出错: {str(e)}")
        continue

print(f"成功处理 {len(embeddings_dict)} 个序列的嵌入")

# 可选：保存嵌入到文件
output_path = f"../dataset/{args.dataset}/protein_embeddings.pth"
torch.save(embeddings_dict, output_path)
print(f"嵌入已保存到: {output_path}")
