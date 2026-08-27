import dgl
import numpy as np
import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from dgl.data.utils import save_graphs

from scipy import sparse as sp
from itertools import permutations
from scipy.spatial import distance_matrix
from dgl import load_graphs
import pickle

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import warnings

warnings.filterwarnings("ignore")

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}


def label_smiles(line, smi_ch_ind, MAX_SMI_LEN=100):
    X = np.zeros(MAX_SMI_LEN, dtype=np.int64())
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i] = smi_ch_ind[ch]
    return X


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix(scipy_fmt='csr').astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    if EigVec.shape[1] < pos_enc_dim + 1:
        PadVec = np.zeros((EigVec.shape[0], pos_enc_dim + 1 - EigVec.shape[1]), dtype=EigVec.dtype)
        EigVec = np.concatenate((EigVec, PadVec), 1)
    g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()
    return g


def atom_features(atom, explicit_H=False, use_chirality=True):
    """Generate atom features including atom symbol(17),degree(7),formal charge(1),
    radical electrons(1),hybridization(6),aromatic(1),hydrogen atoms attached(5),Chirality(3)
    """
    symbol = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'B', 'Si', 'Fe', 'Zn', 'Cu', 'Mn', 'Mo', 'other']  # 17-dim
    degree = [0, 1, 2, 3, 4, 5, 6]  # 7-dim
    hybridizationType = [Chem.rdchem.HybridizationType.SP,
                         Chem.rdchem.HybridizationType.SP2,
                         Chem.rdchem.HybridizationType.SP3,
                         Chem.rdchem.HybridizationType.SP3D,
                         Chem.rdchem.HybridizationType.SP3D2,
                         'other']  # 6-dim
    results = one_of_k_encoding_unk(atom.GetSymbol(), symbol) + \
              one_of_k_encoding(atom.GetDegree(), degree) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) + [
                  atom.GetIsAromatic()]  # 17+7+2+6+1=33

    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                  [0, 1, 2, 3, 4])  # 33+5=38
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]  # 38+3 =41
    return results


def bond_features(bond, use_chirality=True):
    """Generate bond features including bond type(4), conjugated(1), in ring(1), stereo(4)"""
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    return np.array(bond_feats).astype(int)


def smiles_to_graph(smiles, explicit_H=False, use_chirality=True):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        raise RuntimeError("SMILES cannot been parsed!")
    g = dgl.DGLGraph()
    # Add nodes
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)

    atom_feats = np.array([atom_features(a, explicit_H=explicit_H) for a in mol.GetAtoms()])
    if use_chirality:
        chiralcenters = Chem.FindMolChiralCenters(mol, force=True, includeUnassigned=True,
                                                  useLegacyImplementation=False)
        chiral_arr = np.zeros([num_atoms, 3])
        for (i, rs) in chiralcenters:
            if rs == 'R':
                chiral_arr[i, 0] = 1
            elif rs == 'S':
                chiral_arr[i, 1] = 1
            else:
                chiral_arr[i, 2] = 1
        atom_feats = np.concatenate([atom_feats, chiral_arr], axis=1)

    g.ndata["feat"] = torch.tensor(atom_feats)

    # Add edges
    src_list = []
    dst_list = []
    bond_feats_all = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        bond_feats = bond_features(bond, use_chirality=use_chirality)
        src_list.extend([u, v])
        dst_list.extend([v, u])
        bond_feats_all.append(bond_feats)
        bond_feats_all.append(bond_feats)

    g.add_edges(src_list, dst_list)

    g.edata["feat"] = torch.tensor(np.array(bond_feats_all))
    g = limit_graph_nodes(g)
    if '_ID' in g.ndata:
        g.ndata.pop('_ID')
    return g

def limit_graph_nodes(graph, max_nodes=200):
    """
    限制DGL图的节点数量，超过max_nodes则取前max_nodes个节点的子图

    Args:
        graph: DGL Graph对象
        max_nodes: 最大节点数，默认200

    Returns:
        subgraph: 处理后的子图（节点数≤max_nodes）
    """
    # 获取当前图的节点数量
    num_nodes = graph.num_nodes()

    # 如果节点数≤max_nodes，直接返回原图
    if num_nodes <= max_nodes:
        return graph

    # 如果节点数>max_nodes，提取前max_nodes个节点的索引
    selected_nodes = torch.arange(max_nodes)  # 生成0~199的索引

    # 生成子图（自动保留选中节点的关联边和所有特征）
    subgraph = graph.subgraph(selected_nodes)

    # 验证子图信息（可选，用于调试）
    print(f"原图节点数: {num_nodes} → 子图节点数: {subgraph.num_nodes()}")
    print(f"子图边数: {subgraph.num_edges()}")

    return subgraph

def process_davis_dataset(dataset="BindingDB_cluster"):
    """处理Davis数据集并保存图数据字典"""
    # 读取Davis数据集
    davis_path = f"../dataset/{dataset}/SEFLIES.csv"

    try:
        df = pd.read_csv(davis_path)
        print(f"成功读取数据集，共 {len(df)} 行")
    except Exception as e:
        print(f"读取数据集失败: {e}")
        return

    # 获取独特的SMILES
    unique_smiles = df['SMILES'].unique()
    print(f"发现 {len(unique_smiles)} 个独特的SMILES")



    # 创建字典：SMILES为键，图对象为值
    graph_dict = {}
    failed_smiles = []

    # 处理每个SMILES
    for i, smiles in enumerate(unique_smiles):
        if pd.isna(smiles) or smiles == "":
            failed_smiles.append(smiles)
            continue

        print(f"处理 {i + 1}/{len(unique_smiles)}: {smiles}")

        try:
            graph = smiles_to_graph(smiles)

            if graph is None:
                failed_smiles.append(smiles)
                print(f"  -> 失败")
            else:
                # 将SMILES作为键，图对象作为值添加到字典
                graph_dict[smiles] = graph
                print(f"  -> 成功 (节点数: {graph.number_of_nodes()}, 边数: {graph.number_of_edges()})")

        except Exception as e:
            failed_smiles.append(smiles)
            print(f"  -> 异常: {e}")

    # 输出统计结果
    print("\n" + "=" * 50)
    print("统计结果:")
    print(f"总独特SMILES数: {len(unique_smiles)}")
    print(f"成功生成图的数量: {len(graph_dict)}")
    print(f"失败生成图的数量: {len(failed_smiles)}")
    print(f"成功率: {len(graph_dict) / len(unique_smiles) * 100:.2f}%")

    # 保存图字典
    if graph_dict:
        output_dir = f"../dataset/{dataset}"
        os.makedirs(output_dir, exist_ok=True)

        # 保存图字典
        graph_file = os.path.join(output_dir, "davis_graphs_dict.pkl")
        with open(graph_file, 'wb') as f:
            pickle.dump(graph_dict, f)
        print(f"图字典已保存到: {graph_file}")

        # 保存失败列表
        if failed_smiles:
            failed_file = os.path.join(output_dir, "failed_smiles.txt")
            with open(failed_file, 'w') as f:
                for smiles in failed_smiles:
                    f.write(f"{smiles}\n")
            print(f"失败SMILES列表已保存到: {failed_file}")

        # 打印一些示例信息
        print("\n前5个图的详细信息:")
        for i, (smiles, graph) in enumerate(list(graph_dict.items())[:5]):
            print(f"图 {i + 1}:")
            print(f"  SMILES: {smiles}")
            print(f"  节点数: {graph.number_of_nodes()}")
            print(f"  边数: {graph.number_of_edges()}")
            print(f"  节点特征维度: {graph.ndata['feat'].shape[1]}")
            print(f"  边特征维度: {graph.edata['feat'].shape[1]}")
            print()
    else:
        print("没有成功生成的图，无法保存数据")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="BindingDB_cluster", help="Dataset name under dataset/")
    args = parser.parse_args()
    process_davis_dataset(dataset=args.dataset)