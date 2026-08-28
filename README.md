# 🧬 PML-DTI

<div align="center">

**Fine-grained Drug–Target Interaction Prediction via Token-level Fusion**

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-ee4c2c)](https://pytorch.org/)
[![DGL](https://img.shields.io/badge/DGL-2.4.0-green)](https://www.dgl.ai/)

</div>

PML-DTI is a deep-learning framework for **Drug–Target Interaction (DTI)** prediction. It learns fine-grained interactions between drugs and protein targets via a **token-level fusion** module, combining:

- 💊 **Drug inputs** — SELFIES (recommended) or SMILES, encoded by a molecular pre-trained language model (SELFormer), together with a molecular graph.
- 🧬 **Protein inputs** — amino-acid sequences (encoded by ESM-C) and **structure-aware (SA)** sequences (encoded by SaProt).

---

## 📦 Requirements

Create a conda environment and install the dependencies from `requirements.txt`:

```bash
conda create -n pml_dti python=3.10
conda activate pml_dti
```

> 🧩 **mamba-ssm & causal-conv1d** — these two require matching CUDA/PyTorch builds. Download the pre-built `.whl` files from [Google Drive](https://drive.google.com/file/d/15RLTgf1GYXip5YTdNScaKiTRd1xJWdmt/view?usp=sharing), then install them **in order** (causal-conv1d first, then mamba-ssm):
>
> ```bash
> pip install causal_conv1d-1.1.2+cu118torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
> pip install mamba_ssm-1.1.3+cu118torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
> ```

Key dependencies (full list in `requirements.txt`):

| Package | Version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.2.0+cu118 |
| DGL | 2.4.0+cu118 |
| transformers | 4.57.1 |
| scikit-learn | 1.7.2 |
| pandas | 2.3.3 |
| numpy | 1.26.4 |
| RDKit | 2024.3.2 |
| PyYAML | 6.0.3 |

### 🤗 Pre-trained encoders

Download the pre-trained encoders into the `language/` folder:

| Model | Path | Input | HuggingFace |
|---|---|---|---|
| SELFormer | `language/SELFormer` | SELFIES | [HUBioDataLab/SELFormer](https://huggingface.co/HUBioDataLab/SELFormer) |
| SaProt | `language/SaProt_650M_AF2` | structure-aware sequence | [westlake-repl/SaProt_650M_AF2](https://huggingface.co/westlake-repl/SaProt_650M_AF2) |

> ⚠️ **Note:** ESM-C (`esmc_300m`) requires the [`esm`](https://github.com/Biohub/esm) package and has dependency conflicts with the main environment — install it in a **separate conda environment** and run `predata/protein_pre_esmc.py` there. See [Biohub/esm](https://github.com/Biohub/esm) for installation instructions.

---

## ⬇️ Downloads

| Resource | Description | Link |
|---|---|---|
| 📂 Datasets | Processed datasets (CSV files under `dataset/`) | [Google Drive](https://drive.google.com/file/d/1D3-Z-KfmPCWav6EewzusYbK2ExeJPRS8/view?usp=sharing) |
| 🧠 ESM-C weights | ESM-C (`esmc_300m`) model weights | [Google Drive](https://drive.google.com/file/d/1KmYUtf_ArJv0iqJsNDqTny7hxW_u2o6W/view?usp=sharing) |
| 🏋️ Trained checkpoints | Model weights for each dataset | [Google Drive](https://drive.google.com/file/d/19X8QQUbgYdMxGjTZGR6tINUCyfiuLCUm/view?usp=sharing) |

---

## 🚀 Data preparation (3 steps)

Each dataset folder under `dataset/<DATASET>/` should contain:

- 📄 `full.csv` (or `train.csv` / `val.csv` / `test.csv`) — columns `SMILES`, `uid`, `label`
- 🧬 `Alpha_seq.csv` — `uid` → amino-acid sequence
- 🏗️ `Alpha_stru_seq.csv` — `uid` → structure-aware (SA) sequence

### Step 1 — 🧪 Generate the SELFIES representation

Convert the `SMILES` column into `selfies`:

```bash
python generate_selfies.py --dataset <DATASET>
```

This produces `SEFLIES.csv` (SMILES + SELFIES).

### Step 2 — 🔬 Preprocess the data (4 scripts in `predata/`)

Run the following four scripts to generate the embeddings and molecular graphs required by training:

| Script | Input | Output | Description |
|---|---|---|---|
| `predata/drug_pre_sefiles.py` | `SEFLIES.csv` | `selfies_embeddings.pth` | SELFormer embeddings of SELFIES |
| `predata/protein_pre_saprot.py` | `Alpha_stru_seq.csv` | `protein_embeddings.pth` | SaProt embeddings of SA sequences |
| `predata/protein_pre_esmc.py` | `Alpha_seq.csv` | `protein_embeddings_esmc.pth` | ESM-C embeddings of amino-acid sequences |
| `predata/drug_graph_creat.py` | `SEFLIES.csv` | `davis_graphs_dict.pkl` | Molecular graphs of drugs |

```bash
python predata/drug_pre_sefiles.py --dataset <DATASET>
python predata/protein_pre_saprot.py --dataset <DATASET>
python predata/protein_pre_esmc.py --dataset <DATASET>
python predata/drug_graph_creat.py --dataset <DATASET>
```

Each script reads from / writes to `dataset/<DATASET>/`.

### Step 3 — 🚀 Train

```bash
# cluster split (source / target domain)
python main_token.py --dataset BindingDB_cluster --task cluster

# or a standard random split
python main_token.py --dataset BindingDB --task random
```

On the first run, `main_token.py` loads the four files generated in Step 2 (`selfies_embeddings.pth`, `protein_embeddings.pth`, `protein_embeddings_esmc.pth`, `davis_graphs_dict.pkl`) and caches token-level features; subsequent runs reuse the cache.

---

## ⚙️ Dataset configuration

Per-dataset hyper-parameters live in `config/{dataset}_{task}.yaml` (e.g. `config/BindingDB_cluster.yaml`):

```yaml
batch_size: 64
num_heads: 1
mm_output_sizes: [256]
patience: 25
```

Global model parameters live in `config/config.yaml`.

---

## 📁 Directory layout

```
.
├── config/                     # yaml configuration
├── dataset/                    # datasets (CSV + generated .pth/.pkl)
├── language/                   # pre-trained encoders
├── predata/                    # preprocessing scripts (Step 2)
├── generate_selfies.py         # Step 1
├── main_token.py               # Step 3 (training)
└── utils/                      # model and helper modules
```

---
