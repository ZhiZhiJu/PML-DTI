import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

metrics = ['AUROC', 'ACC', 'Specificity', 'Sensitivity', 'AUPRC']

# ======================
# ✅ BioSNAP_cluster 数据（已重排）
# [AUROC, ACC, Specificity, Sensitivity, AUPRC]
# ======================
data = {
    'MolTrans':      [0.631, 0.604, 0.649, 0.560, 0.632],
    'DrugBAN':       [0.685, 0.654, 0.786, 0.523, 0.722],
    'DrugLAMP':      [0.738, 0.674, 0.806, 0.542, 0.766],
    'MMDG-DTI':      [0.717, 0.662, 0.683, 0.640, 0.748],
    'TAPB':          [0.791, 0.732, 0.789, 0.676, 0.800],
    'MIF-DTI':       [0.683, 0.643, 0.760, 0.526, 0.695],
    'ConfBiXtCPI':   [0.776, 0.715, 0.782, 0.648, 0.799],
    'PLM-DTI(Ours)': [0.794, 0.731, 0.768, 0.693, 0.799]
}

colors = ['#8FA3C3', '#99BADB', '#ADC9E4', '#C6D49D', '#F6BF94', '#BBBAB9', '#F6D0CD', '#F1A094']

N = len(metrics)
theta = np.linspace(0, 2 * np.pi, N, endpoint=False)

fig, ax = plt.subplots(figsize=(7,7), subplot_kw={'projection':'polar'})

ax.set_theta_offset(np.pi/2)
ax.set_theta_direction(-1)
ax.set_xticks(theta)
ax.set_xticklabels([])

# ======================
# ✅ 半径范围（你要求）
# ======================
ax.set_ylim(0.4, 1)

ax.set_rgrids(
    radii=[0.4, 0.6, 0.8],
    labels=['', '', ''],
    angle=90
)

ax.xaxis.grid(True, linestyle='-', linewidth=2, color='#D9D9DA', alpha=0.8)
ax.yaxis.grid(True, linestyle=(0, (10, 6)), linewidth=2, color='#D9D9DA', alpha=0.8)
ax.spines['polar'].set_linewidth(1.5)

# ======================
# 画线
# ======================
for i, (name, values) in enumerate(data.items()):
    vs = values + [values[0]]
    ts = list(theta) + [theta[0]]
    ax.plot(ts, vs, color=colors[i], linewidth=2)
    ax.scatter(ts, vs, color=colors[i], s=20, zorder=5)

# ======================
# 半径刻度文字
# ======================
ax.text(np.pi/2, 0.4, '0.4', fontsize=16, ha='center', va='center')
ax.text(np.pi/2, 0.6, '0.6', fontsize=16, ha='center', va='center')
ax.text(np.pi/2, 0.8, '0.8', fontsize=16, ha='center', va='center')

# ======================
# 保存
# ======================
plt.savefig(
    'BioSNAP_cluster_radar.png',
    dpi=600,
    bbox_inches='tight',
    pad_inches=0.05
)

# plt.savefig(r'D:\STUDY\npjDM\画图\BioSNAP_cluster.pdf', dpi=300, bbox_inches='tight')
# plt.savefig(r'D:\STUDY\npjDM\画图\BioSNAP_cluster.png', dpi=300, bbox_inches='tight')