import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

metrics = ['AUROC', 'ACC', 'Specificity', 'Sensitivity', 'AUPRC']

# ======================
# ✅ BindingDB_cluster 数据（已重排）
# [AUROC, ACC, Specificity, Sensitivity, AUPRC]
# ======================
data = {
    'MolTrans':      [0.605, 0.582, 0.534, 0.638, 0.554],
    'DrugBAN':       [0.601, 0.579, 0.612, 0.541, 0.570],
    'DrugLAMP':      [0.645, 0.608, 0.478, 0.758, 0.591],
    'MMDG-DTI':      [0.638, 0.596, 0.576, 0.621, 0.583],
    'TAPB':          [0.676, 0.630, 0.565, 0.705, 0.628],
    'MIF-DTI':       [0.628, 0.602, 0.586, 0.621, 0.573],
    'ConfBiXtCPI':   [0.676, 0.618, 0.544, 0.703, 0.589],
    'PLM-DTI(Ours)': [0.684, 0.636, 0.528, 0.761, 0.619]
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
# ✅ 半径范围（统一）
# ======================
ax.set_ylim(0.3, 1)

ax.set_rgrids(
    radii=[0.3, 0.5, 0.7],
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
ax.text(np.pi/2, 0.4, '0.3', fontsize=16, ha='center', va='center')
ax.text(np.pi/2, 0.6, '0.5', fontsize=16, ha='center', va='center')
ax.text(np.pi/2, 0.8, '0.7', fontsize=16, ha='center', va='center')

# ======================
# 保存
# ======================
plt.savefig(
    'BindingDB_cluster_radar.png',
    dpi=600,
    bbox_inches='tight',
    pad_inches=0.05
)

# plt.savefig(r'D:\STUDY\npjDM\画图\BindingDB_cluster.pdf', dpi=300, bbox_inches='tight')
# plt.savefig(r'D:\STUDY\npjDM\画图\BindingDB_cluster.png', dpi=300, bbox_inches='tight')