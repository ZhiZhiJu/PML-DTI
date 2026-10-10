import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

metrics = ['AUROC', 'ACC', 'Specificity', 'Sensitivity', 'AUPRC']

# ======================
# ✅ DrugBank 数据（已重排）
# [AUROC, ACC, Specificity, Sensitivity, AUPRC]
# ======================
data = {
    'MolTrans':      [0.859, 0.775, 0.833, 0.718, 0.862],
    'DrugBAN':       [0.892, 0.822, 0.833, 0.811, 0.898],
    'DrugLAMP':      [0.914, 0.844, 0.853, 0.835, 0.918],
    'MMDG-DTI':      [0.905, 0.838, 0.842, 0.834, 0.907],
    'MIF-DTI':       [0.923, 0.855, 0.872, 0.838, 0.924],
    'ConfBiXtCPI':   [0.932, 0.870, 0.893, 0.848, 0.934],
    'PLM-DTI(Ours)': [0.946, 0.882, 0.883, 0.880, 0.951]
}

colors = ['#8FA3C3', '#99BADB', '#ADC9E4', '#C6D49D', '#BBBAB9', '#F6D0CD', '#F1A094']

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
ax.set_ylim(0.7, 1)

ax.set_rgrids(
    radii=[0.7, 0.8, 0.9],
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
ax.text(np.pi/2, 0.7, '0.70', fontsize=16, ha='center', va='center')
ax.text(np.pi/2, 0.8, '0.80', fontsize=16, ha='center', va='center')
ax.text(np.pi/2, 0.9, '0.90', fontsize=16, ha='center', va='center')

# ======================
# 保存
# ======================
plt.savefig(
    'DrugBank_radar.png',
    dpi=600,
    bbox_inches='tight',
    pad_inches=0.05
)

# plt.savefig(r'D:\STUDY\npjDM\画图\DrugBank.pdf', dpi=300, bbox_inches='tight')
# plt.savefig(r'D:\STUDY\npjDM\画图\DrugBank.png', dpi=300, bbox_inches='tight')