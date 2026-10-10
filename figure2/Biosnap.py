import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

metrics = ['AUROC', 'ACC', 'Specificity', 'Sensitivity', 'AUPRC']

data = {
    'MolTrans':      [0.893, 0.819, 0.839, 0.798, 0.900],
    'DrugBAN':       [0.903, 0.841, 0.868, 0.810, 0.907],
    'DrugLAMP':      [0.917, 0.850, 0.855, 0.844, 0.922],
    'MMDG-DTI':      [0.916, 0.848, 0.851, 0.845, 0.922],
    'TAPB':          [0.943, 0.881, 0.887, 0.874, 0.944],
    'MIF-DTI':       [0.935, 0.867, 0.873, 0.861, 0.936],
    'ConfBiXtCPI':   [0.939, 0.868, 0.880, 0.856, 0.944],
    'PML-DTI(Ours)': [0.949, 0.889, 0.914, 0.864, 0.953]
}

colors = ['#8FA3C3', '#99BADB', '#ADC9E4', '#C6D49D', '#F6BF94', '#BBBAB9', '#F6D0CD', '#F1A094']

N = len(metrics)
theta = np.linspace(0, 2 * np.pi, N, endpoint=False)

fig, ax = plt.subplots(figsize=(7,7), subplot_kw={'projection':'polar'})

ax.set_theta_offset(np.pi/2)
ax.set_theta_direction(-1)
ax.set_xticks(theta)
ax.set_xticklabels([])  # ✅ 完全关闭默认标签
ax.set_ylim(0.7, 1)

# 网格
ax.set_rgrids(radii=[0.7, 0.8, 0.9], labels=['', '', ''], angle=90, fontsize=18)
ax.xaxis.grid(True, linestyle='-', linewidth=2, color='#D9D9DA', alpha=0.8)
ax.yaxis.grid(True, linestyle=(0, (10, 6)), linewidth=2, color='#D9D9DA', alpha=0.8)
ax.spines['polar'].set_linewidth(1.5)

# 画线
for i, (name, values) in enumerate(data.items()):
    vs = values + [values[0]]
    ts = list(theta) + [theta[0]]
    ax.plot(ts, vs, color=colors[i], linewidth=2, label=name)
    ax.scatter(ts, vs, color=colors[i], s=20, zorder=5)

# ======================
# ✅ 半径刻度文字
# ======================
ax.text(np.pi/2, 0.7, '0.70', fontsize=18, ha='center', va='center', zorder=999)
ax.text(np.pi/2, 0.8, '0.80', fontsize=18, ha='center', va='center', zorder=999)
ax.text(np.pi/2, 0.9, '0.90', fontsize=18, ha='center', va='center', zorder=999)

# ======================
# ✅ 每个指标独立控制（核心）
# ======================
label_radii = {
    'AUROC': 1.02,
    'ACC': 1.028,
    'Specificity': 1.045,
    'Sensitivity': 1.045,
    'AUPRC': 1.045
}

angle_offsets = {
    'AUROC': 0.0,
    'ACC': 0.0,
    'Specificity': -0.05,
    'Sensitivity': 0.05,
    'AUPRC': 0.00
}

for angle, label in zip(theta, metrics):
    r = label_radii.get(label, 1.05)
    angle_adj = angle + angle_offsets.get(label, 0)

    ax.text(
        angle_adj,
        r,
        label,
        fontsize=18,
        ha='center',
        va='center',
        zorder=999
    )
ax.legend(loc='upper left', bbox_to_anchor=(2, 2), fontsize=10)

# ======================
# legend（论文关键）
# ======================
# ax.legend(
#     loc='upper center',
#     bbox_to_anchor=(0.5, -0.08),
#     ncol=3,
#     fontsize=12,
#     frameon=False
# )
#
# ax.set_axisbelow(True)

# ======================
# 导出（核心！！！）
# ======================
# plt.tight_layout()

# plt.savefig(
#     'Biosnap_radar.pdf',   # ✅ 投稿用
#     dpi=600,
#     bbox_inches='tight',
#     pad_inches=0.05
# )

plt.savefig(
    'Biosnap_radar.png',   # ✅ 备用
    dpi=600,
    bbox_inches='tight',
    pad_inches=0.05
)

# plt.savefig(r'D:\STUDY\npjDM\画图\Biosnap.pdf', dpi=300, bbox_inches='tight')
# plt.savefig(r'D:\STUDY\npjDM\画图\Biosnap.png', dpi=300, bbox_inches='tight')