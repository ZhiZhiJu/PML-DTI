import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

metrics = ['AUROC', 'ACC', 'Specificity', 'Sensitivity', 'AUPRC']

# ======================
# ✅ Davis 数据（已按雷达图顺序重排）
# [AUROC, ACC, Specificity, Sensitivity, AUPRC]
# ======================
data = {
    'MolTrans':      [0.900, 0.802, 0.798, 0.865, 0.341],
    'DrugBAN':       [0.886, 0.809, 0.808, 0.831, 0.319],
    'DrugLAMP':      [0.891, 0.827, 0.826, 0.845, 0.366],
    'MMDG-DTI':      [0.897, 0.823, 0.822, 0.847, 0.349],
    'TAPB':          [0.933, 0.860, 0.860, 0.871, 0.440],
    'MIF-DTI':       [0.902, 0.789, 0.785, 0.876, 0.353],
    'ConfBiXtCPI':   [0.944, 0.862, 0.860, 0.892, 0.459],
    'PLM-DTI(Ours)': [0.934, 0.875, 0.875, 0.875, 0.471]
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
# ✅ 半径范围（你要的）
# ======================
ax.set_ylim(0.2, 1)

ax.set_rgrids(
    radii=[0.2, 0.4, 0.6, 0.8],
    labels=['', '', '', ''],
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
    ax.plot(ts, vs, color=colors[i], linewidth=2, label=name)
    ax.scatter(ts, vs, color=colors[i], s=20, zorder=5)

# ======================
# 半径刻度文字
# ======================
ax.text(np.pi/2, 0.2, '0.2', fontsize=16, ha='center', va='center')
ax.text(np.pi/2, 0.4, '0.4', fontsize=16, ha='center', va='center')
ax.text(np.pi/2, 0.6, '0.6', fontsize=16, ha='center', va='center')
ax.text(np.pi/2, 0.8, '0.8', fontsize=16, ha='center', va='center')

# ======================
# 指标标签（建议稍微往外放一点）
# ======================
# label_radii = {
#     'AUROC': 1.013,
#     'ACC': 1.02,
#     'Specificity': 1.03,
#     'Sensitivity': 1.03,
#     'AUPRC': 1.03
# }
#
# angle_offsets = {
#     'AUROC': 0.0,
#     'ACC': 0.0,
#     'Specificity': -0.05,
#     'Sensitivity': 0.05,
#     'AUPRC': 0.00
# }
#
# for angle, label in zip(theta, metrics):
#     r = label_radii.get(label, 1.05)
#     angle_adj = angle + angle_offsets.get(label, 0)
#
#     ax.text(
#         angle_adj,
#         r,
#         label,
#         fontsize=18,
#         ha='center',
#         va='center',
#         zorder=999
#     )
#
# plt.tight_layout()

plt.savefig(
    'Davis_radar.png',
    dpi=600,
    bbox_inches='tight',
    pad_inches=0.05
)

# plt.savefig(r'D:\STUDY\npjDM\画图\Davis.pdf', dpi=300, bbox_inches='tight')
# plt.savefig(r'D:\STUDY\npjDM\画图\Davis.png', dpi=300, bbox_inches='tight')