import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

metrics = ['AUROC', 'ACC', 'Specificity', 'Sensitivity', 'AUPRC']

# ======================
# 顺序：[AUROC, ACC, Specificity, Sensitivity, AUPRC]
# ======================
data = {
    'MolTrans':      [0.938, 0.868, 0.862, 0.877, 0.913],
    'DrugBAN':       [0.963, 0.908, 0.909, 0.906, 0.953],
    'DrugLAMP':      [0.960, 0.904, 0.907, 0.902, 0.951],
    'MMDG-DTI':      [0.962, 0.906, 0.909, 0.903, 0.952],
    'TAPB':          [0.961, 0.901, 0.902, 0.899, 0.950],
    'MIF-DTI':       [0.956, 0.901, 0.904, 0.897, 0.943],
    'ConfBiXtCPI':   [0.965, 0.905, 0.898, 0.915, 0.954],
    'PLM-DTI(Ours)': [0.970, 0.920, 0.921, 0.917, 0.959]
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
# ✅ 半径范围修改
# ======================
ax.set_ylim(0.80, 1)

# 网格
ax.set_rgrids(radii=[0.80, 0.85, 0.90, 0.95], labels=['', '', '', ''], angle=90, fontsize=18)
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
ax.text(np.pi/2, 0.80, '0.80', fontsize=18, ha='center', va='center', zorder=999)
ax.text(np.pi/2, 0.85, '0.85', fontsize=18, ha='center', va='center', zorder=999)
ax.text(np.pi/2, 0.90, '0.90', fontsize=18, ha='center', va='center', zorder=999)
ax.text(np.pi/2, 0.95, '0.95', fontsize=18, ha='center', va='center', zorder=999)

# ======================
# 指标标签
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
# # plt.tight_layout()
#
plt.savefig(
    'BindingDB_radar.png',
    dpi=600,
    bbox_inches='tight',
    pad_inches=0.05
)

# plt.savefig(r'D:\STUDY\npjDM\画图\BindingDB.pdf', dpi=300, bbox_inches='tight')
# plt.savefig(r'D:\STUDY\npjDM\画图\BindingDB.png', dpi=300, bbox_inches='tight')