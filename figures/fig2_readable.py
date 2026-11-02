"""Figure 2 redrawn for readability (same data, size, and layout as v2).

v2 used a diverging RdBu map on [-1, 1]; since almost all similarities lie in
[0.5, 1], every cell rendered as near-identical dark red. Similarity is a
magnitude, so this version uses a single-hue sequential map (light = low,
dark = high) fitted to each panel's range, with labelled contour lines.
Data: the 50-position SaProt matrices shown in the v2 manuscript.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'revision_results', 'server_results', 'cka_saprot')
OUT = os.path.join(HERE, 'output_v2')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'pdf.fonttype': 42,
})

cos = np.load(os.path.join(SRC, 'cosine_matrix_avg.npy'))
cka = np.load(os.path.join(SRC, 'cka_matrix_avg.npy'))

PANELS = [
    (cos, 'Cosine', (0.5, 1.0), [0.8, 0.9]),
    (cka, 'Linear CKA', (0.5, 1.0), [0.7, 0.8, 0.9]),
]

fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.8))
for ax, (m, title, (lo, hi), levels) in zip(axes, PANELS):
    cmap = plt.get_cmap('Blues').copy()
    cmap.set_under('white')
    im = ax.imshow(m, cmap=cmap, vmin=lo, vmax=hi, interpolation='nearest',
                   rasterized=True)
    cs = ax.contour(m, levels=levels, colors='#333333', linewidths=0.5)
    ax.clabel(cs, fmt='%.1f', fontsize=5.5, inline=True, inline_spacing=2)
    ax.set_title(title + r'  (L0$\leftrightarrow$L32 = %.2f)' % m[0, 32], fontsize=7)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Layer')
    ax.set_xticks([0, 8, 16, 24, 32])
    ax.set_yticks([0, 8, 16, 24, 32])
    for s in ax.spines.values():
        s.set_visible(True)
    below = m.min() < lo
    cb = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02,
                      extend='min' if below else 'neither')
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.5)
for ax, lab in zip(axes, 'ab'):
    ax.text(-0.22, 1.06, lab, transform=ax.transAxes, fontsize=9,
            fontweight='bold', va='bottom')
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, 'fig2_similarity_heatmap.%s' % ext))
print('saved')
