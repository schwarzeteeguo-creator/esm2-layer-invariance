"""Revision figures (v2) for BIOINF-2026-2348.

Regenerates the data figures from the unified revision results:
  fig1_logit_lens       validated fp32 lens curves (ESM-2, SaProt real-3Di), n=63
  fig2_similarity_heatmap  SaProt cosine + linear CKA 33x33 matrices
  fig3_probing_pooling_summary  (a) probing curves random vs leakage-free + control
                                (b) pooling bands (mutant@modulo, both models)
                                (c) cosine vs layer distance (unique pairs)
  fig5_position_level   (a) random split flat profile + lookup control
                        (b) leakage-free splits lower level, late-layer best
Consolidates the submitted version's separate probing/pooling and summary
figures into one three-panel figure (reviewer m12).
"""
import os, sys, json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SKILL_SCRIPTS = r'C:\Users\19700\.zcode\skills\nature-figure\scripts'
if os.path.isdir(SKILL_SCRIPTS) and SKILL_SCRIPTS not in sys.path:
    sys.path.insert(0, SKILL_SCRIPTS)
try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:
    require_matplotlib_panel_alignment = None

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output')
os.makedirs(OUT, exist_ok=True)

SR = r'G:\蛋白质课题\revision_results\server_results'
GCV = r'G:\蛋白质课题\revision_results\probing_v4_gcv'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.5,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'lines.linewidth': 1.2, 'lines.markersize': 3,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'svg.fonttype': 'none',
})

C = dict(blue='#3182BD', dark_blue='#08519C', light_blue='#6BAED6',
         orange='#E6550D', green='#31A354', red='#DE2D26', purple='#756BB1',
         gray='#636363', light_gray='#BDBDBD')
LAYERS = np.arange(33)


def layer_curve(suite, arm, split, feat='combined'):
    path = os.path.join(SR, suite, 'v4__%s__%s__%s.json' % (arm, feat, split))
    d = json.load(open(path))
    per = {l: [] for l in range(33)}
    for v in d.values():
        for l, x in v['result'].items():
            s = x.get('spearman')
            if s is None or (isinstance(s, float) and math.isnan(s)):
                continue
            per[int(l)].append(s)
    return np.array([np.mean(per[l]) if per[l] else np.nan for l in range(33)])


def gate(fig, name, axes, row, exclude=()):
    """Render-time alignment gate with explicit comparable row group."""
    if require_matplotlib_panel_alignment is None:
        return
    ids = {a: lbl for a, lbl in zip(axes, [chr(97 + i) for i in range(len(axes))])}
    require_matplotlib_panel_alignment(
        fig, panel_ids=ids, row_groups=[row], exclude_axes=list(exclude),
        json_out=os.path.join(OUT, '%s.alignment.json' % name),
        overlay_svg=os.path.join(OUT, '%s.alignment.svg' % name),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)


def save(fig, name):
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(os.path.join(OUT, '%s.%s' % (name, ext)), dpi=300)
    plt.close(fig)
    print('saved', name)


# ── fig1: logit lens ────────────────────────────────────────────────
def fig1():
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    for key, model, col in [('logit_lens_v2_esm2', 'ESM-2', C['orange']),
                            ('logit_lens_v2_saprot_3di', 'SaProt (real 3Di)', C['blue'])]:
        s = json.load(open(os.path.join(SR, key, 'logit_lens_v2_summary.json')))
        pl = s['per_layer']
        mean = np.array([pl[str(l)]['mean'] for l in range(33)])
        std = np.array([pl[str(l)]['std'] for l in range(33)])
        ax.plot(LAYERS, mean, color=col, label=model)
        ax.fill_between(LAYERS, mean - std, mean + std, color=col, alpha=0.15, lw=0)
    ax.axhline(0, color=C['light_gray'], lw=0.6, ls='-')
    ax.set_xlabel('Layer')
    ax.set_ylabel(r'Zero-shot Spearman $\rho$')
    ax.set_xlim(0, 32)
    ax.legend(loc='upper left')
    ax.text(0.02, 0.02, '$n=63$, fp32, gate-passed', transform=ax.transAxes,
            fontsize=6, color=C['gray'])
    save(fig, 'fig1_logit_lens')


# ── fig2: similarity heatmaps (cosine + CKA, SaProt) ───────────────
def fig2():
    cos = np.load(os.path.join(SR, 'cka_saprot', 'cosine_matrix_avg.npy'))
    cka = np.load(os.path.join(SR, 'cka_saprot', 'cka_matrix_avg.npy'))
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.8))
    cbs = []
    for ax, m, title in [(axes[0], cos, 'Cosine'),
                         (axes[1], cka, 'Linear CKA')]:
        vmax = max(abs(np.nanmin(m)), abs(np.nanmax(m)))
        im = ax.imshow(m, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       rasterized=True, interpolation='nearest')
        ax.set_title(title + r'  (L0$\leftrightarrow$L32 = %.2f)' % (m[0, 32]), fontsize=7)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Layer')
        ax.set_xticks([0, 8, 16, 24, 32])
        ax.set_yticks([0, 8, 16, 24, 32])
        cb = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cb.ax.tick_params(labelsize=6)
        cbs.append(cb.ax)
    axes[0].text(-0.22, 1.06, 'a', transform=axes[0].transAxes,
                 fontsize=9, fontweight='bold', va='bottom')
    axes[1].text(-0.22, 1.06, 'b', transform=axes[1].transAxes,
                 fontsize=9, fontweight='bold', va='bottom')
    fig.tight_layout()
    gate(fig, 'fig2_similarity_heatmap', list(axes), ['a', 'b'], exclude=cbs)
    save(fig, 'fig2_similarity_heatmap')


# ── fig3: probing + pooling + distance (consolidated) ───────────────
def pooling_matrix(suite):
    d = json.load(open(os.path.join(SR, suite,
        'pooling_v3__%s.json' % ('esm2_650m' if 'esm2' in suite else 'saprot_650m'))))
    strats = ['last_layer', 'mean_20_33', 'concat_5L', 'attention']
    ros = ['ridge', 'rf', 'mlp', 'lightgbm']
    m = np.full((4, 4), np.nan)
    for i, st in enumerate(strats):
        for j, ro in enumerate(ros):
            vals = []
            for ds_e in d.values():
                cell = ds_e['arms']['mutant']['modulo'].get('%s__combined' % st, {})
                if 'rho' in cell:
                    cell = cell['rho']
                v = cell.get(ro, {}).get('spearman')
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    vals.append(v)
            m[i, j] = np.mean(vals)
    return m, strats, ros


def fig3():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4))

    # (a) probing curves
    ax = axes[0]
    ax.plot(LAYERS, layer_curve('probing_v4_esm2', 'masked_wt', 'random'),
            color=C['orange'], ls='-', label='ESM-2, random (masked)')
    ax.plot(LAYERS, layer_curve('probing_v4_saprot_3di', 'masked_wt', 'random'),
            color=C['blue'], ls='-', label='SaProt, random (masked)')
    ax.plot(LAYERS, layer_curve('probing_v4_esm2', 'mutant', 'modulo'),
            color=C['orange'], ls='--', label='ESM-2, modulo (mutant)')
    ax.plot(LAYERS, layer_curve('probing_v4_saprot_3di', 'mutant', 'modulo'),
            color=C['blue'], ls='--', label='SaProt, modulo (mutant)')
    d = json.load(open(os.path.join(GCV, 'v4__random_position__combined__random.json')))
    ctl = np.mean([v['result']['rho_mean'] for v in d.values()])
    ax.axhline(ctl, color=C['gray'], ls=':', lw=1.0,
               label='Position lookup control (%.2f)' % ctl)
    ax.set_xlabel('Layer')
    ax.set_ylabel(r'Spearman $\rho$')
    ax.set_xlim(0, 47)
    ax.set_xticks([0, 8, 16, 24, 32])
    # direct labels at curve ends (no legend: five series do not fit inside)
    yrand_e = layer_curve('probing_v4_esm2', 'masked_wt', 'random')[32]
    yrand_s = layer_curve('probing_v4_saprot_3di', 'masked_wt', 'random')[32]
    ymod_e = layer_curve('probing_v4_esm2', 'mutant', 'modulo')[31]
    ymod_s = layer_curve('probing_v4_saprot_3di', 'mutant', 'modulo')[31]
    ax.annotate('random, masked\n(both models)',
                xy=(32.6, (yrand_e + yrand_s) / 2 + 0.030),
                fontsize=5.0, va='center', ha='left', color=C['gray'])
    ax.annotate('position lookup\ncontrol (%.2f)' % ctl, xy=(32.6, ctl - 0.032),
                fontsize=5.0, va='center', ha='left', color=C['gray'])
    ax.annotate('SaProt modulo,\nmutant (%.2f)' % ymod_s, xy=(32.6, ymod_s - 0.006),
                fontsize=5.0, va='center', ha='left', color=C['blue'])
    ax.annotate('ESM-2 modulo,\nmutant (%.2f)' % ymod_e, xy=(32.6, ymod_e),
                fontsize=5.0, va='center', ha='left', color=C['orange'])
    ax.text(-0.28, 1.06, 'a', transform=ax.transAxes, fontsize=9,
            fontweight='bold', va='bottom')

    # (b) pooling bands
    ax = axes[1]
    me, strats, ros = pooling_matrix('pooling_v3_esm2')
    ms, _, _ = pooling_matrix('pooling_v3_saprot')
    xpos = np.arange(4)
    for i, (m, col, model) in enumerate([(me, C['orange'], 'ESM-2'),
                                         (ms, C['blue'], 'SaProt')]):
        off = (i - 0.5) * 0.19
        ax.scatter(xpos + off + np.random.RandomState(0).uniform(-0.055, 0.055, 4),
                   m.mean(axis=0), s=10, color=col, alpha=0.55, lw=0, zorder=2)
        ax.plot(xpos + off, m.mean(axis=0), color=col, lw=1.0, label=model)
    ax.set_xticks(xpos)
    ax.set_xticklabels(['Ridge', 'RF', 'MLP', 'LightGBM'])
    ax.set_ylim(0.37, 0.57)
    ax.set_ylabel(r'Spearman $\rho$ (mutant, modulo)')
    ax.legend(loc='upper right', fontsize=6.0, borderaxespad=0.3)
    ax.text(-0.28, 1.06, 'b', transform=ax.transAxes, fontsize=9,
            fontweight='bold', va='bottom')

    # (c) cosine vs distance (direct labels at line ends, no legend)
    ax = axes[2]
    for mkey, col, model in [('cka_esm2', C['orange'], 'ESM-2'),
                             ('cka_saprot', C['blue'], 'SaProt')]:
        m = np.load(os.path.join(SR, mkey, 'cosine_matrix_avg.npy'))
        ds, vs = [], []
        for dd in range(1, 33):
            v = np.array([m[i, i + dd] for i in range(33 - dd)])
            ds.append(np.repeat(dd, len(v)))
            vs.append(v)
        ds, vs = np.concatenate(ds), np.concatenate(vs)
        jitter = np.random.RandomState(1).uniform(-0.35, 0.35, len(ds))
        ax.scatter(ds + jitter, vs, s=1.5, color=col, alpha=0.15, lw=0,
                   rasterized=True)
        means = [np.mean(vs[ds == dd]) for dd in range(1, 33)]
        ax.plot(range(1, 33), means, color=col, lw=1.2)
        ax.annotate('%s (%.2f)' % (model, means[-1]),
                    xy=(33.4, means[-1] + (0.05 if model == 'ESM-2' else -0.07)),
                    fontsize=5.4, va='center', ha='left', color=col)
    ax.axhline(0, color=C['light_gray'], lw=0.6)
    ax.set_xlabel('Layer distance $d$')
    ax.set_ylabel('Cosine similarity')
    ax.set_xlim(0, 42)
    ax.set_xticks([0, 8, 16, 24, 32])
    ax.text(-0.28, 1.06, 'c', transform=ax.transAxes, fontsize=9,
            fontweight='bold', va='bottom')

    fig.tight_layout(w_pad=1.8)
    gate(fig, 'fig3_probing_pooling_summary', list(axes), ['a', 'b', 'c'])
    save(fig, 'fig3_probing_pooling_summary')


# ── fig5: random vs leakage-free ────────────────────────────────────
def fig5():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))

    ax = axes[0]
    ax.plot(LAYERS, layer_curve('probing_v4_saprot_3di', 'masked_wt', 'random'),
            color=C['blue'], label='SaProt (3Di), masked, random')
    ax.plot(LAYERS, layer_curve('probing_v4_esm2', 'masked_wt', 'random'),
            color=C['orange'], label='ESM-2, masked, random')
    d = json.load(open(os.path.join(GCV, 'v4__random_position__combined__random.json')))
    ctl = np.mean([v['result']['rho_mean'] for v in d.values()])
    ax.axhline(ctl, color=C['gray'], ls=':', lw=1.0,
               label='Position lookup control')
    ax.set_ylim(0.45, 0.68)
    ax.set_xlabel('Layer')
    ax.set_ylabel(r'Spearman $\rho$')
    ax.set_title(r'Random split: flat, $\sigma \approx 10^{-3}$', fontsize=7)
    ax.legend(loc='lower right', fontsize=5.8)
    ax.text(-0.22, 1.06, 'a', transform=ax.transAxes, fontsize=9,
            fontweight='bold', va='bottom')

    ax = axes[1]
    styles = [('modulo', '-'), ('position_groupkfold', '--'), ('contiguous', ':')]
    for split, ls in styles:
        ax.plot(LAYERS, layer_curve('probing_v4_saprot_3di', 'mutant', split),
                color=C['blue'], ls=ls, label='SaProt (3Di), %s' % split.replace('_',' '))
        ax.plot(LAYERS, layer_curve('probing_v4_esm2', 'mutant', split),
                color=C['orange'], ls=ls, label='ESM-2, %s' % split.replace('_',' '))
    ax.set_xlabel('Layer')
    ax.set_ylabel(r'Spearman $\rho$')
    ax.set_title(r'Leakage-free: lower level, best at L30--32', fontsize=7)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.30), ncol=3,
              fontsize=5.2, handlelength=1.4, columnspacing=0.8,
              labelspacing=0.25, borderaxespad=0.2)
    ax.text(-0.22, 1.06, 'b', transform=ax.transAxes, fontsize=9,
            fontweight='bold', va='bottom')

    fig.tight_layout(w_pad=1.6)
    gate(fig, 'fig5_position_level', list(axes), ['a', 'b'])
    save(fig, 'fig5_position_level')


if __name__ == '__main__':
    fig1()
    fig2()
    fig3()
    fig5()
    print('all done ->', OUT)
