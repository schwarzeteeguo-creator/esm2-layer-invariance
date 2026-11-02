"""R3 figures for BIOINF-2026-2348 (third revision).

Changes from output_v2 (reviewer R2.1 and R2.m4):
  fig2_similarity_heatmap  same design as fig2_readable.py, 200-position
                           SaProt matrices instead of 50-position ones
  fig3_probing_pooling_summary  (a) leakage-free curves: mutant arm on
                           single-substitution variants (primary analysis);
                           (b) unchanged; (c) 200-position cosine matrices
  fig5_position_level      (b) mutant arm on single-substitution variants
Figures 0 (concept) and 1 (logit lens) are unchanged and are copied as is.
Output: figures/output_r3/
"""
import os, json, math, shutil

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import generate_figures_v2 as base   # rcParams, colours, pooling helper

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output_r3')
os.makedirs(OUT, exist_ok=True)
SR = base.SR
SS = os.path.join(HERE, '..', 'revision_results', 'single_site_local')
GCV = base.GCV
C, LAYERS = base.C, base.LAYERS


def curve(root, suite, arm, split, feat='combined'):
    d = json.load(open(os.path.join(root, suite, 'v4__%s__%s__%s.json' % (arm, feat, split))))
    per = {l: [] for l in range(33)}
    for v in d.values():
        for l, x in v['result'].items():
            s = x.get('spearman')
            if s is None or (isinstance(s, float) and math.isnan(s)):
                continue
            per[int(l)].append(s)
    return np.array([np.mean(per[l]) if per[l] else np.nan for l in range(33)])


full = lambda suite, arm, split: curve(SR, suite, arm, split)
single = lambda suite, arm, split: curve(SS, suite.replace('probing_v4_', 'probing_v4_single_'),
                                         arm, split)


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, '%s.%s' % (name, ext)), dpi=300)
    plt.close(fig)
    print('saved', name)


def lookup_control():
    d = json.load(open(os.path.join(GCV, 'v4__random_position__combined__random.json')))
    return np.mean([v['result']['rho_mean'] for v in d.values()])


# ── fig2: 200-position similarity matrices (design of fig2_readable.py) ──
def fig2():
    src = os.path.join(HERE, 'cka200_saprot')
    cos = np.load(os.path.join(src, 'cosine_matrix_avg.npy'))
    cka = np.load(os.path.join(src, 'cka_matrix_avg.npy'))
    panels = [(cos, 'Cosine', [0.8, 0.9]), (cka, 'Linear CKA', [0.7, 0.8, 0.9])]
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.8))
    for ax, (m, title, levels) in zip(axes, panels):
        cmap = plt.get_cmap('Blues').copy()
        cmap.set_under('white')
        im = ax.imshow(m, cmap=cmap, vmin=0.5, vmax=1.0, interpolation='nearest',
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
        cb = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02,
                          extend='min' if m.min() < 0.5 else 'neither')
        cb.ax.tick_params(labelsize=6)
        cb.outline.set_linewidth(0.5)
    for ax, lab in zip(axes, 'ab'):
        ax.text(-0.22, 1.06, lab, transform=ax.transAxes, fontsize=9,
                fontweight='bold', va='bottom')
    fig.tight_layout()
    save(fig, 'fig2_similarity_heatmap')
    return {'cos_L0_L32': float(cos[0, 32]), 'cka_L0_L32': float(cka[0, 32]),
            'cka_min': float(cka.min()), 'n_cka_below_0.5': int((cka < 0.5).sum()),
            'n_cos_below_0.5': int((cos < 0.5).sum())}


# ── fig3 ────────────────────────────────────────────────────────────
def fig3():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4))
    ax = axes[0]
    yr_e = full('probing_v4_esm2', 'masked_wt', 'random')
    yr_s = full('probing_v4_saprot_3di', 'masked_wt', 'random')
    ym_e = single('probing_v4_esm2', 'mutant', 'modulo')
    ym_s = single('probing_v4_saprot_3di', 'mutant', 'modulo')
    ax.plot(LAYERS, yr_e, color=C['orange'], ls='-')
    ax.plot(LAYERS, yr_s, color=C['blue'], ls='-')
    ax.plot(LAYERS, ym_e, color=C['orange'], ls='--')
    ax.plot(LAYERS, ym_s, color=C['blue'], ls='--')
    ctl = lookup_control()
    ax.axhline(ctl, color=C['gray'], ls=':', lw=1.0)
    ax.set_xlabel('Layer')
    ax.set_ylabel(r'Spearman $\rho$')
    ax.set_xlim(0, 47)
    ax.set_xticks([0, 8, 16, 24, 32])
    be, bs = int(np.nanargmax(ym_e)), int(np.nanargmax(ym_s))
    ax.annotate('random, masked\n(both models)', xy=(32.6, (yr_e[32] + yr_s[32]) / 2 + 0.030),
                fontsize=5.0, va='center', ha='left', color=C['gray'])
    ax.annotate('position lookup\ncontrol (%.2f)' % ctl, xy=(32.6, ctl - 0.032),
                fontsize=5.0, va='center', ha='left', color=C['gray'])
    ax.annotate('SaProt modulo,\nmutant, single\n(%.2f, L%d)' % (ym_s[bs], bs),
                xy=(32.6, ym_s[32] + 0.012), fontsize=5.0, va='center', ha='left', color=C['blue'])
    ax.annotate('ESM-2 modulo,\nmutant, single\n(%.2f, L%d)' % (ym_e[be], be),
                xy=(32.6, ym_e[32] - 0.040), fontsize=5.0, va='center', ha='left', color=C['orange'])
    ax.text(-0.28, 1.06, 'a', transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom')

    ax = axes[1]
    me, _, _ = base.pooling_matrix('pooling_v3_esm2')
    ms, _, _ = base.pooling_matrix('pooling_v3_saprot')
    xpos = np.arange(4)
    for i, (m, col, model) in enumerate([(me, C['orange'], 'ESM-2'), (ms, C['blue'], 'SaProt')]):
        off = (i - 0.5) * 0.19
        ax.scatter(xpos + off + np.random.RandomState(0).uniform(-0.055, 0.055, 4),
                   m.mean(axis=0), s=10, color=col, alpha=0.55, lw=0, zorder=2)
        ax.plot(xpos + off, m.mean(axis=0), color=col, lw=1.0, label=model)
    ax.set_xticks(xpos)
    ax.set_xticklabels(['Ridge', 'RF', 'MLP', 'LightGBM'])
    ax.set_ylim(0.37, 0.57)
    ax.set_ylabel(r'Spearman $\rho$ (mutant, modulo)')
    ax.legend(loc='upper right', fontsize=6.0, borderaxespad=0.3)
    ax.text(-0.28, 1.06, 'b', transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom')

    ax = axes[2]
    ends = {}
    for mkey, col, model in [('cka200_esm2', C['orange'], 'ESM-2'),
                             ('cka200_saprot', C['blue'], 'SaProt')]:
        m = np.load(os.path.join(HERE, mkey, 'cosine_matrix_avg.npy'))
        ds, vs = [], []
        for dd in range(1, 33):
            v = np.array([m[i, i + dd] for i in range(33 - dd)])
            ds.append(np.repeat(dd, len(v)))
            vs.append(v)
        ds, vs = np.concatenate(ds), np.concatenate(vs)
        jitter = np.random.RandomState(1).uniform(-0.35, 0.35, len(ds))
        ax.scatter(ds + jitter, vs, s=1.5, color=col, alpha=0.15, lw=0, rasterized=True)
        means = [np.mean(vs[ds == dd]) for dd in range(1, 33)]
        ends[model] = means[-1]
        ax.plot(range(1, 33), means, color=col, lw=1.2)
        ax.annotate('%s (%.2f)' % (model, means[-1]),
                    xy=(33.4, means[-1] + (0.05 if model == 'ESM-2' else -0.07)),
                    fontsize=5.4, va='center', ha='left', color=col)
    ax.axhline(0, color=C['light_gray'], lw=0.6)
    ax.set_xlabel('Layer distance $d$')
    ax.set_ylabel('Cosine similarity')
    ax.set_xlim(0, 42)
    ax.set_xticks([0, 8, 16, 24, 32])
    ax.text(-0.28, 1.06, 'c', transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom')
    fig.tight_layout(w_pad=1.8)
    save(fig, 'fig3_probing_pooling_summary')
    return {'esm2_single_mod_best': [be, float(ym_e[be])], 'saprot_single_mod_best': [bs, float(ym_s[bs])],
            'cos_end': ends, 'pool_mean_esm2': me.mean(axis=0).tolist(),
            'pool_mean_saprot': ms.mean(axis=0).tolist()}


# ── fig5 ────────────────────────────────────────────────────────────
def fig5():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
    ax = axes[0]
    ax.plot(LAYERS, full('probing_v4_saprot_3di', 'masked_wt', 'random'),
            color=C['blue'], label='SaProt (3Di), masked, random')
    ax.plot(LAYERS, full('probing_v4_esm2', 'masked_wt', 'random'),
            color=C['orange'], label='ESM-2, masked, random')
    ax.axhline(lookup_control(), color=C['gray'], ls=':', lw=1.0, label='Position lookup control')
    ax.set_ylim(0.45, 0.68)
    ax.set_xlabel('Layer')
    ax.set_ylabel(r'Spearman $\rho$')
    ax.set_title(r'Random split: flat, $\sigma \approx 10^{-3}$', fontsize=7)
    ax.legend(loc='lower right', fontsize=5.8)
    ax.text(-0.22, 1.06, 'a', transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom')

    ax = axes[1]
    info = {}
    for split, ls in [('modulo', '-'), ('position_groupkfold', '--'), ('contiguous', ':')]:
        for suite, col, lab in [('probing_v4_saprot_3di', C['blue'], 'SaProt (3Di)'),
                                ('probing_v4_esm2', C['orange'], 'ESM-2')]:
            y = single(suite, 'mutant', split)
            info['%s|%s' % (lab, split)] = [int(np.nanargmax(y)), float(np.nanmax(y)), float(y[32])]
            ax.plot(LAYERS, y, color=col, ls=ls, label='%s, %s' % (lab, split.replace('_', ' ')))
    ax.set_xlabel('Layer')
    ax.set_ylabel(r'Spearman $\rho$')
    ax.set_title('Leakage-free (mutant arm, single substitutions): best at L30--31', fontsize=7)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.30), ncol=3, fontsize=5.2,
              handlelength=1.4, columnspacing=0.8, labelspacing=0.25, borderaxespad=0.2)
    ax.text(-0.22, 1.06, 'b', transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom')
    fig.tight_layout(w_pad=1.6)
    save(fig, 'fig5_position_level')
    return info


if __name__ == '__main__':
    v2 = os.path.join(HERE, 'output_v2')
    for f in ('fig0_concept.pdf', 'fig1_logit_lens.png'):
        shutil.copy2(os.path.join(v2, f), os.path.join(OUT, f))
    info = {'fig2': fig2(), 'fig3': fig3(), 'fig5': fig5()}
    json.dump(info, open(os.path.join(OUT, 'figure_values_r3.json'), 'w'), indent=1)
    print(json.dumps(info, indent=1))
