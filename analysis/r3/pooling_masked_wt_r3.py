"""R3 addition: the pooling benchmark on the masked_wt arm (already computed in
pooling_v3, previously unreported). Same 20 datasets, strategies, readouts and
folds as the mutant-arm benchmark (main Table 3).

Outputs revision_results/r3_analysis/pooling_masked_wt_r3.json with
  means[model][split][strategy__readout]
  fusion_vs_last[model|split][strategy|readout]   paired, bootstrap 90% CI
  wt_vs_mutant[model|split][readout]               last layer, masked_wt minus
                                                   mutant, excluding the two
                                                   multi-site datasets
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'revision_docs'))
import analysis_r2 as A

STRATS = ['last_layer', 'mean_20_33', 'concat_5L', 'attention']
READS = ['ridge', 'rf', 'mlp', 'lightgbm']
MULTI = {'A4_HUMAN_Seuma_2021', 'DLG4_HUMAN_Faure_2021'}
out = {'means': {}, 'fusion_vs_last': {}, 'wt_vs_mutant': {}}
for model, sub in (('ESM-2', 'pooling_v3_esm2'), ('SaProt', 'pooling_v3_saprot')):
    f = next((A.SR / sub).glob('pooling_v3__*.json'))
    wt = A.pooling_rhos(f, arm='masked_wt')
    mu = A.pooling_rhos(f, arm='mutant')
    for split in ('random', 'modulo', 'contiguous', 'position_groupkfold'):
        for s in STRATS:
            for r in READS:
                v = [d[(split, s, r)] for d in wt.values() if (split, s, r) in d]
                if v:
                    out['means'].setdefault(model, {}).setdefault(split, {})['%s__%s' % (s, r)] = {
                        'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)), 'n': len(v)}
    for split in ('random', 'modulo', 'contiguous'):
        t = A.paired_diff_table(wt, split, 'last_layer', STRATS[1:], READS)
        out['fusion_vs_last']['%s|%s' % (model, split)] = {
            '%s|%s' % k: {kk: vv for kk, vv in v.items()} for k, v in t.items()}
        cmp = {}
        for r in READS:
            pairs = [wt[n][(split, 'last_layer', r)] - mu[n][(split, 'last_layer', r)]
                     for n in wt if n in mu and n not in MULTI
                     and (split, 'last_layer', r) in wt[n] and (split, 'last_layer', r) in mu[n]]
            d = np.array(pairs)
            cmp[r] = {'n': len(d), 'mean_delta': float(d.mean()), 'boot_ci90': A.boot_ci90(d),
                      'wilcoxon_p': float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else None}
        out['wt_vs_mutant']['%s|%s' % (model, split)] = cmp

dst = ROOT / 'revision_results/r3_analysis/pooling_masked_wt_r3.json'
json.dump(out, open(dst, 'w'), indent=1)

if __name__ == '__main__':
    for key, t in out['fusion_vs_last'].items():
        better = [k for k, v in t.items() if v['boot_ci90'][0] > 0]
        worse = [k for k, v in t.items() if v['boot_ci90'][1] < 0]
        big = max(t.items(), key=lambda kv: kv[1]['mean_delta'])
        print('%-18s n=%d  better %d  worse %d  | largest %s %+.3f [%+.3f, %+.3f]' % (
            key, next(iter(t.values()))['n'], len(better), len(worse), big[0], big[1]['mean_delta'],
            *big[1]['boot_ci90']))
        if better:
            print('    better:', ', '.join('%s %+.3f' % (k, t[k]['mean_delta']) for k in better))
    for model in ('ESM-2', 'SaProt'):
        for split in ('random', 'modulo'):
            m = out['means'][model][split]
            print('%s %s last-layer means: %s' % (model, split, ' '.join(
                '%s=%.3f' % (r, m['last_layer__%s' % r]['mean']) for r in READS)))
    for key, c in out['wt_vs_mutant'].items():
        print('wt-minus-mutant %-18s %s' % (key, ' '.join(
            '%s %+.3f[%+.3f,%+.3f]' % (r, v['mean_delta'], *v['boot_ci90']) for r, v in c.items())))
