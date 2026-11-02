"""Automated R3 consistency check (main text + SI against result files).

1. Supplementary tables are first cited in order S1..S11 (R2.m2).
2. Stale R2 values that R3 replaced no longer appear in the main text.
3. Key numbers in the main text and SI match revision_results/r3_analysis.
4. Every reviewer comment has at least one change anchored in the R3 source.
Exit code 1 if any check fails.
"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ms = (ROOT / 'manuscript_revised_R3_clean.tex').read_text(encoding='utf-8')
red = (ROOT / 'manuscript_revised_R3.tex').read_text(encoding='utf-8')
si = (ROOT / 'supplementary_R3.tex').read_text(encoding='utf-8')
N = json.load(open(ROOT / 'revision_results/r3_analysis/r3_numbers.json'))
MS = json.load(open(ROOT / 'revision_results/r3_analysis/multisite_share_r3.json'))['summary']
XS = json.load(open(ROOT / 'revision_results/r3_analysis/single_site_excl_small_r3.json'))['summary']
results = []


def check(name, ok, detail=''):
    results.append((name, bool(ok), detail))


body = ms[ms.index(r'\begin{abstract}'):ms.index(r'\begin{thebibliography}')]

# ---- 1. citation order
cites = []
for m in re.finditer(r'Supplementary\s+Tables?\s+(?:\\rthree\{|\\rtwo\{)?S(\d+)(?:--S(\d+))?', body):
    a = int(m.group(1))
    b = int(m.group(2)) if m.group(2) else a
    cites.extend(range(a, b + 1))
first = []
for c in cites:
    if c not in first:
        first.append(c)
check('SI first-citation order is S1..S11', first == list(range(1, 12)), str(first))
check('SI defines tables S1..S11', all((r'\label{tab:S%d}' % i) in si for i in range(1, 12)))

# ---- 2. stale values
stale = {
    r'within $0.02$ on single substitutions': 'old single-substitution wording',
    r'0.027--0.53': '', r'0.27--0.53': 'old leakage-free range',
    r'3.8\%': 'old multi-site variant share', r'8.4\%': 'old multi-site row share',
    r'leaving 20': 'old number of equivalence tests', r'8 were equivalent': 'old count',
    r'CKA 0.81': 'old 50-position CKA', r'0.808': 'old 50-position CKA',
    r'CKA 0.554': 'old ESM-2 CKA', r'0.15$--$0.20': 'old inflation range',
    r'up to 0.53': 'old supervised maximum', r'$+0.06$ to $+0.12$': 'old 3Di range',
    r'No pooling strategy beat': 'non-significance phrasing (m7)',
    r'Supplementary Table S10).} The depth': 'old SI number',
    r'zenodo.23264208': 'old Zenodo version DOI (m1)',
}
for s, why in stale.items():
    check('stale string absent: %s' % s, s not in body, why)

# ---- 3. key numbers
eqs = N['equivalence_summary']
fmt = lambda x: '%.3f' % x
check('equivalence counts 21/7/14/0',
      (eqs['n_tests'], eqs['equivalent'], eqs['inconclusive'], eqs['difference']) == (21, 7, 14, 0)
      and 'leaving 21' in body and '7 were equivalent' in body and '14 were inconclusive' in body)
check('max point estimate 0.019 and upper bound 0.036',
      fmt(eqs['max_mean']) == '0.019' and fmt(eqs['max_upper']) == '0.036'
      and '0.019' in body and '0.036' in body)
check('wild-type max 0.009', fmt(eqs['max_mean_wt']) == '0.009' and 'at most $\\Delta\\rho = 0.009$ for the wild-type' in body)
check('equivalent at margin = 3; t/TOST = 6',
      eqs['equivalent_at_margin'] == 3 and eqs['equivalent_t'] == 6 and eqs['equivalent_tost'] == 6
      and 'Three of the seven' in body and 'one fewer' in body)
check('best layers 26/27 in 30-32', eqs['n_best_30_32'] == 26 and '26 of 27' in body)
T2 = N['table2']
for key in ('ESM-2|mutant|modulo', 'ESM-2|mutant|contiguous', 'ESM-2|mutant|position_groupkfold',
            'SaProt (mask)|mutant|modulo', 'SaProt (mask)|mutant|contiguous',
            'SaProt (mask)|mutant|position_groupkfold', 'SaProt (3Di)|mutant|modulo',
            'SaProt (3Di)|mutant|contiguous', 'SaProt (3Di)|mutant|position_groupkfold'):
    v = T2[key]
    cell = '%s (%d) $\\rightarrow$ %s$^{\\dagger}$' % (fmt(v['best_rho']), v['best'], fmt(v['L32']))
    check('Table 2 cell %s' % key, cell in body, cell)
for arm, sp in (('mutant', 'modulo'), ('mutant', 'contiguous'), ('mutant', 'position_groupkfold')):
    v = N['table4']['%s|%s' % (arm, sp)]
    check('Table 4 %s %s' % (arm, sp), ('$+%.3f$' % v['mean']) in body, '%+.3f' % v['mean'])
lf4 = [v['mean'] for k, v in N['table4'].items() if not k.endswith('random')]
check('3Di range +0.057..+0.132', ('%.3f' % min(lf4), '%.3f' % max(lf4)) == ('0.057', '0.132')
      and '$+0.057$ to $+0.132$' in body and '$+0.06$ to $+0.13$' in body)
r46 = N['ranges']['shared46_best']
check('46-assay range 0.24--0.47', ('%.2f' % r46[0], '%.2f' % r46[1]) == ('0.25', '0.47')
      or ('%.2f' % N['ranges']['shared46_L32'][0], '%.2f' % r46[1]) == ('0.24', '0.47'),
      str(N['ranges']))
check('46-assay range in text', '0.24--0.47' in body)
a = N['anchors_this_work']
check('anchor single-sub L32 modulo ESM-2 0.407 / SaProt 0.469',
      fmt(a['ESM-2']['modulo']['L32']) == '0.407' and fmt(a['SaProt (3Di)']['modulo']['L32']) == '0.469'
      and '0.407 (ESM-2)' in body and '0.469 (SaProt with real 3Di)' in body)
inf = N['inflation']
mp = [v[s] for v in inf.values() for s in ('modulo', 'position_groupkfold')]
cg = [v['contiguous'] for v in inf.values()]
check('inflation ranges 0.14-0.26 / 0.20-0.34',
      ('%.2f' % min(mp), '%.2f' % max(mp), '%.2f' % min(cg), '%.2f' % max(cg)) == ('0.14', '0.26', '0.20', '0.34'))
check('multi-site rows 49,131 (23.0%)', MS['rows_multi'] == 49131 and '49,131' in body and '23.0\\%' in body)
check('shared test rows 15,624 / 15,960 / 6,158',
      (MS['test_rows_shared']['modulo'], MS['test_rows_shared']['position_groupkfold'],
       MS['test_rows_shared']['contiguous']) == (15624, 15960, 6158)
      and all(x in body for x in ('15,624', '15,960', '6,158')))
check('small-dataset sensitivity 0.014 / 0.025',
      fmt(XS['max_mean']) == '0.014' and fmt(XS['max_upper']) == '0.025' and 'estimate 0.014' in body)
check('ESM-2 mutant vs wt on single subs (0.443, 0.476, 0.492)',
      all(x in body for x in ('0.443 versus\n0.476 and 0.492', '0.492 versus\n0.484 and 0.500'))
      or all(x in body.replace('\n', ' ') for x in ('0.443 versus 0.476 and 0.492', '0.492 versus 0.484 and 0.500')))
check('CKA 200 values 0.78 / 0.41 / cos 0.54', all(x in body for x in ('0.78', '0.41', '0.54')))
check('SI S4 caption counts', '(21 tests): 7 equivalent, 14 inconclusive, 0 difference' in si)
check('SI S8 counts', '3 comparisons are equivalent' in si and '20 inconclusive' in si)

# ---- 4. reviewer coverage (anchors inside \rthree{} in the red version)
red_body = red[red.index(r'\begin{abstract}'):]
cover = {
    'R2.1 single-sub primary': 'primary mutant-arm analysis',
    'R2.1 multi-site share': '49,131',
    'R2.1 mutant vs wt restated': 'did not outperform',
    'editor alt text under legends': 'Alt text:',
    'm7 heading and 4.3 wording': 'Does Not Reliably Improve',
    'R2.2 margin verdicts': 'depend on the interval',
    'R2.2 upper bounds': '90\\% upper bound',
    'm1 Zenodo new version': 'zenodo.23878920',
    'm3 random-split pooling': 'Under the random split, by contrast',
    'm3 readout protocol-dependent': 'protocol-dependent',
    'm4 CKA 200': 'not estimable',
    'm5 Kumar leakage-free': 'broadly in line with this observation',
    'm6 NLP hypothesis': 'is\ntherefore a hypothesis',
    'm7 46 assays': '0.24--0.47',
    'm7 significance wording': 'significantly improved',
}
for k, s in cover.items():
    check('coverage %s' % k, s in red_body or s.replace('\n', ' ') in red_body.replace('\n', ' '))

bad = [r for r in results if not r[1]]
for n, ok, d in results:
    print('%s  %s%s' % ('PASS' if ok else 'FAIL', n, ('  [%s]' % d) if (d and not ok) else ''))
print('\n%d checks, %d failed' % (len(results), len(bad)))
sys.exit(1 if bad else 0)
