"""R3 audit of the Supplementary Information against the reviewer's SI-related
points (R1 m8 of round 2; R2.1, R2.2, m1-m4, m7 of round 3), the result files,
and the main-text citations. Exit code 1 if any check fails."""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
si = (ROOT / 'supplementary_R3.tex').read_text(encoding='utf-8')
ms = (ROOT / 'manuscript_revised_R3_clean.tex').read_text(encoding='utf-8')
R2 = ROOT / 'revision_results/r2_analysis'
R3 = ROOT / 'revision_results/r3_analysis'
N = json.load(open(R3 / 'r3_numbers.json'))
MS = json.load(open(R3 / 'multisite_share_r3.json'))
POOL = json.load(open(R2 / 'a2_a3_pooling.json'))
EXCL = json.load(open(R3 / 'pooling_excl_multisite_r3.json'))
res = []


def check(name, ok, detail=''):
    res.append((name, bool(ok), detail))


def table_block(label):
    i = si.index(r'\label{%s}' % label)
    j = si.find(r'\end{tabular}', i)
    k = si.find(r'\end{longtable}', i)
    end = min(x for x in (j, k) if x > 0)
    return si[i:end]


def data_rows(block):
    rows = []
    for line in block.splitlines():
        line = line.strip()
        if line.endswith(r'\\') and '&' in line and not line.startswith(('Dataset', 'Layer', 'Model', 'Readout',
                                                                         'Method', '&', r'\multicolumn', 'Snapshot', r' &')):
            rows.append([c.strip() for c in line[:-2].split('&')])
    return rows


# ---- numbering, order, labels
labels = re.findall(r'\\label\{tab:S(\d+)\}', si)
check('SI tables labelled S1..S11 in document order', [int(x) for x in labels] == list(range(1, 12)), str(labels))
check('SI is unmarked (no colour commands)', 'textcolor' not in si and 'rthree' not in si)

# ---- round-2 m8: completeness
s1 = [r for r in data_rows(table_block('tab:S1')) if len(r) == 13]
names63 = [l.strip() for l in open(R2 / 'assay_names_63.txt') if l.strip()]
s1_names = [r[0].replace('\\_', '_') for r in s1]
check('S1 lists all 63 datasets', sorted(s1_names) == sorted(names63), '%d rows' % len(s1))
check('S1 includes VKOR1 x2 and YAP1 (missing in round 2)',
      all(n in s1_names for n in ('VKOR1_HUMAN_Chiasson_abundance_2020', 'VKOR1_HUMAN_Chiasson_activity_2020',
                                  'YAP1_HUMAN_Araya_2012')))
for lab in ('tab:S5', 'tab:S6', 'tab:S7'):
    rows = data_rows(table_block(lab))
    check('%s has 33 layers x 12 columns' % lab, len(rows) == 33 and all(len(r) == 13 for r in rows),
          '%d rows' % len(rows))
s2 = data_rows(table_block('tab:S2'))
n3di = sum(1 for r in s2 for k in (0, 3) if len(r) == 6 and r[k])
check('S2 (3Di) lists 62 datasets in both columns', n3di == 62, str(n3di))
log = (ROOT / 'supplementary_R3.log').read_text(encoding='latin-1')
check('no overfull boxes in SI (nothing clipped)', 'Overfull' not in log)

# ---- stale content from the Zenodo v3.0.0 copy (R2.m1) and earlier SI
for s in ('95.8', '84.6', 'all 63 assays are in', 'Table 6', 'Table~S6', 'within 0.02',
          'mutant arm only', 'gave higher', 'unstable'):
    check('SI does not contain "%s"' % s, s not in si)
check('SI states 46 / 13 / 4 v1.3 provenance', 'Of these 63 assay identifiers, 46 are unchanged' in si
      and '13\ncorrespond' in si or '13 correspond' in si.replace('\n', ' '))

# ---- R2.1: multi-site share, single-substitution table with all arms
summ = MS['summary']
check('S1 text: 49,131 of 213,393 rows (23.0%)', '49,131 of the 213,393 analyzed rows (23.0' in si.replace('\n', ' '))
tot_multi = sum(int(r[4]) for r in s1)
tot_rows = sum(int(r[3]) for r in s1)
tot_shared = sum(int(r[5]) for r in s1)
check('S1 column sums match (rows, multi-site rows, shared rows)',
      (tot_rows, tot_multi, tot_shared) == (summ['rows'], summ['rows_multi'], summ['test_rows_shared']['modulo']),
      '%d %d %d' % (tot_rows, tot_multi, tot_shared))
s8 = data_rows(table_block('tab:S8'))
arms8 = {r[1] for r in s8}
check('S8 has 27 rows covering all three arms', len(s8) == 27 and arms8 == {r'masked\_wt', r'unmasked\_wt', 'mutant'})
verd8 = [r[8] for r in s8 if r[8] != '--']
check('S8 verdicts 3 equivalent / 20 inconclusive / 0 difference',
      (verd8.count('Equivalent'), verd8.count('Inconclusive'), verd8.count('Difference')) == (3, 20, 0))
row = [r for r in s8 if r[0] == 'ESM-2' and r[1] == 'mutant' and r[2] == 'Modulo'][0]
check('S8 ESM-2 mutant modulo: single 0.443 vs full 0.483', row[5] == '0.443' and row[10] == '0.483', str(row))
row = [r for r in s8 if r[0] == 'ESM-2' and r[1] == 'mutant' and r[2] == 'Pos.-grouped'][0]
check('S8 ESM-2 mutant pos.-grouped CI upper 0.036', '+0.036]' in row[7], row[7])

# ---- R2.2: equivalence table matches r3_numbers
s4 = data_rows(table_block('tab:S4'))
check('S4 has 21 comparisons', len(s4) == 21, str(len(s4)))
eq = {(e['model'], e['arm'], e['split']): e for e in N['equivalence']}
SPL = {'Modulo': 'modulo', 'Contiguous': 'contiguous', 'Pos.-grouped': 'position_groupkfold'}
ok = True
for r in s4:
    e = eq[(r[0], r[1].replace('\\_', '_'), SPL[r[2]])]
    want = '[%+.4f, %+.4f]' % tuple(e['boot90'])
    ok &= (r[7] == want and r[11] == e['verdict'] and r[12] == e['verdict_t']
           and r[3] == ('single' if e['arm'] == 'mutant' else 'full'))
check('S4 intervals, verdicts and data source match result file', ok)
check('S4 verdict counts 7/14/0, t-verdict 6', [r[11] for r in s4].count('Equivalent') == 7
      and [r[11] for r in s4].count('Inconclusive') == 14 and [r[12] for r in s4].count('Equivalent') == 6)
check('S4 text names 5th/95th percentiles', '5th and 95th' in si)

# ---- m3: random-split pooling and readout contrast
s9 = data_rows(table_block('tab:S9'))
check('S9 has 12 rows x 6 data columns', len(s9) == 12 and all(len(r) == 8 for r in s9))
v = POOL['fusion_vs_last']['ESM-2|random']['concat_5L|ridge']
check('S9 ESM-2 random Concat5L Ridge = %+.3f' % v['mean_delta'],
      any(r[0] == 'Ridge' and r[1] == 'Concat 5L' and r[3].startswith('%+.3f' % v['mean_delta']) for r in s9))
v = EXCL['SaProt|modulo']['concat_5L|lightgbm']
check('S9 SaProt modulo-excl Concat5L LightGBM = %+.3f' % v['mean_delta'],
      any(r[0] == 'LightGBM' and r[1] == 'Concat 5L' and r[7].startswith('%+.3f' % v['mean_delta']) for r in s9))
s10 = data_rows(table_block('tab:S10'))
check('S10 has random/modulo/contiguous for both models', len(s10) == 6)
check('S10 random-split contrast -0.035 / -0.065',
      any(r[0] == 'ESM-2' and r[1] == 'Random' and r[2] == '-0.035' for r in s10)
      and any(r[0] == 'SaProt' and r[1] == 'Random' and r[2] == '-0.065' for r in s10))

# ---- m4: CKA
s3 = data_rows(table_block('tab:S3'))
check('S3 lists 200 positions first (primary) and labels debiased as not estimable',
      'ESM-2 (200)' in s3[0][0] and 'not estimable' in si)
check('S3 SaProt 200 CKA 0.779, ESM-2 0.409',
      any(r[0] == 'SaProt (200)' and r[4] == '0.779' for r in s3) and any(r[0] == 'ESM-2 (200)' and r[4] == '0.409' for r in s3))

# ---- m7: anchors
a = N['anchors_this_work']
s11 = si[si.index(r'\label{tab:S11}'):si.index(r'\end{table}', si.index(r'\label{tab:S11}'))]
want = [r'This work: %s final layer & %.3f & %.3f & %.3f \\' % (
    m, a[m]['random']['L32'], a[m]['modulo']['L32'], a[m]['contiguous']['L32']) for m in ('ESM-2', 'SaProt (3Di)')]
check('S11 this-work rows are single-substitution final-layer values',
      all(w in s11 for w in want) and 'single-substitution variants of the same' in si.replace('\n', ' '))

# ---- main-text citations point to the right content
flat = re.sub(r'\s+', ' ', ms)
cites = {
    'S1': ('per-assay status', 'per-dataset counts', 'per-dataset results'),
    'S2': ('3Di token lengths',),
    'S3': ('debiased estimator', 'estimable'),
    'S4': ('comparisons', 'equivalent'),
    'S8': ('single substitutions', 'single-substitution'),
    'S9': ('Paired per-dataset', 'multi-site', 'paired comparisons'),
    'S10': ('Ridge',),
    'S11': ('anchor', 'single substitutions', 'shared'),
}
for tab, keys in cites.items():
    ctx = [flat[max(0, m.start() - 300):m.end() + 50] for m in re.finditer(r'Table (?:\\rthree\{)?%s\b' % tab, flat)]
    good = bool(ctx) and all(any(k in c for k in keys) for c in ctx)
    check('main-text citations of %s match its content (%d)' % (tab, len(ctx)), good)
ctx = re.search(r'Tables (?:\\rthree\{)?S5--S7', flat)
check('main text cites S5-S7 for the layer curves', bool(ctx))

bad = [r for r in res if not r[1]]
for n, ok, d in res:
    print('%s  %s%s' % ('PASS' if ok else 'FAIL', n, ('  [%s]' % d) if d and not ok else ''))
print('\n%d checks, %d failed' % (len(res), len(bad)))
sys.exit(1 if bad else 0)
