"""R3 audit: wording the response letter says is now in the manuscript must be
present, and wording it says was removed must be absent (clean R3 sources)."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BS = chr(92)


def norm(t):
    t = re.sub(r'\\(rthree|rtwo|revised|emph|textbf|textit)\{', '', t)
    t = re.sub(r'\$([^$]*)\$', r'\1', t)
    t = t.replace(BS + 'rightarrow', '→').replace('^' + BS + 'dagger', '')
    for a, b in ((BS + 'Delta' + BS + 'rho', 'Δρ'), (BS + 'rho', 'ρ'), (BS + 'pm', '±'),
                 (BS + 'leq', '≤'), ('---', ', '), ('--', '-'), ('~', ' '), (BS + '%', '%'),
                 (BS + '_', '_'), (BS, ''), ('{', ''), ('}', ''), ('−', '-'), ('–', '-')):
        t = t.replace(a, b)
    return re.sub(r'\s+', ' ', t).lower()


ms = norm((ROOT / 'manuscript_revised_R3_clean.tex').read_text(encoding='utf-8'))
si = norm((ROOT / 'supplementary_R3.tex').read_text(encoding='utf-8'))
both = ms + ' ' + si

PRESENT = [
    # R2.1
    'analyzed on single-substitution variants only',
    'this is the primary mutant-arm analysis throughout',
    '0.443 (30) → 0.427', '0.492 (30) → 0.486',
    'the mutant arm exceeded the wild-type arms under the random split (esm-2: 0.681 versus 0.606)',
    'modulo, best layer: 0.443 versus 0.476 and 0.492',
    '0.492 versus 0.484 and 0.500',
    'did not outperform the wild-type context embedding combined with the auxiliary substitution features',
    'alt text:',
    'multi-scale pooling does not significantly improve on the final layer under a leakage-free protocol',
    'concat 5l offered no significant benefit',
    '0.24-0.47', '0.44 versus up to 0.49', '0.35-0.49',
    '49,131 analyzed rows (23.0%) come from multi-site variants',
    '15,624 of the 213,393 analyzed rows (7.3%)',
    'fewer than 50 single-substitution variants',
    '17 of 24 significantly worse',
    '+0.057 to +0.132', '+0.06 to +0.13', 'p ≤ 0.0011',
    # R2.2
    'exceeded the final layer by at most δρ = 0.019 (90% upper bound 0.036). this deficit is small relative to the cross-dataset standard deviation of 0.17',
    'small compared with the cross-dataset standard deviation of about 0.17',
    'within δρ = 0.019 of the best layer (point estimates; 90% upper bounds up to 0.036)',
    'three of the seven equivalence verdicts have upper bounds within 0.0005 of the margin',
    'classify one fewer comparison as equivalent',
    'leaving 21 comparisons',
    'three of the seven equivalence verdicts nevertheless lie at the margin',
    '3 comparisons are equivalent, 20 inconclusive',
    # minors
    'v3.1.0',
    'under the random split, by contrast, some fusion strategies did exceed the last layer',
    '(+0.009 to +0.012, 90% cis excluding zero)', '10 of 12 strategy-readout combinations for saprot (+0.010 to +0.064)',
    'under leakage-free evaluation',
    'the readout ranking is itself protocol-dependent',
    'this ranking holds under the modulo split and reverses under the random split',
    'cosine 0.54 with ckaa'[:0] + 'cosine 0.54 with cka 0.41', 'cosine -0.01 with cka 0.78',
    'not estimable',
    'broadly in line with this observation',
    'the difference is one of degree',
    'therefore a hypothesis',
    'reaches 0.407 (esm-2) and 0.469 (saprot with real 3di) under the modulo split',
    'significantly improved on the final layer under leakage-free evaluation',
    'which does not exclude a gain of that size',
]
ABSENT = [
    'the mutant arm exceeded the masked arms in every split',
    'operates on a single level of abstraction',
    'within 0.02 on single substitutions',
    '3.8% of analyzed variants', '8.4% of analyzed rows',
    'gave higher, not lower, values',
    'random-split probing shows little layer dependence',
    'no pooling strategy beat',
    'zenodo.23264208',
    'added no detectable information',
    'shows no practical gain',
    'concat 5l offered no benefit',
]
bad = 0
for p in PRESENT:
    ok = norm(p) in both
    bad += not ok
    print('%s  present: %s' % ('PASS' if ok else 'FAIL', p))
for p in ABSENT:
    ok = norm(p) not in both
    bad += not ok
    print('%s  absent:  %s' % ('PASS' if ok else 'FAIL', p))
print('\n%d checks, %d failed' % (len(PRESENT) + len(ABSENT), bad))
sys.exit(1 if bad else 0)
