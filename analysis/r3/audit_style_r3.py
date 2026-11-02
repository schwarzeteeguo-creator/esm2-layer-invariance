"""R3 style audit of the new (red) text: sentences over 30 words, strong
claim words, and terminology variants. Reports only; edits are made in
build_manuscript_R3.py."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BS = chr(92)
red = (ROOT / 'manuscript_revised_R3.tex').read_text(encoding='utf-8')
clean = (ROOT / 'manuscript_revised_R3_clean.tex').read_text(encoding='utf-8')


def plain(t):
    t = re.sub(r'\$[^$]*\$', 'X', t)
    t = re.sub(r'\\(cite|ref|label)\{[^}]*\}', '', t)
    t = re.sub(r'\\[a-zA-Z]+\*?', ' ', t)
    return re.sub(r'\s+', ' ', t.replace('{', '').replace('}', '').replace('~', ' ')).strip()


# extract \rthree{...} arguments with brace matching
segs, i, key = [], 0, BS + 'rthree{'
while True:
    i = red.find(key, i)
    if i < 0:
        break
    j, depth = i + len(key), 1
    while depth:
        depth += {'{': 1, '}': -1}.get(red[j], 0)
        j += 1
    segs.append(red[i + len(key):j - 1])
    i = j
new_text = ' '.join(plain(s) for s in segs)

# sentences in the full clean body that overlap new text
body = clean[clean.index('begin{abstract}'):clean.index('begin{thebibliography}')]
sents = re.split(r'(?<=[.;])\s+(?=[A-Z(])', plain(body))
print('== sentences > 30 words that contain R3 text')
for s in sents:
    n = len(s.split())
    if n > 30 and any(len(seg) > 25 and seg[:25] in s for seg in (plain(x) for x in segs)):
        print('  [%d] %s' % (n, s[:220]))

print('\n== strong claim words in R3 text')
for w in ('demonstrat', 'prove', 'establish', 'confirm', 'no detectable', 'clearly', 'always', 'never'):
    for m in re.finditer(w, new_text):
        print('  %-14s ...%s...' % (w, new_text[max(0, m.start() - 80):m.end() + 60]))

print('\n== terminology counts (clean body)')
for t in ('leakage-free split', 'position-based split', 'single-substitution variant',
          'single substitution', 'wild-type arm', 'masked arm', 'structure-masked', 'real 3Di',
          'real-3Di', 'position-grouped', 'position groupkfold', 'multi-site variant', 'multi-mutant'):
    print('  %-28s %d' % (t, len(re.findall(re.escape(t), body))))
