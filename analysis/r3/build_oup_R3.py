"""Build the OUP-template (bioinfo) version of the R3 manuscript.

Same procedure as build_oup_R2.py: front matter (authors, addresses, keywords)
from manuscript_revised_v2_oup.tex, abstract and body verbatim from
manuscript_revised_R3[_clean].tex. Earlier-round markup (\\rone, \\rtwo) is
black; third-round changes (\\rthree) are red unless --clean.

Usage: python build_oup_R3.py            -> manuscript_revised_R3_oup.tex
       python build_oup_R3.py --clean    -> manuscript_revised_R3_oup_clean.tex
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BS = chr(92)
CLEAN = '--clean' in sys.argv
SRC = ROOT / ('manuscript_revised_R3_clean.tex' if CLEAN else 'manuscript_revised_R3.tex')
OUT = ROOT / ('manuscript_revised_R3_oup_clean.tex' if CLEAN else 'manuscript_revised_R3_oup.tex')
V2OUP = (ROOT / 'manuscript_revised_v2_oup.tex').read_text(encoding='utf-8')
r3 = SRC.read_text(encoding='utf-8')

front = V2OUP[:V2OUP.index(BS + 'abstract{')]
old_macro = BS + 'newcommand{' + BS + 'rev}[1]{' + BS + 'textcolor{red}{#1}}'
assert front.count(old_macro) == 1
front = front.replace(
    old_macro,
    BS + 'newcommand{' + BS + 'rone}[1]{#1}\n'
    + BS + 'newcommand{' + BS + 'rtwo}[1]{#1}\n'
    + BS + 'newcommand{' + BS + 'rthree}[1]{' + ('#1' if CLEAN else BS + 'textcolor{red}{#1}') + '}')
front = front.replace(BS + 'graphicspath{{figures/output/}}', BS + 'graphicspath{{figures/output_r3/}}')
t0 = front.index(BS + 'title[')
t1 = front.index(BS + 'author[')
front = (front[:t0]
         + BS + 'title[Layer Choice in Protein Language Models]{Leakage-Free '
         'Evaluation of Layer Choice in Protein\nLanguage Models for Mutation Effect Prediction}\n\n'
         + front[t1:])

k0 = V2OUP.index(BS + 'keywords{')
k1 = V2OUP.index(BS + 'maketitle')
keywords = V2OUP[k0:k1].strip()

a0 = r3.index(BS + 'begin{abstract}') + len(BS + 'begin{abstract}')
a1 = r3.index(BS + 'end{abstract}')
abstract = re.sub(r'\n\s*\n', '\n', r3[a0:a1].strip())

b0 = r3.index('%=============================================================================\n'
              + BS + 'section{Introduction}')
body = r3[b0:]
for env in ('figure', 'table'):
    body = body.replace(BS + 'begin{%s}[ht]' % env, BS + 'begin{%s*}[!tbp]' % env)
    body = body.replace(BS + 'end{%s}' % env, BS + 'end{%s*}' % env)

out = front + BS + 'abstract{' + abstract + '}\n' + keywords + '\n\n' + BS + 'maketitle\n\n' + body
out = out.replace(BS + 'revised{', BS + 'rone{')
out = out.replace(BS + 'rone{19 September 2026}{0}{2026}', BS + 'revised{19 September 2026}{0}{2026}')
OUT.write_text(out, encoding='utf-8')
print('written', OUT.name)
