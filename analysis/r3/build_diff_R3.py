"""Build the R2 -> R3 comparison manuscript (old text blue strikethrough, new red).

Same procedure as build_diff_R2.py: latexdiff_lite.py (paragraph level) on the
clean R2 manuscript (figure paths pointed at the R2 figures in output_v2) and
the clean R3 manuscript, then add the ulem package for strikethrough.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BS = chr(92)
old_src = (ROOT / 'manuscript_revised_R2_clean.tex').read_text(encoding='utf-8')
new_path = ROOT / 'manuscript_revised_R3_clean.tex'
new_src = new_path.read_text(encoding='utf-8')
tmp = ROOT / 'revision_docs' / '_r2_paths.tex'
tmp.write_text(old_src, encoding='utf-8')
out = ROOT / 'manuscript_revised_R3_diff.tex'
subprocess.run([sys.executable, str(ROOT / 'revision_docs' / 'latexdiff_lite.py'),
                str(tmp), str(new_path), str(out)], check=True)
tmp.unlink()
s = out.read_text(encoding='utf-8')
anchor = BS + 'usepackage{xcolor}'
assert s.count(anchor) == 1
s = s.replace(anchor, anchor + '\n' + BS + 'usepackage[normalem]{ulem}', 1)
# keep the unchanged enumerate list verbatim (multi-line \cite)
start = BS + 'begin{enumerate}\n    ' + BS + 'item Structural features'
end = BS + 'end{enumerate}'
if start in s:
    i0 = s.index(start)
    i1 = s.index(end, i0) + len(end)
    j0 = new_src.index(start)
    j1 = new_src.index(end, j0) + len(end)
    s = s[:i0] + new_src[j0:j1] + s[i1:]
out.write_text(s, encoding='utf-8')
print('written', out.name)
