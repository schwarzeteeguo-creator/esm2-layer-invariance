"""Paragraph-level visual diff between two .tex files (latexdiff-lite).

Old-only / replaced blocks are emitted as blue strikethrough (LaTeX escaped to
plain text since ulem cannot nest arbitrary macros); new blocks are red.
Good enough for surgical whole-paragraph revision markup."""
import sys, difflib, re

BS = chr(92)  # backslash


def blocks(path):
    with open(path, encoding='utf-8') as f:
        return f.read().split('\n\n')


def strip_markup(s):
    return re.sub(r'\\revised\{([^{}]*)\}', r'\1', s)


ENV = re.compile(r'^\s*' + re.escape(BS) + r'(begin|end)\{[^}]*\}.*$')
ITEM = re.compile(r'^\s*' + re.escape(BS) + r'item\b.*$')
LISTENV = ('enumerate', 'itemize', 'description')


def envname(line):
    m = re.match(r'^\s*' + re.escape(BS) + r'(begin|end)\{([^}]*)\}', line)
    return m.group(2) if m else None


def delblock(b):
    return wrapblock(b, 'del')

def addblock(b):
    # tables cannot be line-wrapped in color groups; emit verbatim
    if BS + 'begin{tabular}' in b or BS + 'begin{table}' in b:
        return b
    return wrapblock(b, 'add')


def wrapblock(b, mode):
    """New blocks keep env lines raw; deleted blocks drop env lines (they are
    re-emitted by the replacement block), marking only text runs."""
    out, run = [], []
    for line in b.split('\n'):
        keep_raw = False
        if ENV.match(line):
            keep_raw = (mode != 'del') or (envname(line) in LISTENV)
        elif ITEM.match(line):
            if mode == 'del':
                # \textbf{...} on the item line may span into struck text;
                # keep only the \item prefix and strike the rest
                rest = re.sub(r'^\s*' + re.escape(BS) + r'item\b\s*', '', line)
                if run:
                    out.append(mark('\n'.join(run), mode))
                    run = []
                out.append(BS + 'item ')
                run.append(rest)
                continue
            keep_raw = True
        if keep_raw:
            if run:
                out.append(mark('\n'.join(run), mode))
                run = []
            out.append(line)
        else:
            run.append(line)
    if run:
        out.append(mark('\n'.join(run), mode))
    return '\n'.join(out)


def mark(t, mode):
    if not t.strip():
        return t
    if mode == 'add':
        return BS + 'textcolor{red}{' + t + '}'
    # deleted: fully plainify so ulem sees only ordinary characters
    s = re.sub(r'\$[^$]*\$', '[math]', t)
    s = re.sub(re.escape(BS) + r'(begin|end)\{[^}]*\}', '', s)
    s = re.sub(re.escape(BS) + r'[a-zA-Z]+', '', s)
    s = re.sub(re.escape(BS) + r'.', '', s)
    s = s.replace('{', '').replace('}', '')
    s = re.sub(r'[&_&%#^~]', ' ', s)
    s = s.replace('_', ' ')
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r'\n\s*\n+', '\n', s).strip('\n ')   # no blank lines inside sout
    if not s.strip():
        return ''
    return BS + 'textcolor{blue}{' + BS + 'sout{' + s + '}}'


old, new = blocks(sys.argv[1]), blocks(sys.argv[2])

# keep the NEW preamble verbatim (it defines the color/ulem macros)
cut_new = next(i for i, b in enumerate(new) if BS + 'begin{document}' in b)
cut_old = next(i for i, b in enumerate(old) if BS + 'begin{document}' in b)
preamble = new[:cut_new + 1]
old, new = old[cut_old + 1:], new[cut_new + 1:]

sm = difflib.SequenceMatcher(a=[strip_markup(b) for b in old],
                             b=[strip_markup(b) for b in new], autojunk=False)

out = list(preamble)
for tag, i1, i2, j1, j2 in sm.get_opcodes():
    if tag == 'equal':
        out.extend(old[i1:i2])
    elif tag in ('delete', 'replace'):
        for b in old[i1:i2]:
            out.append(delblock(b))
        if tag == 'replace':
            for b in new[j1:j2]:
                out.append(addblock(b))
    elif tag == 'insert':
        for b in new[j1:j2]:
            out.append(addblock(b))

body = '\n\n'.join(out)
with open(sys.argv[3], 'w', encoding='utf-8') as f:
    f.write(body)
print('written', sys.argv[3])
