#!/usr/bin/env bash
#
# Typeset PAPER_ABSTRACT.md as a PDF.
#
# The Markdown file is the single source of truth. paper/body.tex is generated
# from it and MUST NOT be edited by hand -- this script overwrites it. Edit the
# Markdown, then re-run. Figure PDFs under paper/figures/ are produced by
# scripts/generate-paper-figures.py (`npm run figures`); `npm run paper`
# regenerates figures before typesetting.
#
# Usage:  scripts/build-paper.sh
#         scripts/build-paper.sh --xelatex   # optional; needs system Unicode fonts
#
# Requires pandoc, python3, and a TeX distribution. Default engine is pdfLaTeX;
# main.tex declares the non-ASCII characters the source uses under that engine.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_md="${repo_root}/PAPER_ABSTRACT.md"
paper_dir="${repo_root}/paper"
body_tex="${paper_dir}/body.tex"

engine="pdf"
if [[ "${1:-}" == "--xelatex" ]]; then
  engine="xelatex"
fi

for tool in pandoc latexmk python3; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "FAIL: $tool not found on PATH" >&2
    exit 1
  fi
done

if [[ ! -f "$source_md" ]]; then
  echo "FAIL: $source_md does not exist" >&2
  exit 1
fi

echo "== converting Markdown to LaTeX"
# --shift-heading-level-by=-1: Markdown H1 becomes the document title metadata
# (omitted from the body), H2 -> \section, H3 -> \subsection. Without this,
# the paper title becomes section 1 and every real section nests under it.
#
# --wrap=preserve keeps the source line structure so a diff of body.tex maps
# onto a diff of the Markdown rather than reflowing the whole file.
pandoc "$source_md" -t latex --wrap=preserve --shift-heading-level-by=-1 \
  -o "${body_tex}.raw"

python3 - "$body_tex" <<'PY'
import re, sys
from pathlib import Path

path = Path(sys.argv[1])
raw = Path(str(path) + ".raw").read_text(encoding="utf-8")

# Strip manual numbers from headings ("1. Introduction", "2.1 Foo") so LaTeX's
# automatic numbering is not doubled in the PDF.
def strip_num(m: re.Match[str]) -> str:
    cmd, title = m.group(1), m.group(2)
    title = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", title)
    return f"\\{cmd}{{{title}}}"

raw = re.sub(
    r"\\(section|subsection|subsubsection)\{([^}]+)\}",
    strip_num,
    raw,
)

# Pandoc escapes literal citation brackets as {[}1{]}; unwrap to [1].
raw = re.sub(r"\{\[\}(\d+)\{\]\}", r"[\1]", raw)

# Abstract is not a numbered section. Use str.replace (not re.sub): re.sub
# treats \b in the replacement as a backspace character.
raw = raw.replace(
    "\\section{Abstract}\\label{abstract}\n",
    "\\begin{abstract}\n",
    1,
)
raw = raw.replace(
    "\n\\textbf{Keywords:",
    "\n\\end{abstract}\n\n\\textbf{Keywords:",
    1,
)

# References: unnumbered, still bookmarkable via the label pandoc emits.
raw = raw.replace("\\section{References}", "\\section*{References}", 1)

path.write_text(raw, encoding="utf-8")
Path(str(path) + ".raw").unlink(missing_ok=True)
print(f"wrote {path} ({raw.count(chr(10)) + 1} lines)")
PY

echo "== typesetting (${engine}latex)"
cd "$paper_dir"
if [[ "$engine" == "xelatex" ]]; then
  latexmk -xelatex -interaction=nonstopmode main.tex >/tmp/build-paper.log 2>&1 || true
else
  latexmk -pdf -interaction=nonstopmode main.tex >/tmp/build-paper.log 2>&1 || true
fi

# latexmk returns non-zero for warnings as well as errors, so the PDF's
# existence and the absence of TeX errors are the real success conditions.
errors=$(grep -cE '^!' /tmp/build-paper.log || true)
if [[ ! -f main.pdf ]]; then
  echo "FAIL: no PDF produced. Last 20 lines of the log:" >&2
  tail -20 /tmp/build-paper.log >&2
  exit 1
fi
if [[ "$errors" -gt 0 ]]; then
  echo "FAIL: ${errors} TeX error(s):" >&2
  grep -A3 -E '^!' /tmp/build-paper.log | head -40 >&2
  exit 1
fi

pages=$(grep -oE 'Output written on main\.pdf \([0-9]+ pages' /tmp/build-paper.log | grep -oE '[0-9]+' | head -1)
echo "== PASS: paper/main.pdf (${pages:-?} pages)"
