#!/usr/bin/env bash
# Local build of the manuscript (same pdfLaTeX + BibTeX toolchain as Overleaf).
# Requires: pdflatex, bibtex (e.g. `sudo apt install texlive-latex-extra texlive-science texlive-fonts-recommended`).
# Output: paper/manuscript.pdf ; full log in paper/manuscript.log
set -eu
cd "$(dirname "$0")"
command -v pdflatex >/dev/null || { echo "pdflatex not found - install TeX Live first (see header)"; exit 1; }
run() { pdflatex -interaction=nonstopmode -halt-on-error -shell-escape manuscript.tex > /dev/null || {
        echo "pdflatex failed; first errors:"; grep -n -A3 "^!" manuscript.log | head -30; exit 1; }; }
run
bibtex manuscript > /dev/null || { echo "bibtex reported problems:"; grep -iE "error|warning" manuscript.blg | head -20; }
run; run
echo "OK -> $(pwd)/manuscript.pdf"
grep -E "undefined|Overfull|multiply" manuscript.log | sort | uniq -c | sort -rn | head -10 || true
