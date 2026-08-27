#!/usr/bin/env bash
# Compile artifacts/manuscript/draft_v0.tex and optionally split/glue at
# the Supplementary Material heading (after journal Appendices A--F).
#
# Usage:
#   scripts/compile_manuscript.sh            # combined PDF only (default)
#   scripts/compile_manuscript.sh all        # same
#   scripts/compile_manuscript.sh split      # split an existing combined PDF
#   scripts/compile_manuscript.sh glue       # join main + supplement → combined
#   scripts/compile_manuscript.sh both       # compile combined, then split (submission bundle)
#
# Outputs (under artifacts/manuscript/):
#   draft_v0.pdf              combined (article + Supplementary Material)
#   draft_v0_main.pdf         journal article: main text + Appendices A--F
#   draft_v0_supplement.pdf   Supplementary Material (S1--S12)
#   draft_v0.appsplit         first Supplementary page (written by LaTeX)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MS="$ROOT/artifacts/manuscript"
TEX="draft_v0.tex"
COMBINED="$MS/draft_v0.pdf"
MAIN_PDF="$MS/draft_v0_main.pdf"
SUPP_PDF="$MS/draft_v0_supplement.pdf"
SPLITFILE="$MS/draft_v0.appsplit"
MODE="${1:-all}"
TECTONIC="${TECTONIC:-$HOME/.local/bin/tectonic}"

gs_pages() {
  local infile="$1" outfile="$2" first="$3" last="$4"
  local args=(-dBATCH -dNOPAUSE -dSAFER -dQUIET -sDEVICE=pdfwrite "-sOutputFile=$outfile")
  if [[ -n "$first" ]]; then
    args+=("-dFirstPage=$first")
  fi
  if [[ -n "$last" ]]; then
    args+=("-dLastPage=$last")
  fi
  gs "${args[@]}" "$infile"
}

pdf_page_count() {
  gs -q -dNODISPLAY -dNOSAFER -c "($1) (r) file runpdfbegin pdfpagecount = quit"
}

# First page whose text starts with the Supplementary Material heading.
# LaTeX \\thepage at that source line can be early if appendix floats defer.
find_sm_page() {
  python3 - "$COMBINED" <<'PY'
import subprocess, sys
pdf = sys.argv[1]
try:
    info = subprocess.check_output(["pdfinfo", pdf], text=True)
    total = int(next(line.split(":", 1)[1] for line in info.splitlines() if line.startswith("Pages:")))
except Exception as e:
    sys.stderr.write(f"page count failed: {e}\n")
    sys.exit(1)
for p in range(1, total + 1):
    t = subprocess.check_output(
        ["pdftotext", "-f", str(p), "-l", str(p), pdf, "-"],
        text=True, errors="replace",
    )
    head = t.lstrip("\ufeff \t\r\n")
    if head.startswith("Supplementary Material"):
        print(p)
        sys.exit(0)
sys.stderr.write("Supplementary Material heading not found in PDF\n")
sys.exit(1)
PY
}

read_split_page() {
  local start
  start="$(find_sm_page)"
  if [[ ! "$start" =~ ^[0-9]+$ ]] || [[ "$start" -lt 2 ]]; then
    echo "bad Supplementary start page: '$start'" >&2
    exit 1
  fi
  printf '%s\n' "$start" >"$SPLITFILE"
  printf '%s\n' "$start"
}

compile_combined() {
  if [[ ! -x "$TECTONIC" ]]; then
    echo "tectonic not found at $TECTONIC" >&2
    exit 1
  fi
  echo "compiling $TEX → $COMBINED"
  (cd "$MS" && "$TECTONIC" --keep-logs --keep-intermediates -p "$TEX")
}

do_split() {
  if [[ ! -f "$COMBINED" ]]; then
    echo "missing $COMBINED — compile first" >&2
    exit 1
  fi
  local start total last_main
  start="$(read_split_page)"
  if [[ ! "$start" =~ ^[0-9]+$ ]] || [[ "$start" -lt 2 ]]; then
    echo "bad Supplementary start page in $SPLITFILE: '$start'" >&2
    exit 1
  fi
  total="$(pdf_page_count "$COMBINED")"
  last_main=$((start - 1))
  echo "split: journal article pp. 1–${last_main} (main + Apps A–F); Supplementary Material pp. ${start}–${total} (of ${total})"
  gs_pages "$COMBINED" "$MAIN_PDF" 1 "$last_main"
  gs_pages "$COMBINED" "$SUPP_PDF" "$start" ""
  echo "wrote $MAIN_PDF"
  echo "wrote $SUPP_PDF"
}

do_glue() {
  if [[ ! -f "$MAIN_PDF" || ! -f "$SUPP_PDF" ]]; then
    echo "need $MAIN_PDF and $SUPP_PDF — run split first" >&2
    exit 1
  fi
  echo "glue: $MAIN_PDF + $SUPP_PDF → $COMBINED"
  gs -dBATCH -dNOPAUSE -dSAFER -dQUIET -sDEVICE=pdfwrite "-sOutputFile=$COMBINED" "$MAIN_PDF" "$SUPP_PDF"
  echo "wrote $COMBINED"
}

case "$MODE" in
  all|combined)
    compile_combined
    ;;
  split)
    do_split
    ;;
  glue|join)
    do_glue
    ;;
  both|bundle)
    compile_combined
    do_split
    ;;
  -h|--help|help)
    sed -n '2,20p' "$0"
    ;;
  *)
    echo "unknown mode: $MODE (all|split|glue|both)" >&2
    exit 2
    ;;
esac
