#!/usr/bin/env bash
# Compile artifacts/manuscript/draft_v0.tex and optionally split/glue at Appendix A.
#
# Usage:
#   scripts/compile_manuscript.sh            # combined PDF only (default)
#   scripts/compile_manuscript.sh all        # same
#   scripts/compile_manuscript.sh split      # split an existing combined PDF
#   scripts/compile_manuscript.sh glue       # join main + supplement → combined
#   scripts/compile_manuscript.sh both       # compile combined, then split (submission bundle)
#
# Outputs (under artifacts/manuscript/):
#   draft_v0.pdf              combined (main + appendix)
#   draft_v0_main.pdf         pages 1 .. appendix-start-1
#   draft_v0_supplement.pdf   appendix-start .. end
#   draft_v0.appsplit         first appendix page (written by LaTeX)
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

read_split_page() {
  if [[ ! -f "$SPLITFILE" ]]; then
    echo "missing $SPLITFILE — compile with 'all' or 'both' first" >&2
    exit 1
  fi
  tr -d ' \t\r\n' <"$SPLITFILE"
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
    echo "bad appendix start page in $SPLITFILE: '$start'" >&2
    exit 1
  fi
  total="$(pdf_page_count "$COMBINED")"
  last_main=$((start - 1))
  echo "split: main pp. 1–${last_main}; supplement pp. ${start}–${total} (of ${total})"
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
