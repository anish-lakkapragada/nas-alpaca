#!/bin/zsh
# Usage: ./queue.sh STEPS spec1.json spec2.json ...   (runs sequentially, commits each attempt)
cd "$(dirname "$0")"
export MPLCONFIGDIR=/tmp/nas-mpl
steps=$1; shift
for s in "$@"; do
  .venv/bin/python search.py "$s" --steps "$steps"
  git add experiments/search && git commit -q -m "Search attempt $(basename $s .json) at K=$steps

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || true
done
