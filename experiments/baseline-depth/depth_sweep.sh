#!/bin/zsh
cd "$(dirname "$0")/.."
for n in 6 8; do
  .venv/bin/python -u -m nas.mlx.train --out runs/n$n-d384 --layers $n --steps 5000 --eval-every 250 > runs/n$n-d384.log 2>&1 || exit 1
done
