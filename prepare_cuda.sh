#!/usr/bin/env bash
# Run on the rented Linux GPU host from the checkout directory.
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv --system-site-packages .venv-cuda
.venv-cuda/bin/python -m pip install -r requirements-cuda.txt
.venv-cuda/bin/python - <<'PY'
import json
import torch
assert torch.cuda.is_available(), 'No working CUDA device'
print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda,
                  'gpus': [{'index': i, 'name': torch.cuda.get_device_name(i),
                            'memory_gb': torch.cuda.get_device_properties(i).total_memory / 1e9}
                           for i in range(torch.cuda.device_count())]}, indent=2))
PY

