# NAS: NVIDIA architecture search on Stanford Alpaca

The primary checkout is this repository root on branch `main`. It contains the
NVIDIA experiment formerly developed in the `research/nvidia-minparams` worktree.
That redundant worktree has been removed; its branch and the earlier repository
histories are preserved. No GitHub remote is currently configured.

The user selected **candidate C**, an 11,458,944-parameter transformer, after the
completed search. It has 57.79% fewer parameters than the 27,144,960-parameter
baseline. It uses width 192 and 37 block applications, with shared attention and
FFN modules. Its exact specification is
[`experiments/cuda/candidate-c/spec.json`](experiments/cuda/candidate-c/spec.json).

The three existing search seeds beat their paired baseline on held-out CE. C has
not received five-fresh-seed confirmation. This later user choice does not change
the original experiment outcome: the frozen 10.99M candidate failed its five-seed
held-out comparison. The held-out split was reused from an earlier study.

## Code

| File | Purpose |
|---|---|
| [`cuda_model.py`](cuda_model.py) | Configurable PyTorch transformer and shared modules |
| [`cuda_train.py`](cuda_train.py) | Fixed training recipe, checkpoints, and train/dev curves |
| [`data.py`](data.py) | Fixed data splits, tokenization, and conversation packing |
| [`cuda_minparams_search.py`](cuda_minparams_search.py) | Architecture proposals, scheduling, and run logging |
| [`cuda_evaluate.py`](cuda_evaluate.py) | Checkpoint loading and paired evaluation |
| [`experiments/cuda/candidate-c/metadata.json`](experiments/cuda/candidate-c/metadata.json) | C's dimensions, seeds, results, and checkpoint identities |

Every completed run trained from scratch for 5,000 steps using the same recipe.
The split contains 48,375 training, 1,000 dev, and 2,599 held-out conversations.
The completed study trained 368 distinct architectures across 616 runs; see
[`experiments/cuda/minparams-search-volume.json`](experiments/cuda/minparams-search-volume.json)
for counts and the distinction between Python source and architecture specifications.

## Local experiment artifacts

The run logs, architecture registry, curves, reports, and backed-up checkpoints
are under the ignored directory:

```text
.lambda/artifacts/287cdb48d8e64da3b849dceaaf9b3bc2/cuda/minparams-20261111/
```

`CANDIDATE_C_LOCAL_BACKUP.json` lists C's three checkpoints and the three paired
baseline checkpoints, with verified hashes. These weights, plus the ten original
fresh-confirmation checkpoints, are stored locally. Credentials and artifacts
remain outside Git tracking.

See [`HANDOFF.md`](HANDOFF.md) for the current state and complete research history,
and [`CUDA_README.md`](CUDA_README.md) for the CUDA training workflow.
