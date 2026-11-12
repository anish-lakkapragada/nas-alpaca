# Neural Architecture Search on Stanford Alpaca

![Baseline architecture (27.14M parameters) and Codex-optimized architecture (11.46M parameters), showing the full execution order and shared attention and FFN weights.](figures/baseline_vs_codex_architecture.png)

[Full-resolution PNG](figures/baseline_vs_codex_architecture.png) ·
[Vector SVG](figures/baseline_vs_codex_architecture.svg) ·
[PDF](figures/baseline_vs_codex_architecture.pdf)

The **Codex-optimized architecture** has **11,458,944 parameters—57.79% fewer**
than the 27,144,960-parameter baseline. In the three paired search seeds, it
achieved lower dev and held-out cross-entropy (CE) than the baseline. Both models
were trained from scratch on Stanford Alpaca with a fixed 5,000-step budget and
the same training recipe. The goal was to reduce total parameters while matching
or improving the baseline's dev CE.

## CE results

Values below are means across **training seeds 0, 1, and 2**. CE is measured in
nats per supervised answer token, including the answer's end-of-sequence token;
prompt and padding tokens are excluded. Lower CE is better.

| Metric | Baseline architecture | Codex-optimized architecture | Change (optimized − baseline) |
|---|---:|---:|---:|
| Parameters | 27,144,960 | 11,458,944 | −57.79% |
| Train diagnostic CE (256 conversations) | 2.32171 | 2.75179 | +0.43008 |
| Dev CE (1,000 conversations) | 3.42073 | **3.41036** | −0.01037 |
| Held-out CE (2,599 conversations) | 3.35512 | **3.34595** | −0.00917 |

The train diagnostic is a fixed subset used only to measure training loss;
optimization uses the full **48,375-conversation training split**. Evaluation
runs with dropout disabled. The optimized model has higher training diagnostic
CE but lower dev and held-out CE on these seeds.

All six runs completed 5,000 optimizer steps. Scores use each run's **best-dev
checkpoint**, selected at 250-step evaluation intervals: steps 4,750 / 4,750 /
4,500 for the baseline and step 5,000 for all three optimized runs. Thus the
training budget is fixed, but the reported checkpoints are not all from the
same step.

<details>
<summary>Per-seed CE comparison</summary>

| Training seed | Baseline dev CE | Optimized dev CE | Baseline held-out CE | Optimized held-out CE |
|---|---:|---:|---:|---:|
| 0 | 3.40885 | 3.40485 | 3.34454 | 3.33870 |
| 1 | 3.41911 | 3.40413 | 3.35449 | 3.34199 |
| 2 | 3.43423 | 3.42211 | 3.36634 | 3.35716 |

</details>

**Evaluation scope.** The Codex-optimized architecture (formerly candidate C)
was selected by the user after the completed search and disclosure of an
independent held-out diagnostic. Those diagnostic scores were withheld from
the architecture search while it ran. This model has **not received fresh-seed
confirmation**, and the held-out split was reused from an earlier study. The
original frozen 10.99M search winner was a different architecture and failed its
five-fresh-seed held-out comparison; the results above do not replace that
outcome. Exact optimized-model scores and run identities are in
[`metadata.json`](experiments/cuda/candidate-c/metadata.json); the comparison and
selection history are recorded in [`HANDOFF.md`](docs/HANDOFF.md).

## How the architectures compare

- **Total parameters:** 27,144,960 → 11,458,944, a reduction of 15,686,016
  parameters (57.79%). Parameter counts include embeddings, projections,
  attention, FFNs, and learned LayerNorm parameters, counting shared weights once.
- **Model width:** 384 → 192. Token embeddings, position embeddings, and the
  residual stream all use this narrower dimension.
- **Depth:** 8 → 37 sequential block applications. Each application contains one
  attention sublayer and one FFN, giving 16 → 74 sublayer applications. Reusing a
  sublayer adds computation without allocating another copy of its parameters.
- **Attention heads:** 6 → 3, retaining 64 dimensions per head. Both models use
  causal self-attention, with packed conversations prevented from attending to
  one another.
- **Attention weight sharing:** the baseline has 8 independent attention
  modules. The optimized model cycles through 5 shared attention modules across
  its 35 middle positions, using each 7 times. Independent first and last
  modules bring its total to 7 distinct attention modules.
- **FFN weight sharing:** the baseline has 8 independent FFNs. The optimized
  model cycles through 7 shared middle FFNs, using each 5 times, plus independent
  first and last FFNs: 9 distinct FFNs in total.
- **How the cycles combine:** at zero-based middle position `k`, the optimized
  model uses `Attn (k % 5)` and `FFN (k % 7)`. This visits all 35 attention/FFN
  pairings once. Repeated labels and weight symbols in the figure refer to the
  exact same learned module, applied to a different incoming activation.
- **FFN dimensions:** every baseline FFN is `384 → 1,536 → 384`. Optimized middle
  FFNs are `192 → 1,024 → 192`; its first and last FFNs are
  `192 → 1,536 → 192`. Capacity can increase in individual locations as long as
  the architecture's total parameter count decreases.
- **Activations:** the baseline uses GELU throughout. The optimized model uses
  SiLU in its first FFN and squared ReLU, `ReLU(x)²`, in every other FFN.
- **Sublayer order:** every baseline block runs attention → FFN. The optimized
  model runs FFN → attention except at block 32 (one-indexed), where
  `Attn 0 → FFN 2` reverses that order; the figure highlights this row.
- **Normalization and residual connections:** both use pre-LayerNorm and a final
  LayerNorm. Each shared optimized sublayer also shares its LayerNorm's learned
  scale and bias. Both retain residual additions with branch scale 1.
- **Embeddings and output:** both use the same 32,768-token BPE tokenizer,
  learned position embeddings, and a 1,024-token context. Both tie the vocabulary
  output projection to the input token-embedding matrix and use dropout 0.1.
- **Compute tradeoff:** fewer parameters did not make this model faster to
  train. The recorded training-loop time averaged about 87 seconds for the
  baseline and 141 seconds for the optimized model on one H100 per run,
  excluding preparation and process startup. The deeper execution schedule
  reuses weights but still performs all 37 block applications.

The exact optimized architecture is in
[`spec.json`](experiments/cuda/candidate-c/spec.json). The figure's
[`source data`](figures/architecture_comparison.data.json) contains both models'
dimensions and execution schedules; the
[`renderer`](scripts/render_architecture_comparison.py) generates the PNG, SVG,
and PDF directly from those specifications. Historical artifact paths retain
the `candidate-c` name.

## Fixed experiment setup

- **Data:** the same Stanford Alpaca split throughout: 48,375 training, 1,000 dev
  for architecture/checkpoint selection, and 2,599 held-out conversations.
- **Training budget and batches:** 5,000 optimizer steps from scratch, with 8
  packed rows of 1,024 tokens per step. Tokenizer, answer-only loss, packing,
  and data order within each paired seed are fixed. Paired runs consumed the
  same supervised token counts, approximately 28 million per seed.
- **Optimizer:** AdamW without bias correction, learning rate 0.001, 100 warmup
  steps, cosine decay to 10% of the peak rate, betas `(0.9, 0.95)`, weight decay
  0.1, and gradient clipping at 1.0. Both use BF16 training.
- **Search compute:** four H100s ran independent training jobs in parallel,
  one GPU per run. The completed study contains **616 training runs across 368
  distinct architectures**: 606 search runs and 10 fresh-confirmation runs for
  the original frozen winner and baseline. These totals include repeated seeds;
  they are not 616 distinct architectures. See the
  [search-volume record](experiments/cuda/minparams-search-volume.json).

## Repository layout

The primary checkout is this repository root on branch `main`. It contains the
NVIDIA experiment formerly developed in the `research/nvidia-minparams` worktree.
That redundant worktree has been removed; its branch and the earlier repository
histories are preserved. No GitHub remote is currently configured.

```text
nas/             NVIDIA model, training, search, and evaluation
  cloud/         Lambda access, backups, and watchdog utilities
  mlx/           Original MLX implementation
tests/           Model, scheduler, reporting, and backup tests
scripts/         Environment setup and queue helpers
docs/            CUDA guide and research handoff
experiments/     Architecture specifications and recorded experiments
data/            Fixed tokenizer and dataset splits
```

## Code

| File | Purpose |
|---|---|
| [`cuda_model.py`](nas/cuda_model.py) | Configurable PyTorch transformer and shared modules |
| [`cuda_train.py`](nas/cuda_train.py) | Fixed training recipe, checkpoints, and train/dev curves |
| [`data.py`](nas/data.py) | Fixed data splits, tokenization, and conversation packing |
| [`cuda_minparams_search.py`](nas/cuda_minparams_search.py) | Architecture proposals, scheduling, and run logging |
| [`cuda_evaluate.py`](nas/cuda_evaluate.py) | Checkpoint loading and paired evaluation |
| [`experiments/cuda/candidate-c/metadata.json`](experiments/cuda/candidate-c/metadata.json) | Codex-optimized architecture dimensions, seeds, results, and checkpoint identities |

Run commands from the repository root using Python modules:

```bash
# Configure a Linux CUDA host:
bash scripts/prepare_cuda.sh

# Inspect the training and search interfaces:
.venv-cuda/bin/python -m nas.cuda_train --help
.venv-cuda/bin/python -m nas.cuda_minparams_search --help

# Run CPU verification in an environment with the CUDA requirements and pytest:
python -m pytest -q
```

The older MLX implementation lives in `nas.mlx` and requires MLX; its optional
suite can be run with `python -m pytest -o addopts= tests/test_arch.py`. Cloud
commands use the same module convention, for example
`python -m nas.cloud.lambda_restore_backup --help`.

## Local experiment artifacts

The run logs, architecture registry, curves, reports, and backed-up checkpoints
are under the ignored directory:

```text
.lambda/artifacts/287cdb48d8e64da3b849dceaaf9b3bc2/cuda/minparams-20261111/
```

`CANDIDATE_C_LOCAL_BACKUP.json` lists the Codex-optimized architecture's three
checkpoints and the three paired baseline checkpoints, with verified hashes.
These weights, plus the ten original fresh-confirmation checkpoints, are stored
locally. Credentials and artifacts remain outside Git tracking.

The original experiment checkout is preserved in Git commit `4823477` and in
`frozen-checkout-4823477.tar.gz` in the artifact directory above. Use that snapshot
when reproducing the original source-hash audit. The package reorganization
changes imports and paths; it does not change the recorded experiment results.

See [`HANDOFF.md`](docs/HANDOFF.md) for the current state and complete research history,
and [`CUDA_README.md`](docs/CUDA.md) for the CUDA training workflow.
