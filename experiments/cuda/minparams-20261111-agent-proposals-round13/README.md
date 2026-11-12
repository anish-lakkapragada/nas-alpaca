# Round 13: seven-FFN banks at widths 192 and 208

Four unrun architecture proposals for review. Only main-search dev records in `/home/ubuntu/nas-minparams/runs/cuda/minparams-20261111` informed this batch. No held-out/test artifacts or independent-evaluator reports were accessed.

Every model uses **37 applications**, three cyclic middle attention modules and seven cyclic middle FFNs (**21 distinct middle pairs**), plus independent first/last attention and FFN modules. First FFN uses SiLU, other FFNs use ReLU², and all residual scales are fixed at one. Vocabulary 32,768, context 1,024, dropout 0.1, common initialization, data, optimizer, seed protocol, precision and **5000 training steps** remain fixed. No pretrained weights or scalar values are reused.

| # | Proposal | Heads | Middle / boundary FFN | Parameters | Comparison |
|---|---|---|---|---:|---|
| 01 | [r13-d192-h3x64-n37-attn3ffn7-mid1024-ends1408](01-r13-d192-h3x64-n37-attn3ffn7-mid1024-ends1408.json) | 3×64 | 1024 / 1408 | 11,064,960 | Width192 capacity-first seven-bank candidate. |
| 02 | [r13-d192-h3x64-n37-attn3ffn7-mid896-ends1408](02-r13-d192-h3x64-n37-attn3ffn7-mid896-ends1408.json) | 3×64 | 896 / 1408 | 10,720,896 | Only middle FFN width changes from01. |
| 03 | [r13-d208-h4x52-n37-attn3ffn7-mid768-ends1024](03-r13-d208-h4x52-n37-attn3ffn7-mid768-ends1024.json) | 4×52 | 768 / 1024 | 10,988,640 | Width208 reference; four52-dimensional heads. |
| 04 | [r13-d208-h8x26-n37-attn3ffn7-mid768-ends1024](04-r13-d208-h8x26-n37-attn3ffn7-mid768-ends1024.json) | 8×26 | 768 / 1024 | 10,988,640 | Only head count changes from03; eight26-dimensional heads. |

## Measured dev evidence

- `59e7ab5484425fed7ce35e171314950f3937757c609e2e00cba196ab4113488e` (round12B): width192, heads6×32, A3/F5, 10,179,456 parameters; seed0 dev **3.456402851568**, diagnostic train **2.844627356776**, wall209.0s. The remaining round12 candidates were pending at selection time. This poor screen lowers confidence in transferring the narrow-head configuration directly to width192.
- `b8886b5f1c19c3d91d011f449c2fbbf410b29343f83d19ebf21501174ed63c17`: width192, heads3×64, A5/F7, 11,458,944 parameters; three-seed dev mean **3.4057126965**. Proposals01/02 retain seven middle FFNs and the established head factorization while reducing the unique attention bank.
- `ea186450fd8b232a88f4a4ce903f91d5b98fda0b3ec95d9fb142112d13702630`: the only width208 architecture previously tried, heads2×104, A4/F5, FFN896/1280, 11,001,536 parameters; seed0 dev **3.425313952892**, wall173.8s. Proposals03/04 increase FFN-bank diversity at a similar total budget and provide a matched moderate-head comparison.
- `f6e3e15232e15455872a95dcbd8036d1289a4e0a6c78d0152273a212dae45df0` has the same total unique-module counts as A3/F7, but reuses boundary attention within an A5/F7 middle schedule. Its seed0 dev was3.4327406376. It is structurally different from the independent-boundary cyclic schedule here.
- `a69c76c44af2225278354d7ed3d9de0b7d6f9d099a848b02d9ef8591ff677b60` (round11 last768): 11,101,888 parameters, seed0 dev **3.410054622084**, train **2.742868152376**. Other round11 results were still running/queued at inspection; this batch explores module diversity rather than extrapolating final-FFN cuts.

No candidate loss or runtime is predicted as a measured result. Width208 heads4×52 and8×26 satisfy residual-width divisibility and frozen-model schema requirements. Their CUDA attention throughput is unmeasured; meta validation establishes model structure and counts, not backend performance. The existing scheduler admission/runtime guard remains unchanged.

## Validation

Local schema/count validation passed. The four canonical IDs match the preceding in-memory analysis and are absent from 139 prior manual proposal JSONs, breadth candidates and 318 live main-search specifications. Fresh frozen `SpecGPT` meta-device counts exactly match the table for all four proposals, using PyTorch 2.7.0 with CUDA uninitialized. Every model has five unique attention modules, nine unique FFNs, 37 applications and no learned residual scalars. Proposals 01/02 have identical schedules; proposals 03/04 have matching schedules, FFN definitions, state-dict keys and parameter shapes, differing only in head count. No training, admission, deployment or frozen-source edits were performed.

| # | Canonical architecture SHA-256 |
|---|---|
| 01 | `905fa4e3e3d67ce49838b0db043124bdcad1cf6af3e43616e771294bd265aa02` |
| 02 | `149e9a14e4fe631068b53811b4da9770efb70acae8feb3e51411fc3f3dab6457` |
| 03 | `e9758c8d136ddf1ebf7f5eff8e1fbd311329e0169e8c3dc63e4945f9ad508ade` |
| 04 | `912e36c73c1914bdb49f3b23e660360ab405c144421659f1650acb7eed514cff` |
