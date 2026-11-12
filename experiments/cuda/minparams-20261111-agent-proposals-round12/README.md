# Round 12: width 192, finer head factorization and FFN reinvestment

Unrun proposals in review/admission priority order **B, D, A, C, F, E**. Evidence is restricted to main-search dev records from `/home/ubuntu/nas-minparams/runs/cuda/minparams-20261111`. No held-out/test artifacts or independent-evaluator reports were accessed.

Main structural parent: `1ccfddeffaaf17682d19d034075788cac1175f6dddb23d2ff1a8a11afa084457` — width 224, heads 7×32, 37 applications, cyclic A3/F5 middle bank, first/middle/last FFN widths 1024/768/1024, 11,216,576 parameters. Its seed 0/1/2 dev losses were 3.4084683852 / 3.4157168105 / 3.4105685262, mean **3.411584573951**, versus baseline **3.4207296756**.

The new width 192 family retains that sharing schedule, enlarges the FFNs, and tests heads 3×64 / 6×32 / 8×24. Reducing residual width 224→192 saves 1,081,344 token/position embedding parameters before reallocating capacity. B is the primary head-factorization transfer; D adds FFN capacity; A/C are matched head controls; F adds shared computation; E tests the sub-10M limit. All candidates use independent first/last attention and FFN modules.

All candidates preserve first-FFN SiLU, remaining FFNs ReLU², unit fixed scales, vocabulary 32,768, context 1,024, dropout 0.1 and the common initialization rule. Data, optimizer, seed protocol, precision and **5000 training steps** stay fixed. F has 47 applications; all others have 37. No trained weights or scalar values are reused.

| Order | Design | Heads | Applications | Middle / boundary FFN | Parameters | Hypothesis |
|---|---|---|---:|---|---:|---|
| 01 | [B](01-r12-b-d192-h6x32-n37-attn3ffn5-mid1024-ends1280.json) | 6×32 | 37 | 1024 / 1280 | 10,179,456 | Primary 32-dimensional-head transfer. |
| 02 | [D](02-r12-d-d192-h6x32-n37-attn3ffn5-mid1152-ends1280.json) | 6×32 | 37 | 1152 / 1280 | 10,425,216 | More FFN capacity at the same width/heads. |
| 03 | [A](03-r12-a-d192-h3x64-n37-attn3ffn5-mid1024-ends1280.json) | 3×64 | 37 | 1024 / 1280 | 10,179,456 | Matched three-head reference. |
| 04 | [C](04-r12-c-d192-h8x24-n37-attn3ffn5-mid1024-ends1280.json) | 8×24 | 37 | 1024 / 1280 | 10,179,456 | Matched eight-head alternative. |
| 05 | [F](05-r12-f-d192-h6x32-n47-attn3ffn5-mid1024-ends1280.json) | 6×32 | 47 | 1024 / 1280 | 10,179,456 | Three complete sharing cycles; unchanged unique weights. |
| 06 | [E](06-r12-e-d192-h6x32-n37-attn3ffn5-mid896-ends1280.json) | 6×32 | 37 | 896 / 1280 | 9,933,696 | More aggressive sub-10M compression. |

## Relevant dev evidence

- All 101 accepted width 192 configurations inspected used heads 3×64. No accepted width 192 model used this exact independent-boundary cyclic A3/F5 schedule. The selected head/layout combinations are untried.
- `b8886b5f1c19c3d91d011f449c2fbbf410b29343f83d19ebf21501174ed63c17`: width 192, heads 3×64, A5/F7, 11,458,944 parameters; three-seed mean 3.4057126965. This shows width 192 can support the target at a larger bank, not that compression will preserve it.
- `a2fcd7611de2e4a763ff67d66da830d3506de43a2fc85b5deb0ed5768bce3e24`: width 192, A4/F5, middle 896/ends 1408, 10,179,840 parameters; seed 0 dev 3.4319274619. B reallocates attention/boundary capacity toward wider middle FFNs at nearly the same total count, while changing heads and the sharing period.
- `0980e8b56092084d5906224edbf57d3b2d6dc42481109ac0ed23ca5e2cd4ee81`: width 192, merged nonuniform A3/F5 reuse, 10,032,000 parameters; seed 0 dev 3.4462824431. It has different reuse from the cyclic schedule proposed here.
- At width 224, matched seed 0 dev was3.4084683852 for heads 7×32, 3.4118703302 for 4×56 (`4cb3c50753795ccf56b07cd81ab5c7f173adbd9453159f2f9ff3351acec12ab2`), and 3.4489483771 for 2×112 (`7212cb6f37aa69b0c36efac2f563989eae17561c340bd9ce97181a73da8f4364`). Gains did not consistently transfer to narrower FFN allocations; they remain a hypothesis in this family.

## Runtime estimates — not measured candidate results

Existing width 192 three-head runs took approximately 145–170 seconds wall time for 5000 steps. The width 224 seven-head parent took 209–234 seconds. Head factorization affects runtime despite identical parameter counts.

Planning estimates: A approximately 145–185 seconds; B/D/E approximately 190–260 seconds; C approximately 210–290 seconds; F approximately 235–330 seconds. These are rough allowances inferred from related runs, not benchmarks or promises. The scheduler must use its normal admission guard.

The frozen compute proxy gives B/A/C 0.918×, D 0.957×, E 0.880× and F 1.127× the width 224 winner's arithmetic cost. It does not model head-dependent kernel efficiency. F is approximately 23% more arithmetic than B at exactly the same parameter count; shared weights save storage, while repeated applications still cost compute.

## Validation

All six proposals passed scheduler schema validation and fresh frozen SpecGPT construction on the PyTorch 2.7 meta device. Actual unique parameter counts exactly match the table. Every model has five unique attention modules, seven unique FFNs and zero learned scalar parameters. A/B/C have matching parameter shapes and state-dict keys. Canonical identities match the earlier in-memory analysis and are absent from 133 prior manual proposals, breadth candidates and all 312 live main-search specification IDs checked. GPU visibility was disabled and CUDA remained uninitialized. No training, inbox admission, deployment or frozen execution-source edits were performed.

| Design | Canonical architecture SHA-256 |
|---|---|
| B | `59e7ab5484425fed7ce35e171314950f3937757c609e2e00cba196ab4113488e` |
| D | `f584a178ff8dbecba4333fbadb0f0846511be5a61117640926b8f8de2383607f` |
| A | `695be5d390be09a68ccfc9681a68f6bbe0b4c570c9e7bceed442259783fb0a63` |
| C | `59ed300e1e9b0af13f37dacc2761775d942a8831da98dfeecb9b2efd78dd08c4` |
| F | `233e627880824e4c24b6e0585c5b2d2bab7a0d7b5c254bdfb9c3925e7615f318` |
| E | `9167ced76ff6e8d479dd1a8625308fc991a123eadbd43d903925068e3b92e688` |
