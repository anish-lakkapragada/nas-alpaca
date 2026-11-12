# Round 10: compress the confirmed seven-head frontier

Unrun architecture proposals for review. Evidence comes only from completed dev records in `/home/ubuntu/nas-minparams/runs/cuda/minparams-20261111`. No independent held-out directory/report or test scores were accessed.

Confirmed parent: `1ccfddeffaaf17682d19d034075788cac1175f6dddb23d2ff1a8a11afa084457` — width 224, heads 7×32, exactly 37 applications, middle A3/F5 banks, FFN widths 768/1024, **11,216,576 parameters**. Seed 0/1/2 dev CE: **3.4084683852 / 3.4157168105 / 3.4105685262**; mean **3.411584573951**, versus baseline **3.4207296756**. Diagnostic train CE: 2.754209 / 2.755601 / 2.740726. All runs completed 5000 steps; selected checkpoints occurred at 4750/5000/5000. These are search-dev results, not held-out claims.

The seven-head parent differs only in head factorization from the original four-head parent. Its seed 0 dev CE improved from 3.4118703302 to 3.4084683852; this motivates compression transfers but does not establish that head count will improve every smaller candidate. The four-head balanced allocation reached 3.4168174410, middle-only compression 3.4209569713, and boundary-only compression 3.4344621075.

Proposals 01–10 preserve the parent A3/F5 schedule exactly and change only FFN widths. Every candidate retains width 224, heads 7×32, exactly 37 paired FFN→attention applications, first-FFN SiLU, remaining FFNs ReLU² and unit fixed scales. Proposal 11 is the explicitly identified A4/F4 phase-cycle transfer control, whose four-head screen improved over aligned sharing. No speculative depth, activation, residual-scale or embedding changes are included.

Vocabulary 32,768, context 1,024, dropout 0.1, common initialization policy, data, optimizer, seed protocol, precision and **5000 training steps** remain fixed. No trained weights or learned scalar values are reused. All learned-scalar counts are zero.

Matched comparisons: 01/02/03 have identical counts and differ in FFN allocation; each changes only heads from its round9 counterpart. Proposals 04/05/06 remove the same total hidden width from the first boundary, final boundary or middle bank. Proposals 09/10 reverse the same asymmetric boundary allocation at identical count and schedule.

| # | Proposal | Parameters | Hypothesis |
|---|---|---:|---|
| 01 | [r10-d224-h7-n37-middle640-ends1024-matched](01-r10-d224-h7-n37-middle640-ends1024-matched.json) | 10,929,856 | Head7 transfer: middle 640 / both boundaries 1024. |
| 02 | [r10-d224-h7-n37-middle768-ends704-matched](02-r10-d224-h7-n37-middle768-ends704-matched.json) | 10,929,856 | Head7 transfer: middle 768 / both boundaries 704. |
| 03 | [r10-d224-h7-n37-middle704-ends864-matched](03-r10-d224-h7-n37-middle704-ends864-matched.json) | 10,929,856 | Head7 transfer: middle 704 / both boundaries 864. |
| 04 | [r10-d224-h7-n37-first864-only](04-r10-d224-h7-n37-first864-only.json) | 11,144,896 | First boundary 864; middle 768 / last 1024. |
| 05 | [r10-d224-h7-n37-last864-only](05-r10-d224-h7-n37-last864-only.json) | 11,144,896 | Last boundary 864; middle 768 / first 1024. |
| 06 | [r10-d224-h7-n37-middle736-ends1024](06-r10-d224-h7-n37-middle736-ends1024.json) | 11,144,896 | Middle 736 / boundaries 1024; matched count to 04/05. |
| 07 | [r10-d224-h7-n37-middle704-ends1024](07-r10-d224-h7-n37-middle704-ends1024.json) | 11,073,216 | Middle 704 / boundaries 1024. |
| 08 | [r10-d224-h7-n37-middle704-ends1088](08-r10-d224-h7-n37-middle704-ends1088.json) | 11,130,560 | Middle 704 / wider boundaries 1088; net parameter reduction. |
| 09 | [r10-d224-h7-n37-middle736-first1152-last768](09-r10-d224-h7-n37-middle736-first1152-last768.json) | 11,087,552 | Middle 736; first 1152 / last 768. |
| 10 | [r10-d224-h7-n37-middle736-first768-last1152](10-r10-d224-h7-n37-middle736-first768-last1152.json) | 11,087,552 | Middle 736; first 768 / last 1152; matched count to 09. |
| 11 | [r10-d224-h7-n37-attn4ffn4-phase-cycle-control](11-r10-d224-h7-n37-attn4ffn4-phase-cycle-control.json) | 10,958,528 | Exception: head7 transfer of A4/F4 phase-cycle schedule. |

## Validation

All 11 proposals passed scheduler schema validation and independent frozen SpecGPT construction on the PyTorch 2.7 meta device. Actual unique model counts exactly match the table. Every model has 37 applications, width 224, heads 7×32 and zero learned scalar parameters. Proposals 01–10 retain the live confirmed parent's layer schedule exactly. Canonical IDs are unique within the batch and absent from all 114 prior manual proposal JSONs, scheduler breadth candidates and all 293 live main-search specification IDs checked. All parameter counts lie between 10,929,856 and 11,144,896. GPU visibility was disabled and CUDA remained uninitialized. No training, admission, deployment, commit or frozen execution-source changes were performed.

| # | Canonical architecture SHA-256 |
|---|---|
| 01 | `05838d45cbdd6ca7234e24ca4e450e12f9ab8728eccad0743c5657fd680ea50a` |
| 02 | `2e5d2e95a1063aef929b7079879f6d3dc8f1dc31a92cf53d7d13f617b7594643` |
| 03 | `67e915d5ec13685e4c5de87dd4a6e1a546661b2b3986cdae0d73bcbce80b086a` |
| 04 | `ed7e0ce5884579971930443610bdbb94749b63de55faa680ca0c06ccde77e578` |
| 05 | `0af98067e38c62d94fe635411af4b020af3b7e5e4ec797eb59750a4d993bf606` |
| 06 | `58c641eca304e8990d5121afb128b80a9457d560e93cdc9d82a8059e8dcfc5bd` |
| 07 | `e0052ae3399682ce5258ec4fbc90d211384a8b50a10697d14eb6b7c3278d3972` |
| 08 | `dae832966a547a339f606265547a12621ef86c2032fe258e161ab9f08b0e75cd` |
| 09 | `0d792f8f395c04909d17605f04d570b172fe5d94d72944ca1a994ae564a6b250` |
| 10 | `7dbb491812ed68ced7535670a136f2613a84c30a816c424dbb937535b9c0f7c6` |
| 11 | `72eabb09d96cf5d2e44254658b8eb036adbf56955d8464c52f197c89bbb0a26c` |
