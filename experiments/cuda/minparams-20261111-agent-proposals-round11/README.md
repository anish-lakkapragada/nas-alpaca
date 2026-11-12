# Round 11: preserve nonlinear capacity while compressing the frontier

Unrun proposals for review. Evidence comes only from dev records in `/home/ubuntu/nas-minparams/runs/cuda/minparams-20261111`. No independent-evaluator directory, held-out report, test score or test array was accessed.

Confirmed parent `1ccfddeffaaf17682d19d034075788cac1175f6dddb23d2ff1a8a11afa084457`: width 224, heads 7×32, 37 applications, A3/F5 middle schedule, first/middle/last FFN widths 1024/768/1024; **11,216,576 parameters**, three-seed dev mean **3.411584573951**, versus baseline **3.4207296756**.

Promising smaller parent `0af98067e38c62d94fe635411af4b020af3b7e5e4ec797eb59750a4d993bf606` changes only final FFN width to 864: **11,144,896 parameters**, seed 0 dev CE **3.410739852919**, diagnostic train CE **2.746811760454**, best step 5000. This is a single-seed screen, not a qualified result. First864-only was worse at 3.427175608; stronger middle/both-end compression reached 3.42895–3.43558. These results motivate preserving first and repeated middle FFN capacity.

Allocation: 01/02 continue last-only compression. Proposals 03/04 compare complete sharing cycles at the same parameter count as 01, changing depth under the unchanged initialization rule. Proposals 05/06 share only final attention, with/without the promising final FFN reduction. Proposals 07/08 change only head factorization for two smaller width 216 models; their three-head seed 0 dev CE was 3.4214874494 and 3.4235548063, respectively. Head-factorization benefit at width 216 remains a hypothesis.

All models retain paired FFN→attention order, first-FFN SiLU, other FFNs ReLU² and unit fixed scales. Vocabulary 32,768, context 1,024, dropout 0.1, data, optimizer, seed protocol, precision and **5000 training steps** remain fixed. Initialization remains the same rule based on actual application count; no manual scale tuning or trained-weight reuse. Proposals 03/04 execute 32/47 applications; the others execute 37. All learned-scalar counts are zero.

| # | Proposal | Shape | Parameters | Hypothesis |
|---|---|---|---:|---|
| 01 | [r11-d224-h7-n37-last768-only](01-r11-d224-h7-n37-last768-only.json) | d224, 7×32, n37 | 11,101,888 | Last FFN768; first1024 and all middle FFNs768 retained. |
| 02 | [r11-d224-h7-n37-last640-only](02-r11-d224-h7-n37-last640-only.json) | d224, 7×32, n37 | 11,044,544 | Last FFN640; first/middle capacity retained. |
| 03 | [r11-d224-h7-n32-last768-complete-cycles](03-r11-d224-h7-n32-last768-complete-cycles.json) | d224, 7×32, n32 | 11,101,888 | 32 applications; 2 complete middle-pair cycles; same parameters as 01. |
| 04 | [r11-d224-h7-n47-last768-complete-cycles](04-r11-d224-h7-n47-last768-complete-cycles.json) | d224, 7×32, n47 | 11,101,888 | 47 applications; 3 complete middle-pair cycles; same parameters as 01. |
| 05 | [r11-d224-h7-n37-share-final-attention](05-r11-d224-h7-n37-share-final-attention.json) | d224, 7×32, n37 | 11,015,424 | Final attention reuses A2; every FFN width unchanged. |
| 06 | [r11-d224-h7-n37-share-final-attention-last864](06-r11-d224-h7-n37-share-final-attention-last864.json) | d224, 7×32, n37 | 10,943,744 | Final attention reuses A2 plus last FFN864; below11M. |
| 07 | [r11-d216-h9x24-n37-attn4ffn5-middle704](07-r11-d216-h9x24-n37-attn4ffn5-middle704.json) | d216, 9×24, n37 | 10,830,240 | Width216 head-only comparison:9x24 instead of3x72. |
| 08 | [r11-d216-h9x24-n37-attn3ffn5-middle832](08-r11-d216-h9x24-n37-attn3ffn5-middle832.json) | d216, 9×24, n37 | 10,919,664 | Width216 head-only comparison:9x24 instead of3x72. |

## Validation

All eight proposals passed scheduler schema validation and independent frozen SpecGPT construction on the PyTorch 2.7 meta device. Actual unique model counts exactly match the table; depths are 37 except the explicitly labeled 32/47-application pair. All learned-scalar counts are zero. Canonical IDs are unique and absent from 125 prior manual proposal JSONs, breadth candidates and all 304 live main-search specification IDs checked. Every count is strictly below 11,216,576. GPU visibility was disabled and CUDA remained uninitialized. No training, inbox admission, deployment or frozen execution-source edits were performed.

| # | Canonical architecture SHA-256 |
|---|---|
| 01 | `a69c76c44af2225278354d7ed3d9de0b7d6f9d099a848b02d9ef8591ff677b60` |
| 02 | `544b20823e12ca590309e27e68074db2a3db60e6855dbb9a8d3fde92794b6532` |
| 03 | `2ceb8c152ebbef4a1b44005beae826434425393d64c7f0b96fb9455a4d6f7bb3` |
| 04 | `d898dc56bc64f770b5b77235d9a711a08d958c2fe46dbd3f11207d16b3ccc48a` |
| 05 | `eac8762c695bdeb35bee59ca7859d93346381cc3e6044834aeb23ef62bac050f` |
| 06 | `5809102dbe3e069a7624f889037eebfce31660a2e42b9690e951e702bff1d65a` |
| 07 | `03d8df05de74525d27b8bf29c0f9d68c455f9d8adee646e910968568870a9f53` |
| 08 | `fc2f9ffd999820448834c3352059f9267be5793299dcadf5c4e546baed43715a` |
