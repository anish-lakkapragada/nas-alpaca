# Round 9: width and sharing comparisons

Unrun proposals for review, informed only by dev results from `/home/ubuntu/nas-minparams/runs/cuda/minparams-20261111`. No held-out/test artifacts or other studies were consulted.

Parent A: `4cb3c50753795ccf56b07cd81ab5c7f173adbd9453159f2f9ff3351acec12ab2` — width 224, heads 4×56, 37 applications, middle attention/FFN banks 3/5, FFN widths 768/1024. It has **11,216,576 parameters**, seed 0 dev CE **3.4118703302**, diagnostic train CE **2.7615653127**, and best step 4750.

Parent B: `ff30f7f6189afb551b27002bfcddc937bbb1ffde569321863097241b80b2c6e5` — the same width, heads, depth and FFN widths, with middle banks 4/4. It has **11,073,216 parameters**, seed 0 dev CE **3.4164891691**, diagnostic train CE **2.7731404081**, and best step 5000. Baseline seed 0 dev CE is **3.4088462352**. Both parents completed all 5000 steps. These are single-seed screens, not qualified results.

Completed round 5 frequency, activation, tied-FFN and reinjection screens did not provide a clear improvement. Attention-rich frequency was closest at dev CE 3.417960; altered activations and wide tied FFNs were worse. This batch therefore retains paired FFN→attention order, first-FFN SiLU, remaining FFNs ReLU² and unit fixed residual scales. Width 176 and 208 screens were less promising than 224; width 216 remains an untested allocation hypothesis.

Every proposal keeps **37 paired applications**, independent boundary modules, vocabulary 32,768, context 1,024, dropout 0.1, the common initialization rule, and the fixed data/optimizer/seed/precision/**5000-step** protocol. No trained weights or learned scalar values are reused. All learned-scalar counts are zero.

Comparisons: 01/02 change only heads from Parent A. Proposals 03/04/05 have identical counts and schedules while redistributing FFN capacity. Proposals 06/07 have identical counts, tensor shapes and first-appearance module order; 07 changes reuse phases to create 16 distinct middle pairs instead of four. Proposals 08–11 use width 216 with compatible heads 3×72, so the width change also changes head factorization. Proposal 12 retains width 224 while funding wider FFNs with fewer unique FFN modules. Ten candidates are below 11M; the two larger candidates are exact head-only controls.

| # | Proposal | Width / heads | Parameters | Hypothesis |
|---|---|---|---:|---|
| 01 | [r9-d224-n37-heads7x32-attn3ffn5-control](01-r9-d224-n37-heads7x32-attn3ffn5-control.json) | 224 / 7×32 | 11,216,576 | Head-only control: 7×32 instead of 4×56. |
| 02 | [r9-d224-n37-heads2x112-attn3ffn5-control](02-r9-d224-n37-heads2x112-attn3ffn5-control.json) | 224 / 2×112 | 11,216,576 | Head-only control: 2×112 instead of 4×56. |
| 03 | [r9-d224-n37-attn3ffn5-middle640-ends1024](03-r9-d224-n37-attn3ffn5-middle640-ends1024.json) | 224 / 4×56 | 10,929,856 | Matched FFN budget: middle 640 / boundaries 1024. |
| 04 | [r9-d224-n37-attn3ffn5-middle768-ends704](04-r9-d224-n37-attn3ffn5-middle768-ends704.json) | 224 / 4×56 | 10,929,856 | Matched FFN budget: middle 768 / boundaries 704. |
| 05 | [r9-d224-n37-attn3ffn5-middle704-ends864](05-r9-d224-n37-attn3ffn5-middle704-ends864.json) | 224 / 4×56 | 10,929,856 | Matched FFN budget: middle 704 / boundaries 864. |
| 06 | [r9-d224-n37-attn4ffn4-middle704-paired](06-r9-d224-n37-attn4ffn4-middle704-paired.json) | 224 / 4×56 | 10,958,528 | Aligned bank control: four fixed middle pairs. |
| 07 | [r9-d224-n37-attn4ffn4-middle704-phase-cycled](07-r9-d224-n37-attn4ffn4-middle704-phase-cycled.json) | 224 / 4×56 | 10,958,528 | Same shapes/count as 06; cycle phases to visit 16 pairs. |
| 08 | [r9-d216-n37-heads3x72-attn3ffn5-middle768](08-r9-d216-n37-heads3x72-attn3ffn5-middle768.json) | 216 / 3×72 | 10,781,424 | Width 216, heads 3×72; retain original FFN widths. |
| 09 | [r9-d216-n37-heads3x72-attn3ffn5-middle832](09-r9-d216-n37-heads3x72-attn3ffn5-middle832.json) | 216 / 3×72 | 10,919,664 | Width 216; reinvest savings in five 832-wide FFNs. |
| 10 | [r9-d216-n37-heads3x72-attn4ffn5-middle704](10-r9-d216-n37-heads3x72-attn4ffn5-middle704.json) | 216 / 3×72 | 10,830,240 | Width 216; four/five banks produce 20 pairs. |
| 11 | [r9-d216-n37-heads3x72-attn3ffn4-middle896](11-r9-d216-n37-heads3x72-attn3ffn4-middle896.json) | 216 / 3×72 | 10,670,400 | Width 216; three/four banks, four 896-wide FFNs. |
| 12 | [r9-d224-n37-heads4x56-attn3ffn4-middle832](12-r9-d224-n37-heads4x56-attn3ffn4-middle832.json) | 224 / 4×56 | 10,986,752 | Width 224 retained; three/four banks, four 832-wide FFNs. |

## Validation

All 12 proposals passed scheduler schema validation and independent frozen SpecGPT construction on the PyTorch 2.7 meta device. Actual unique parameter counts match the table exactly; all depths are 37 and learned-scalar counts are zero. All head dimensions are divisible by eight. The paired/phase-cycled comparison has matching state-dict keys, parameter shapes and first-appearance module order.

Canonical identities are unique within the batch and absent from all 102 prior manual proposal JSONs, the scheduler breadth candidates, and all 281 live server specification IDs checked. Every count is below 11,312,256; minimum 10,670,400 and maximum 11,216,576. GPU visibility was disabled for validation and CUDA remained uninitialized. No deployment, training, scheduler admission, frozen source edits or commits were performed.

| # | Canonical architecture SHA-256 |
|---|---|
| 01 | `1ccfddeffaaf17682d19d034075788cac1175f6dddb23d2ff1a8a11afa084457` |
| 02 | `7212cb6f37aa69b0c36efac2f563989eae17561c340bd9ce97181a73da8f4364` |
| 03 | `fecc5c1dd0a75ffc11d3c452bd875608ac4ed9895029ae8a3a6ec15857bb89ae` |
| 04 | `41c07f08fd7c728f4598dd5ed6624d1ab006e099ec7addecac009414ddabeb46` |
| 05 | `057269c0ce3bc95ff71dde0718e3bc3a454117c808f825a9709d0c7d9e0caf62` |
| 06 | `276da29f2c16b634d61ddf051f552127400cb9351f496616be40c0434b850c38` |
| 07 | `0fd74749d536e838a428b81c030f20f948fb39db705e77fdcea49c7019ae604a` |
| 08 | `21e5ca3468a4b03c0250ebfaf022c4ed1b7f09e0397ed4ca0201b053774c9fb6` |
| 09 | `97e70a9c4b42ecbf817bc48a1d3121d3b761e1bc291e3da43b0032d37b7267dd` |
| 10 | `4bbeeb2e175086b751c6313fa2c3bb8d6c2aa90b7824649bd62a798d8a201c8d` |
| 11 | `d1ef0759ab441e9d57d3c83796dcf4cd1e24da38d8a2b3f2116cfcce5b3331cb` |
| 12 | `d5ebd41fc3a5a9eafecba6300a036532f83355c14f8738abf59c7c4e36f92ed8` |
