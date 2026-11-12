# Round 7: compress the improved late-sharing schedule

Prepared from the authorized CUDA study only. These are unrun proposals for review; no training or scheduler admission was performed.

Parent: `b8886b5f1c19c3d91d011f449c2fbbf410b29343f83d19ebf21501174ed63c17` (11,458,944 parameters). Its search-seed 0/1/2 dev CE is 3.3991814001 / 3.4086931923 / 3.4092634971, mean **3.4057126965**, versus baseline mean **3.4207296756**. All three best checkpoints occurred at step 5,000; diagnostic train CE was 2.745113 / 2.742435 / 2.743494. These are dev-selection results, not held-out claims.

The ancestor `2a2d83ddf6b455cf1fce2a602a5370baf199a793a15266d9834652a8d2e00d31` has the same weights and widths. The improved parent changes only zero-based entry 32 from `Amiddle1` to `Amiddle4`, repeating the `Fmiddle3`/`Amiddle4` pairing at entry 25. This does not establish what either module learned.

Every proposal keeps width 192, three heads of size 64, **37 paired FFN→attention applications**, first-FFN SiLU and remaining FFNs ReLU², fixed unit residual scales, and the unchanged initialization rule. No new scalar parameters. Vocabulary 32,768, context 1,024, dropout 0.1, data, optimizer, seed protocol, and **5,000 steps** remain fixed. Unique shared parameters are counted once.

Priority: 01–05 directly test retaining the measured gain under capacity reductions; 06–09 trade unique attention/FFN weights for retained or enlarged FFNs; 10–11 are matched schedule comparisons against 04; 12 is one aggressive approximately 10M hypothesis. Proposals 02/03 have matched counts and opposite boundary allocations.

| # | Proposal | Parameters | Hypothesis |
|---|---|---:|---|
| 01 | [r7-b888-d192n37-middle960](01-r7-b888-d192n37-middle960.json) | 11,286,912 | Exact parent schedule; seven middle FFNs 1024→960. |
| 02 | [r7-b888-d192n37-first960-last1536](02-r7-b888-d192n37-first960-last1536.json) | 11,237,760 | First FFN 1536→960; middle widths and final FFN retained. |
| 03 | [r7-b888-d192n37-first1536-last960](03-r7-b888-d192n37-first1536-last960.json) | 11,237,760 | Last FFN 1536→960; matched count to first-only compression. |
| 04 | [r7-b888-d192n37-middle960-ends1408](04-r7-b888-d192n37-middle960-ends1408.json) | 11,188,608 | Middle 960/boundaries 1408; capacity control for schedule comparisons. |
| 05 | [r7-b888-d192n37-wide-repeated-pair-ffn](05-r7-b888-d192n37-wide-repeated-pair-ffn.json) | 11,213,184 | Fmiddle3=1152, six other middle FFNs=896; ends 1536. |
| 06 | [r7-b888-d192n37-merge-attention1-into4](06-r7-b888-d192n37-merge-attention1-into4.json) | 11,212,800 | Four middle attention modules, seven middle FFNs; boundaries 1408. |
| 07 | [r7-b888-d192n37-share-boundary-attention](07-r7-b888-d192n37-share-boundary-attention.json) | 11,163,264 | Both boundary attentions reuse middle weights; all FFN widths unchanged. |
| 08 | [r7-b888-d192n37-merge-ffn6-grow3](08-r7-b888-d192n37-merge-ffn6-grow3.json) | 11,114,496 | Six middle FFNs; merge F6→F3, grow F3 to 1152. |
| 09 | [r7-b888-d192n37-merge-two-ffns-grow3](09-r7-b888-d192n37-merge-two-ffns-grow3.json) | 10,770,048 | Five middle FFNs; merge F0/F6→F3, grow F3 to 1280. |
| 10 | [r7-b888-d192n37-repeat-pair-every-sweep](10-r7-b888-d192n37-repeat-pair-every-sweep.json) | 11,188,608 | Middle 960/ends 1408; all five F3 applications paired with A4. |
| 11 | [r7-b888-d192n37-repeat-penultimate-pairs](11-r7-b888-d192n37-repeat-penultimate-pairs.json) | 11,188,608 | Middle 960/ends 1408; final seven attention IDs repeat previous seven. |
| 12 | [r7-b888-d192n37-three-attention-five-ffn](12-r7-b888-d192n37-three-attention-five-ffn.json) | 10,032,000 | Three middle attention/five middle FFN modules; middle 896/ends 1408. |

## Identity and validation

CPU scheduler schema validation and independent PyTorch 2.7 model construction on the meta device passed for all 12 proposals. Actual unique model parameter counts exactly match the scheduler counts in the table. Canonical identities are unique within the batch and absent from all 82 prior manual proposal JSONs, the scheduler breadth candidates, and all 261 live server specification IDs checked. Every total is strictly below 11,312,256; minimum 10,032,000, maximum 11,286,912. Every model has 37 applications and zero learned scalar parameters. GPU visibility was disabled for validation, and CUDA remained uninitialized. No training, source edits, scheduler admission, or deployment was performed.

| # | Canonical architecture SHA-256 |
|---|---|
| 01 | `24f45e944e29ffb2cdfbe14882b0c564d731ffaab07469b85bb43b4dddb7db80` |
| 02 | `e025c6254c3dc13b5f911920fb956091b41982a14fc0c8f31801b7b1c4cffd44` |
| 03 | `df535b7f2b68926b975d75011662a211a158b0b9a2246518d478afcd7ee0aa9f` |
| 04 | `99c5d2fdddc66045354a71b6e0a65afb42651703ba3ad2962625c3d07b653b56` |
| 05 | `d3d1f3f856cdf972d93de1cadfb87dc6cd4de70d69d985bce69db842df2c510f` |
| 06 | `f3f3f7e499758d58249abf6afbaebe5c10c4c0123bcc087625af4b16b4bdac64` |
| 07 | `f6e3e15232e15455872a95dcbd8036d1289a4e0a6c78d0152273a212dae45df0` |
| 08 | `6165c19dfa763c37951ce5a3530a119bb8bc5018469e88362dbd07dbf3c509d3` |
| 09 | `954fc4a9948c836eda5dbc0843dd7116834b723b80074241f706670f2c030450` |
| 10 | `eabf7c1882a361b58591f6a5852d02df5e28a0c1067e09ac4b030343e24ec6cc` |
| 11 | `753c6930d424e47b42134c42e177a5965afbaf577b51837ecba3c4b6eb257e9b` |
| 12 | `0980e8b56092084d5906224edbf57d3b2d6dc42481109ac0ed23ca5e2cd4ee81` |
