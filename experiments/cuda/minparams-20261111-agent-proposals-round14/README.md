# Round 14: final-FFN compression of the confirmed width-192 parent

Six unrun architecture proposals, using only the authorized main-search dev records. No final-test or independent-evaluator artifacts were accessed. All proposals retain width 192, three heads of size 64, the first 36 layer entries, seven 1024-wide shared middle FFNs, the first 1536-wide SiLU FFN, all attention definitions and the accepted entry-32 `Amiddle4` tie from parent `b8886b5f1c19c3d91d011f449c2fbbf410b29343f83d19ebf21501174ed63c17`. Other FFNs use ReLU². Unit residual scales remain fixed; no learned scalar parameters are added.

Vocabulary 32,768, context 1,024, dropout 0.1, data, optimizer, seed protocol, precision, initialization rule and **5,000 training steps** remain fixed. No pretrained weights are transferred. Only the independent final FFN's width, presence, application count or order changes. Every total is strictly below the current 11,144,896-parameter dev-qualified threshold.

| # | Proposal | Parameters | Entry applications | Hypothesis |
|---|---|---:|---:|---|
| 01 | [r14-b888-d192n37-last704](01-r14-b888-d192n37-last704.json) | 11,139,456 | 37 | Mildest eligible cut: change only final hidden width 1536→704. |
| 02 | [r14-b888-d192n37-last512](02-r14-b888-d192n37-last512.json) | 11,065,728 | 37 | Intermediate final-FFN compression; reference for the repeat and order controls. |
| 03 | [r14-b888-d192n37-last256](03-r14-b888-d192n37-last256.json) | 10,967,424 | 37 | Stronger final-FFN compression while retaining a nonlinear branch. |
| 04 | [r14-b888-d192n37-delete-last-ffn](04-r14-b888-d192n37-delete-last-ffn.json) | 10,868,736 | 37 | Remove the final FFN and its norm entirely; retain final attention. |
| 05 | [r14-b888-d192n38-last512-double-ffn](05-r14-b888-d192n38-last512-double-ffn.json) | 11,065,728 | 38 | Apply the same 512-wide final FFN twice before final attention; no new weights. |
| 06 | [r14-b888-d192n37-last512-attention-first](06-r14-b888-d192n37-last512-attention-first.json) | 11,065,728 | 37 | Final attention→FFN instead of FFN→attention; same depth and parameter shapes as 02. |

## Evidence and uncertainty

The confirmed parent has 11,458,944 parameters and dev CE **3.3991814001 / 3.4086931923 / 3.4092634971** across seeds 0/1/2, mean **3.4057126965**, versus baseline mean **3.4207296756**. All three selected checkpoints are at step 5,000; diagnostic train CE is approximately 2.7451 / 2.7424 / 2.7435.

The earlier round7 last-FFN-960 cut (`df535b7f2b68926b975d75011662a211a158b0b9a2246518d478afcd7ee0aa9f`, 11,237,760 parameters) reached seed-0 dev **3.4312595415** and diagnostic train **2.7808923735** at step 4,750. It was worse than this parent and the seed-0 baseline. The matched first-FFN-960 cut (`e025c6254c3dc13b5f911920fb956091b41982a14fc0c8f31801b7b1c4cffd44`) reached dev **3.4263628214**. These are negative screens: the new proposals do not assume that further narrowing monotonically improves loss.

The motivation for isolating final capacity comes from a different width-224 family: round10 last-FFN-864 (`0af98067e38c62d94fe635411af4b020af3b7e5e4ec797eb59750a4d993bf606`) qualified at 11,144,896 parameters with three-seed dev mean **3.4186224016**, while round11 last-FFN-768 (`a69c76c44af2225278354d7ed3d9de0b7d6f9d099a848b02d9ef8591ff677b60`) had a seed-0 screen of **3.4100546221**. Transfer of that behavior to width 192 remains an untested hypothesis. The order/repetition controls provide alternatives if narrowing before final attention is especially harmful.

## Controls and initialization

Proposals 02, 05 and 06 have identical unique parameter counts and shapes. Proposal 06 changes only the final sublayer order, retaining 37 applications. Proposal 05 inserts a natural FFN-only entry before the unchanged final FFN→attention entry, sharing the entire final FFN and norm. It therefore performs **one additional final FFN application**, with 38 entry applications and no additional weights. This is a parameter-matched repetition comparison, not an identical-initial-tensor comparison: the frozen initialization rule uses residual standard deviation `0.02 / sqrt(2 × applications)`, so 05 uses 38 instead of 37 (a factor of `sqrt(37/38)` relative to 02). The extra entry also changes the first-appearance order of the final FFN and attention. No artificial entries are added to other proposals to manipulate initialization depth.

Removing the final FFN in 04 leaves a natural attention-only final entry, so application count stays 37. All shared FFNs, their hidden widths and their schedules remain unchanged in every proposal.

## Runtime and validation

Measured parent full wall times were approximately **170.2 / 145.2 / 145.9 seconds**, with training times 158.3 / 133.2 / 133.6 seconds. The six proposals retain the parent's attention workload and change only one narrow final FFN. Runtime near the parent is therefore an estimate, not a measurement; cold compilation may vary. The existing scheduler runtime guard remains unchanged.

Local schema and unique-parameter checks passed. All six canonical IDs are distinct and absent from **143 prior manual proposals**, breadth candidates and **322 live main-search specifications**. Fresh frozen `SpecGPT` construction on the meta device (PyTorch 2.7.0, CUDA uninitialized) exactly matched every parameter count. Every model has seven unique attention modules and no learned scalars; 04 has eight unique FFNs and the others have nine. Parent fidelity checks verified the first 36 entries, entry-32 attention tie and all non-final FFN definitions. The 512-wide controls have matching parameter shapes. No training, admission, deployment or frozen-source edits were performed.

| # | Canonical architecture SHA-256 |
|---|---|
| 01 | `c6108dd64b4ab53fd580ae78109ac72f3e78cbb06f2b7795ce8b77e470561a73` |
| 02 | `3606491d407c6cfd9bf17699aca016b9f4d0a1028c9be9d8a3bba978fb1658bc` |
| 03 | `f3db1a5b88b3a41deb8ef119f8ba5598f0b1f1d946b2eae3e161167f336bd401` |
| 04 | `8d0b3bacbda078b3f9268d8671be91ea344bda2f945708297cbceb57a0433943` |
| 05 | `b5ec7e62bd8392682bbb799e42dddbb2425d31ca283d0b94371f5c137f06919c` |
| 06 | `c3af18ec312a4fafaa01407aea8148fc8d58ff2ed321171532ace1898ccaf3ea` |
