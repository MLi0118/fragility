# Predict-and-repair inside the ICML proving pipeline

Does the fragility ranker from [`proof-fragility-short`](../proof-fragility-short)
help a full agentic prover? This plugs it into the repair step of
[`eth-sri/optimizing-lean-agents`](https://github.com/eth-sri/optimizing-lean-agents)
and compares it with the authors' own error-feedback repair on 85 PutnamBench
problems.

**Controlled rerun: predict_cut 13/85 · random_cut 13/85 · feedback 12/85.**
Cutting and regenerating is at least as good as error feedback, but the learned
cut does no better than a random one. No difference is significant.

| Arm | Solved / 85 | k=1 | k=4 | k=8 | 1 breakdown, k=8 | Prover tokens | Tokens / solve |
|---|---|---|---|---|---|---|---|
| **predict_cut** | **13** | 11 | 12 | 13 | **7.88** | 10.39M | 799k |
| **random_cut** | **13** | 11 | 13 | 13 | 7.34 | 9.70M | 746k |
| feedback | 12 | 11 | 12 | 12 | 6.66 | 5.16M | **430k** |

k=1/4/8 and "1 breakdown" come from replaying each run in the authors'
`proof_simulation` (fixed waterfall, 32 seeds; it scores 84 of the 85
problems). "1 breakdown" averages over which breakdown is tried.

## Design

- **Pipeline:** their seed prover with all models local (vLLM): gpt-oss-20b
  writes breakdowns, Goedel-Formalizer-V2-8B formalizes, Goedel-Prover-V2-8B
  proves. n_b = 4 breakdowns, n_f = 4 formalizations.
- **Attempt chain:** each lemma/theorem target gets k = 8 attempts, 1 fresh +
  7 repairs (`max_correction_rounds = 7`). k = 4 is a prefix of the same chain.
- **Arms differ only in what a repair does:**

  | Arm | Repair |
  |---|---|
  | `feedback` | theirs: failed proof + Lean errors, model rewrites the proof |
  | `predict_cut` | keep `body[:î*]`, prefill it, regenerate the rest; `î*` from the XGBRanker trained on the 209 labels |
  | `random_cut` | same, with `j ~ Uniform{0..n-1}` (the control that beat the ranker on Lean Workbook) |

- **Shared inputs:** the cut arms copy the feedback arm's breakdowns and
  formalizations. In the controlled rerun (`full85s`) they also replay its
  first attempts, so at k=1 all arms are identical and only repairs differ.
- **Router:** their `cost_quality` router only scores fresh PROVE steps and
  always takes the repair, so with this design it reduces to the fixed
  waterfall. The simulator replay uses the fixed policy for every arm.

## Results

### Controlled rerun (`full85s`, shared first attempts) — headline

- At k=1 all three arms solve the same 11 problems: the sharing worked.
- Repairs add +2 (predict_cut), +2 (random_cut), +1 (feedback).
- predict_cut ⊇ feedback, plus `putnam_1998_b1`.
- predict_cut vs random_cut: `putnam_1968_a2` only predict_cut,
  `putnam_1965_a6` only random_cut.
- No passing proof contains `sorry` or `axiom`.
- Sharing is incomplete in later iterations: targets only a cut arm reached got
  fresh first attempts (journal reuse 112/191 in predict_cut's iteration 1,
  112/164 in random_cut's; 300/300 in iteration 0).

### First run (`full85`, each arm sampled its own first attempts)

| Arm | Solved / 85 | k=1 | k=4 | k=8 | 1 breakdown, k=8 | Prover tokens |
|---|---|---|---|---|---|---|
| predict_cut | 16 | 14 | 16 | 16 | 8.22 | 10.19M |
| feedback | 12 | 11 | 12 | 12 | 6.66 | 5.16M |
| random_cut | 11 | 11 | 11 | 11 | 6.94 | 9.91M |

The 16-vs-12 lead was already there at k=1, i.e. first-attempt sampling luck,
which is why the controlled rerun was done. Solved sets are nested; McNemar
predict_cut vs feedback p = 0.125.

### Token usage

| Arm | Repairs sent | Avg repair tokens (in / out) | Repair prompts dropped |
|---|---|---|---|
| predict_cut (`full85s`) | 2163 | 1961 / 1419 | 11.7% |
| random_cut (`full85s`) | 2103 | 2149 / 1114 | 12.0% |
| feedback | 482 | 3346 / 2236 | 37% |

Their prover silently drops any prompt over `3/4 · max_tokens` = 6144 tokens.
Feedback prompts carry the failed proof plus the errors, so 37% of them are
dropped (and a dropped target leaves its chain for good). Feedback therefore
runs far fewer repairs, which is why it is cheapest per solve; each individual
cut repair is ~40% cheaper.

### Caveats

- 85 problems: a one-problem difference is noise.
- Their assembler fails some combined proofs with
  `unknown identifier '…_lemmaN_fM'` (a sibling lemma referenced by name).
  It affects every arm and was not fixed.
- The router could not be evaluated as intended (see Design).

## Changes to optimizing-lean-agents

`patch/optimizing-lean-agents.diff` (against `64e2736`):

| File | Change |
|---|---|
| `prover/components/prover.py` | `repair_mode` config (`feedback` / `predict_cut` / `random_cut`); cut repairs prefill the kept prefix; records `repair_cut` |
| `prover/query/api.py` | one resident vLLM engine with real release; engine sizing reaches `LLM()`; assistant-prefill rendering; over-length prompts get empty output instead of failing the batch; `PF_VLLM_JOURNAL` output journal (crash resume, and with `PF_VLLM_JOURNAL_ROUND0=1` replay of shared first attempts) |
| `prover/utils.py` | `process_for_correction` returns `[]` when the length filter dropped every prompt (it crashed with IndexError) |
| `seed_prover/components/breakdown_json.py` | strip gpt-oss reasoning channels before `json.loads` (local vLLM returns them raw) |

## Files

| file | role |
|---|---|
| `fragility/repair.py` | `choose_cuts` (ranker / random), `prefill_for`, `train_ranker` (fit on all 209 labels, logprob features zeroed) |
| `fragility/ranker.json(.cols)` | the deployed ranker |
| `fragility/{ds_train,exp10,exp8_train}.py`, `ds_labels_final.json` | features, Lean helpers and the 209 labels, copied from `proof-fragility-short` |
| `configs/hydra/seed_prover/pf_icml.yaml` | pipeline config with local vLLM models |
| `configs/**/models/local/*.yaml` | model configs (prover context 16384, output 8192) |
| `configs/proof_simulation/pf/fixed*_*.yaml` | simulator sweeps: `fixed_*` first run, `fixeds_*` controlled rerun |
| `scripts/pf_icml_run_arms.sh` | first run: feedback, then both cut arms on its breakdowns |
| `scripts/pf_icml_shared.sh` | controlled rerun of the cut arms with shared first attempts |
| `scripts/pf_seed_journal.py` | seeds a journal with a run's round-0 prover outputs |
| `scripts/pf_watchdog.py` | kills a silently hung vLLM run so the launcher retries |
| `scripts/score_arms.py` | solved sets, token totals, simulator tables → `results/scores.json` |
| `results/scores.json` | all five runs: solved problem lists, tokens, full simulator sweeps |
| `results/RESULTS.md` | raw numbers |

## Run

```bash
# inside a clone of optimizing-lean-agents at 64e2736, with mathlib4 built
git apply patch/optimizing-lean-agents.diff
mkdir fragility && cp <this dir>/fragility/* fragility/
cp -r <this dir>/configs/* configs/
# optional: retrain the ranker -> fragility/ranker.json
python fragility/repair.py

# scripts hard-code /space/melanieli22 and SCRATCH paths; edit them first
bash pf_icml_run_arms.sh full85 dataset/putnam_rewrite_solved.jsonl 8 <gpu> <gpu> <gpu>
bash pf_icml_shared.sh full85 full85s dataset/putnam_rewrite_solved.jsonl 8 <gpu> <gpu>

# simulator replay (configs point at $SCRATCH/results/<run>/dump/minified)
python proof_simulation/sweep.py --config configs/proof_simulation/pf/fixeds_predict_cut.yaml
python score_arms.py $SCRATCH/results results/simulations scores.json \
    full85_feedback:pf_fixed_feedback full85s_predict_cut:pf_fixeds_predict_cut ...
```

Wall time for the controlled rerun, one GPU per cut arm: ~5 h (random_cut) and
~7 h (predict_cut).
