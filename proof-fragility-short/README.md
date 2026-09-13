# Proof fragility on short Lean proofs

Counterfactual supervision for *"which step broke this proof?"* on
DeepSeek-Prover-V2-7B proofs averaging **13.9 lines**.

**Top-1 27.8% · Top-3 42.1%** — n = 209 labels from 191 problems,
leave-one-problem-out, ±3.1% SE, one proof = 0.5 points.

| Measure | Top-1 | Top-3 |
|---|---|---|
| **Learned ranker** | **27.8%** | **42.1%** |
| Constant predictor (no features) | 11.5% | 31.6% |
| Positional prior | 11.5% | 29.2% |
| Random | 10.1% | 30.0% |
| **Gain over constant** | **+16.3** | **+10.5** |
| Chance-corrected | 19.6% | 17.2% |

Proof length: mean **13.9 lines**, median 12 (p25 8, p75 17, max 89).

Block level (7.4 blocks/proof): ranker **67.9%**, random 25.2%. *Provisional —
the largest-block baseline (49.2% at n=61) has not been recomputed at n=209.
Block Top-3 is not reported: random already reaches 55.4%.*

---

## Why the constant baseline is reported everywhere

A "constant predictor" ignores its input and always answers the most frequent
`i*` in training. It is the floor a model must clear to show its features carry
information — beating *random* only shows the labels are unevenly distributed.

This matters concretely. A comparison system reports 78.9% Top-1, but its labels
concentrate on step 1, so a constant predictor scores **70.9%** on its corpus:
its model contributes **+8.0**. On this corpus a constant scores 11.5%, so the
model contributes **+16.3**.

**Degeneracy criterion:** if the most frequent label exceeds ~30% of examples,
Top-1 measures label skew rather than localization, and the corpus cannot
support an interpretable result. This corpus passes at 11.5%.

## What scaling the corpus did

| Labels | Problems | Top-1 | Top-3 | Gain vs constant | SE |
|---|---|---|---|---|---|
| 12 | 12 | 8.3% | — | below baseline | ±8.0 |
| 32 | 14 | 31.2% | 37.5% | +12.4 | ±8.2 |
| 61 | 43 | 39.3% | 55.7% | +24.6 | ±6.3 |
| **209** | **191** | **27.8%** | **42.1%** | **+16.3** | **±3.1** |

**Top-1 peaked at n=61 and then fell.** That figure was an optimistic
small-sample draw; 27.8% is the number to trust. What survived is the *margin
over baseline*, which stays far outside the error bars from n=32 on.

The task also got harder as the corpus grew: the constant baseline fell
18.8% → 14.8% → 11.5%, and mean `i*` position rose to **0.478 of the proof**,
so labels are spread through the proof rather than clustered near its opening.
Part of the decline reflects a less degenerate task, not only regression.

Effective sample size is the **problem** count, not the label count, because CV
groups by problem.

## Stability

Feature extraction runs the Lean REPL under a timeout; proofs that exceed it
yield no compiler-error position, changing the feature vector.

| REPL timeout | n = 32 | n = 61 |
|---|---|---|
| 120 s | 31.2% | 39.3% |
| 420 s | 18.8% | 37.7% |
| swing | 12.4 pts | **1.6 pts** |

Stable at n=61. **Not re-tested at n=209.**

## Known negative result

Intrinsic accuracy has **not** yet transferred to generation. Cutting at the
model's prediction versus plain restart (3 seeds, early-stopping token
accounting, measured on the n=32 corpus; a rerun at n=209 is in progress):

| Policy | Solved | vs restart |
|---|---|---|
| Without model (restart) | 86/96 | 1.00× |
| With model | 61/96 | **0.86×** |
| Oracle cut | 83/96 | 1.26× |

A wrong cut commits the prover to a bad prefix and the proof fails outright, so
coverage drops. The oracle shows the headroom is real; the localizer is not yet
accurate enough to capture it. **This test was measured at n=32; a rerun on the
209-label corpus is in progress and this section will change.**

## Requires

Place `src/` inside a clone of [`eth-sri/optimizing-lean-agents`](https://github.com/eth-sri/optimizing-lean-agents).
These import `repl_scheduler` from its `lean_compiler/`, and need a built
`mathlib4/` beside it.

**Always run from the repo root, not from the script directory** — `repl_scheduler`
resolves `mathlib4/` as a *relative* path, and every REPL worker dies otherwise.

```
pip install vllm xgboost transformers tokenizers pandas numpy scikit-learn
```

## Run order

```bash
# 1. Harvest + first labels (~2 h GPU for 1000 problems)
python src/exp_ds.py --problems <problems.jsonl> \
    --n_problems 1000 --k_harvest 8 --target_labels 250 \
    --timeout 60 --num_workers 48 --gpu_mem_util 0.70 \
    --harvest_out data/ds_harvest.json --label_out data/ds_labels.json

# 2. More labels from the SAME harvest, no new generation (~35 min GPU).
#    Round-robin across problems so the distinct-problem count stays high.
python src/ds_label_more.py --harvest data/ds_harvest.json \
    --out data/ds_labels_v2.json --max_per_problem 3 --max_candidates 40

# 3. Top-1 and Top-3 (CPU; needs the Lean REPL for proof-state features)
python src/ds_train.py --labels data/ds_labels_final.json

# 4. Compiler/model hybrids + timeout stability (CPU)
python src/short_boost.py --labels data/ds_labels_final.json --timeout 420

# 5. Extrinsic: does USING the model help? 3 seeds (~85 min GPU)
python src/short_pareto.py --labels data/ds_labels_final.json \
    --k 8 --seeds 3 --max_proofs 120 --gpu_mem_util 0.65

# 6. Optional: run an external 53-feature XGBoost on these labels
python src/peer_model_test.py --labels data/ds_labels_final.json
```

## Files

| file | role |
|---|---|
| `src/exp_ds.py` | harvest + labeling; DeepSeek tokenizer fix, `R(j)` binary search |
| `src/ds_label_more.py` | more labels from an existing harvest, round-robin by problem |
| `src/ds_train.py` | 52 features + `XGBRanker`, LOO by problem, Top-1/Top-3 vs baselines |
| `src/short_boost.py` | compiler/model hybrids, constant baseline, timeout stability |
| `src/short_pareto.py` | with/without-model generation, 3 seeds, paired per-proof |
| `src/peer_model_test.py` | applies an external 53-feature XGBoost to these labels |
| `src/exp10.py` | shared: `split_body`, `strip_header`, REPL worker pool, prefill prompts |
| `src/exp8_train.py` | shared: `allTactics` runner, goal/hypothesis parsing |
| `src/long_v2.py` | shared: `fit_rank`, structural segmentation |
| `data/ds_labels_final.json` | **209 counterfactual labels — the main result** |
| `data/ds_labels_all.json` | runs 1+2 combined: 61 labels / 43 problems |
| `data/ds_labels_1000.json` | run 3: 148 labels / 148 problems (disjoint) |
| `data/ds_labels_v2.json`, `ds_labels_run2.json` | runs 1 and 2 separately |
| `data/ds_harvest*.json` | raw verified attempts (run 3 harvest is 20 MB, not committed) |
| `data/short_pareto.json` | per-proof outcomes from the extrinsic test |

## Method

**Labels are counterfactual, not compiler-derived.** `R(j) = 1` if any of K
regenerated suffixes from prefix `j` verifies; `i*` is the first `j` where
`R(j) = 0`. Binary search finds it in ~log2(n) probes. K=8 on the `R(0)` gate,
K=4 during search, K=8 to re-confirm the boundary. The compiler error location
is an input **feature** and never a label — deriving labels from it makes the
task partly circular.

**Adaptive token budget.** Regeneration is capped at
`clamp(2 × tokens(remaining suffix), 256, 2048)`. A fixed cap makes `R(j)`
measure *"recoverable within the budget"* rather than *"recoverable"*, driving
`R(j)=0` at small `j` and pinning `i*` near the start of the proof.

**Tokenizer.** This checkpoint's `transformers.AutoTokenizer` strips **all**
whitespace on encode (`"  have h : 0 < a"` → `"haveh:0<a"`). `exp_ds.RawTok`
loads `tokenizers.Tokenizer` from `tokenizer.json` and passes `prompt_token_ids`
to vLLM directly, with a round-trip assertion at startup. If that assertion
fires, nothing downstream is trustworthy.

**`sorry` attempts are dropped** — recoverability is trivially zero for reasons
unrelated to fragility.

**Features (52):** position and structure, tactic type, Lean proof-state from
`allTactics` (goal count, goal size, hypothesis count and deltas),
compiler-error distance, generator log-probabilities.

## Scope

This does **not** extend to long proofs. On a 41.7-line Goedel-Prover corpus the
same pipeline scores 17.9% Top-1 against a 28.6% constant baseline — worse than
constant — and 0/20 on the subset with a genuine mid-proof failure. Even oracle
localization loses to restart there (0.76×, 3 seeds), so no fixed-cut branching
policy can help in that regime.
