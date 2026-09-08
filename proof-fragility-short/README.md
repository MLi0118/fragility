# Proof fragility on short Lean proofs

Counterfactual supervision for *"which step broke this proof?"* on
DeepSeek-Prover-V2-7B proofs averaging **14.3 lines**.

**Top-1 39.3% · Top-3 55.7%** — n = 61 labels from 43 problems,
leave-one-problem-out, ±6.3% SE.

| Measure | Top-1 | Top-3 |
|---|---|---|
| **Learned ranker** | **39.3%** | **55.7%** |
| Constant predictor (no features) | 14.8% | 36.1% |
| Positional prior | 14.8% | 36.1% |
| Random | 10.8% | 31.6% |
| **Gain over constant** | **+24.6** | **+19.6** |
| Chance-corrected | 32.0% | 35.3% |

Block level (8.2 blocks/proof): ranker **62.3%**, largest-block baseline 49.2%,
random 24.7%. *Block Top-3 is not reported — random already reaches 56.8%.*

---

## Why the constant baseline is reported everywhere

A "constant predictor" ignores its input and always answers the most frequent
`i*` in training. It is the floor a model must clear to show its features carry
information — beating *random* only shows the labels are unevenly distributed.

This matters concretely. A comparison system reports 78.9% Top-1, but its labels
concentrate on step 1, so a constant predictor scores **70.9%** on its corpus:
its model contributes **+8.0**. On this corpus a constant scores 14.8%, so the
model contributes **+24.6**.

**Degeneracy criterion:** if the most frequent label exceeds ~30% of examples,
Top-1 measures label skew rather than localization, and the corpus cannot
support an interpretable result. This corpus passes at 14.8%.

## Results scale with labels

| Labels | Problems | Top-1 | Top-3 | Gain vs constant | SE |
|---|---|---|---|---|---|
| 12 | 12 | 8.3% | — | below baseline | ±8.0 |
| 32 | 14 | 31.2% | 37.5% | +12.4 | ±8.2 |
| **61** | **43** | **39.3%** | **55.7%** | **+24.6** | **±6.3** |

At n=32 Top-3 *lost* to the constant baseline (37.5% vs 46.9%) and was not
reportable. At n=61 it wins by 19.6 points. Effective sample size is the
**problem** count, not the label count, because CV groups by problem.

## Stability

Feature extraction runs the Lean REPL under a timeout; proofs that exceed it
yield no compiler-error position, changing the feature vector.

| REPL timeout | n = 32 | n = 61 |
|---|---|---|
| 120 s | 31.2% | 39.3% |
| 420 s | 18.8% | 37.7% |
| swing | 12.4 pts | **1.6 pts** |

Stable at n=61. Read Top-1 as **37.7–39.3%** depending on timeout.

## Known negative result

Intrinsic accuracy has **not** yet transferred to generation. Cutting at the
model's prediction versus plain restart (3 seeds, early-stopping token
accounting, measured on the n=32 corpus):

| Policy | Solved | vs restart |
|---|---|---|
| Without model (restart) | 86/96 | 1.00× |
| With model | 61/96 | **0.86×** |
| Oracle cut | 83/96 | 1.26× |

A wrong cut commits the prover to a bad prefix and the proof fails outright, so
coverage drops. The oracle shows the headroom is real; the localizer is not yet
accurate enough to capture it. **This test predates the n=61 corpus and should
be rerun.**

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
# 1. Harvest + first labels (~40 min GPU)
python src/exp_ds.py --problems <problems.jsonl> \
    --n_problems 200 --k_harvest 8 --target_labels 80 \
    --harvest_out data/ds_harvest.json --label_out data/ds_labels.json

# 2. More labels from the SAME harvest, no new generation (~35 min GPU).
#    Round-robin across problems so the distinct-problem count stays high.
python src/ds_label_more.py --harvest data/ds_harvest.json \
    --out data/ds_labels_v2.json --max_per_problem 3 --max_candidates 37

# 3. Top-1 and Top-3 (CPU; needs the Lean REPL for proof-state features)
python src/ds_train.py --labels data/ds_labels_all.json

# 4. Compiler/model hybrids + timeout stability (CPU)
python src/short_boost.py --labels data/ds_labels_all.json --timeout 420

# 5. Extrinsic: does USING the model help? 3 seeds (~85 min GPU)
python src/short_pareto.py --labels data/ds_labels_all.json --k 8 --seeds 3

# 6. Optional: run an external 53-feature XGBoost on these labels
python src/peer_model_test.py --labels data/ds_labels_all.json
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
| `data/ds_labels_all.json` | **61 counterfactual labels — the main result** |
| `data/ds_labels_v2.json` | run 1: 32 labels / 14 problems |
| `data/ds_labels_run2.json` | run 2: 29 labels / 29 problems (disjoint) |
| `data/ds_harvest*.json` | raw verified attempts from both harvests |
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
