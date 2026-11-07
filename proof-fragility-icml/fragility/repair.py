"""
fragility/repair.py -- cut-and-regenerate repair for the ICML pipeline's
correction rounds.

Their correction round shows the model its failed proof plus the compiler
error and asks for a full rewrite. Here a failed proof is instead cut after
its first j body lines, the kept prefix is prefilled as the start of the
assistant turn, and the model regenerates the rest:

    predict_cut  j = ranker's predicted break line i*  (ds_train.py features)
    random_cut   j ~ Uniform{0..n-1}  -- the control that beat the ranker on
                 Lean Workbook (90.0% vs 83.7%, p = 0.001)

The router is untouched: these modes only change what a correction attempt
does, never whether one happens.

Logprob features are zeroed at train and test time. The pipeline does not
record generator logprobs, and a feature that is always 0 at test time must be
0 in training too or the ranker scores off-distribution inputs.
"""
import os
import re
import sys
import json
import random
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from exp10 import split_body, segment_blocks, strip_header  # noqa: E402
from exp8_train import alltactics_batch  # noqa: E402
from ds_train import features  # noqa: E402

LABELS = os.path.join(CURRENT_DIR, "ds_labels_final.json")
MODEL_PATH = os.path.join(CURRENT_DIR, "ranker.json")
CUT_MODES = ("predict_cut", "random_cut")

_BY_RE = re.compile(r":=\s*by\b")
_DECL_RE = re.compile(r"^\s*(?:theorem|lemma)\b", re.MULTILINE)


def split_target(code):
    """(head through the target's ':= by', body lines, 1-indexed line numbers).

    Theorem attempts carry the lemmas as axioms ahead of the theorem, and the
    model may add helper lemmas, so the target is the LAST theorem/lemma
    declaration and its first ':= by' is the boundary (the same regex that
    fixed the Putnam splitter)."""
    decls = list(_DECL_RE.finditer(code or ""))
    if not decls:
        return None, [], []
    m = _BY_RE.search(code, decls[-1].start())
    if not m:
        return None, [], []
    head = code[:m.end()]
    head_line = head.count("\n") + 1
    body, abs_lines = [], []
    for off, ln in enumerate(code[m.end():].split("\n")):
        if ln.strip():
            body.append(ln)
            abs_lines.append(head_line + off)
    return head, body, abs_lines


def _rows(body, comp, abs_lines):
    errs = comp.get("errors") or []
    eline = errs[0]["pos"]["line"] if errs and isinstance(errs[0], dict) and errs[0].get("pos") else None
    return features(body, segment_blocks(body), comp.get("tactics", []) or [], eline, abs_lines, None)


def train_ranker(workers=16, timeout=120):
    """Fit the i* ranker on all 209 counterfactual labels (no CV: this is the
    deployed model; ds_train.py holds the leave-one-problem-out evaluation)."""
    import xgboost as xgb
    lab = json.load(open(LABELS))
    at = alltactics_batch([(str(i), strip_header(r["code"])) for i, r in enumerate(lab)], workers, timeout)
    X, y, g, cols = [], [], [], None
    for i, r in enumerate(lab):
        _, body, abs_lines = split_body(strip_header(r["code"]))
        if not body or not (0 <= r["i_star"] < len(body)):
            continue
        rows = _rows(body, at.get(str(i), {}) or {}, abs_lines)
        cols = cols or list(rows[0].keys())
        for k, f in enumerate(rows):
            X.append([f[c] for c in cols])
            y.append(int(k == r["i_star"]))
            g.append(i)
    m = xgb.XGBRanker(objective="rank:pairwise", max_depth=2, n_estimators=60,
                      learning_rate=0.1, n_jobs=4, tree_method="hist")
    _, sizes = np.unique(np.array(g), return_counts=True)  # g is already grouped in order
    m.fit(np.array(X, float), np.array(y), group=sizes)
    m.save_model(MODEL_PATH)
    json.dump(cols, open(MODEL_PATH + ".cols", "w"))
    print(f"ranker trained on {len(sizes)} proofs, {len(X)} lines -> {MODEL_PATH}")


def _load_ranker():
    import xgboost as xgb
    m = xgb.XGBRanker()
    m.load_model(MODEL_PATH)
    return m, json.load(open(MODEL_PATH + ".cols"))


def choose_cuts(items, mode, round_num, workers=16, timeout=120):
    """{uid: (head, body, j)} for every item whose failed code has a body.
    Items missing from the result fall back to j = 0 (full regeneration)."""
    parsed = {}
    for it in items:
        head, body, abs_lines = split_target(it.get("compiled_code_that_failed_in_prev_round", ""))
        if head and body:
            parsed[it["uid"]] = (head, body, abs_lines, it["compiled_code_that_failed_in_prev_round"])
    cuts = {}
    if mode == "random_cut":
        for uid, (head, body, _, _) in parsed.items():
            cuts[uid] = (head, body, random.Random(f"{uid}|{round_num}").randrange(len(body)))
        return cuts
    model, cols = _load_ranker()
    # The correction round only has the error list; proof states need an allTactics pass
    at = alltactics_batch([(uid, p[3]) for uid, p in parsed.items()], workers, timeout)
    for uid, (head, body, abs_lines, _) in parsed.items():
        rows = _rows(body, at.get(uid, {}) or {}, abs_lines)
        s = model.predict(np.array([[f[c] for c in cols] for f in rows], float))
        cuts[uid] = (head, body, int(np.argmax(s)))
    return cuts


def prefill_for(head, body, j):
    """Same shape as exp10.build_prefill: open the lean4 block, restate the
    target, keep body[:j]."""
    kept = "\n".join(body[:j])
    return "```lean4\n" + head + ("\n" + kept if kept else "")


if __name__ == "__main__":
    train_ranker()
