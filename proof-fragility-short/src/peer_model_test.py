"""
pilot/peer_model_test.py -- run peer B's trained XGBoost on OUR labels.

Their model reports 78.9% Top-1 on their own corpus, where 70.9% of labels sit
at step 1 and a constant "always answer 1" scores 70.9%. This applies their
released model (models/exp1_xgboost.json, 53 named features) to proofs labeled
with our corrected pipeline -- adaptive budget, stability rejection -- using
THEIR own feature extractor and THEIR own step splitter, so the model sees the
input distribution it was trained for.

Three questions:
  1. What is their model's Top-1 against counterfactual labels that are not
     concentrated on step 1?
  2. Is it effectively a constant "step 1" predictor? (fraction of proofs where
     it answers the first step)
  3. How does it compare to our ranker on the same proofs?
"""
import os
import sys
import json
import argparse
import collections
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "lean_compiler"))
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "peer_b", "src"))

from exp10 import split_body, strip_header  # noqa: E402
from exp8_train import alltactics_batch  # noqa: E402
import lean_engine as le  # noqa: E402
import xgboost as xgb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="pilot/ds_labels_v2.json")
    ap.add_argument("--model", default="peer_b/models/exp1_xgboost.json")
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    lab = json.load(open(args.labels))
    at = alltactics_batch([(str(i), strip_header(r["code"])) for i, r in enumerate(lab)],
                          12, args.timeout)

    booster = xgb.Booster()
    booster.load_model(args.model)
    feat_names = booster.feature_names
    print(f"loaded peer model: {len(feat_names)} features")

    rows = []
    for i, r in enumerate(lab):
        code = strip_header(r["code"])
        head, body, abs_lines = split_body(code)
        if not body or not (0 <= r["i_star"] < len(body)):
            continue
        comp = at.get(str(i), {}) or {}
        errs = comp.get("errors") or []
        el = errs[0]["pos"]["line"] if errs and errs[0].get("pos") else None
        emsg = errs[0].get("data", "") if errs else ""

        # our i* indexes BODY LINES; map it onto their step segmentation by
        # finding which of their steps contains that line's text
        steps = le.extract_steps("\n".join(body))
        if not steps:
            continue
        target_line = body[r["i_star"]].strip()
        tgt = None
        for si, s in enumerate(steps):
            if target_line and target_line in s:
                tgt = si
                break
        if tgt is None:
            continue

        feats = [le.extract_step_features(step_text=s, step_idx=si, total_steps=len(steps),
                                          header=head, compiler_error_line=el,
                                          compiler_error_msg=emsg, all_steps=steps)
                 for si, s in enumerate(steps)]
        X = pd.DataFrame(feats)
        X = X[[c for c in feat_names]]          # exact training column order
        p = booster.predict(xgb.DMatrix(X, feature_names=feat_names))
        pred = int(np.argmax(p))
        rows.append({"pid": r["pid"], "n_steps": len(steps), "true": tgt,
                     "pred": pred, "maxp": float(np.max(p))})

    n = len(rows)
    top1 = sum(r["pred"] == r["true"] for r in rows)
    top3 = 0
    print(f"\nevaluated on {n} of {len(lab)} proofs (rest: i* line not matched to a step)")
    print(f"mean steps/proof under THEIR splitter: {np.mean([r['n_steps'] for r in rows]):.1f}")

    pred0 = sum(r["pred"] == 0 for r in rows)
    true0 = sum(r["true"] == 0 for r in rows)
    print("\n" + "=" * 66)
    print("PEER B's MODEL ON OUR COUNTERFACTUAL LABELS")
    print("=" * 66)
    print(f"  Top-1 accuracy              : {top1}/{n} = {100.0*top1/n:.1f}%")
    print(f"  predicts the FIRST step     : {pred0}/{n} = {100.0*pred0/n:.1f}%")
    print(f"  our labels AT first step    : {true0}/{n} = {100.0*true0/n:.1f}%")
    cnt = collections.Counter(r["true"] for r in rows)
    const = cnt.most_common(1)[0]
    print(f"  constant baseline (always {const[0]}) : {100.0*const[1]/n:.1f}%")
    print(f"  random                      : {100.0*np.mean([1.0/r['n_steps'] for r in rows]):.1f}%")
    print(f"  mean max-probability        : {np.mean([r['maxp'] for r in rows]):.3f}")
    pc = collections.Counter(r["pred"] for r in rows)
    print(f"\n  distribution of their predictions: {pc.most_common(6)}")
    print(f"  distribution of our labels      : {cnt.most_common(6)}")


if __name__ == "__main__":
    main()
