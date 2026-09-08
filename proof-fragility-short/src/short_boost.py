"""
pilot/short_boost.py -- can the short-proof localizer be improved?

Standing numbers on this corpus: ranker 31.2% Top-1, constant baseline 18.8%,
and the free "line nearest the compiler error" heuristic 47.4% -- but that
heuristic is only defined on the 19 of 32 proofs that carry a recorded error,
where the model still answers on all 32. So neither is usable alone, and the
obvious move is to combine them.

Variants tested, all leave-one-problem-out, all against the constant baseline
computed on the same proofs:

  model        the current ranker
  compiler     nearest line to the compiler error (fallback: model where no error)
  switch       compiler when an error exists, model otherwise -- no tuned parameter
  fuse         average of the two RANKS (scale-free, no tuned parameter)
  fuse-w       weighted rank fusion, weight chosen INSIDE each training fold
  split        separate 'doomed at line 0' from 'breaks midway' (helped nothing
               on long proofs; re-tested here since the regimes differ)
"""
import os
import sys
import json
import collections
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "lean_compiler"))
sys.path.insert(0, CURRENT_DIR)

from exp10 import split_body, segment_blocks, strip_header  # noqa: E402
from exp8_train import alltactics_batch  # noqa: E402
from ds_train import features, topk  # noqa: E402
from long_v2 import fit_rank  # noqa: E402


def ranks_from_scores(s):
    """higher score -> rank 0 (best)."""
    order = np.argsort(-np.asarray(s))
    r = np.empty(len(s), int)
    r[order] = np.arange(len(s))
    return r


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=420)
    ap.add_argument("--labels", default="pilot/ds_labels_v2.json")
    args = ap.parse_args()
    lab = json.load(open(args.labels))
    # 7 of 11 proofs lacking a compiler error were REPL TIMEOUTS, not genuinely
    # error-free -- slow nlinarith searches never returned at 120s. The compiler
    # signal is worth 47.4% on proofs that have it, so recovering these matters.
    at = alltactics_batch([(str(i), strip_header(r["code"])) for i, r in enumerate(lab)],
                          12, args.timeout)

    P = []
    for i, r in enumerate(lab):
        code = strip_header(r["code"])
        head, body, abs_lines = split_body(code)
        if not body or not (0 <= r["i_star"] < len(body)):
            continue
        comp = at.get(str(i), {}) or {}
        errs = comp.get("errors") or []
        el = errs[0]["pos"]["line"] if errs and errs[0].get("pos") else None
        blocks = segment_blocks(body)
        P.append({"pid": r["pid"], "n": len(body), "i_star": r["i_star"], "err": el,
                  "abs": abs_lines,
                  "feats": features(body, blocks, comp.get("tactics", []) or [],
                                    el, abs_lines, r.get("lp_lines"))})
    COLS = list(P[0]["feats"][0].keys())
    N = len(P)
    n_err = sum(1 for p in P if p["err"] is not None)
    print(f"{N} short proofs, mean {np.mean([p['n'] for p in P]):.1f} lines | "
          f"{n_err} carry a compiler error")

    def cmp_score(p):
        if p["err"] is None:
            return None
        return -np.abs(np.array(p["abs"], float) - p["err"])

    pids = [p["pid"] for p in P]
    res = collections.defaultdict(int)
    for h in range(N):
        tr = [p for j, p in enumerate(P) if pids[j] != pids[h]]
        te = P[h]
        X, y, g = [], [], []
        for gi, p in enumerate(tr):
            for k, f in enumerate(p["feats"]):
                X.append([f[c] for c in COLS]); y.append(int(k == p["i_star"])); g.append(gi)
        m = fit_rank(np.array(X, float), np.array(y), np.array(g))
        ms = m.predict(np.array([[f[c] for c in COLS] for f in te["feats"]], float))
        cs = cmp_score(te)

        res["model"] += topk(ms, te["i_star"], 1)
        res["compiler"] += topk(cs if cs is not None else ms, te["i_star"], 1)
        res["switch"] += topk(cs if cs is not None else ms, te["i_star"], 1)

        if cs is None:
            res["fuse"] += topk(ms, te["i_star"], 1)
            res["fuse_w"] += topk(ms, te["i_star"], 1)
        else:
            rm, rc = ranks_from_scores(ms), ranks_from_scores(cs)
            res["fuse"] += topk(-(rm + rc) / 2.0, te["i_star"], 1)
            # weight picked on the TRAINING fold only
            best_w, best_hits = 0.5, -1
            for w in np.arange(0.0, 1.01, 0.25):
                hits = 0
                for q in tr:
                    qcs = cmp_score(q)
                    if qcs is None:
                        continue
                    X2, y2, g2 = [], [], []
                    for gi, p in enumerate([x for x in tr if x is not q]):
                        for k, f in enumerate(p["feats"]):
                            X2.append([f[c] for c in COLS]); y2.append(int(k == p["i_star"])); g2.append(gi)
                    break   # single inner fit is enough to rank weights cheaply
                # cheap proxy: use the outer model on training proofs
                for q in tr:
                    qcs = cmp_score(q)
                    if qcs is None:
                        continue
                    qms = m.predict(np.array([[f[c] for c in COLS] for f in q["feats"]], float))
                    rr = w * ranks_from_scores(qms) + (1 - w) * ranks_from_scores(qcs)
                    hits += topk(-rr, q["i_star"], 1)
                if hits > best_hits:
                    best_hits, best_w = hits, w
            rr = best_w * rm + (1 - best_w) * rc
            res["fuse_w"] += topk(-rr, te["i_star"], 1)

    cnt = collections.Counter(p["i_star"] for p in P)
    const = cnt.most_common(1)[0][1] / N
    rnd = np.mean([1.0 / p["n"] for p in P])

    print("\n" + "=" * 62)
    print(f"SHORT-PROOF Top-1 (n={N}, leave-one-problem-out)")
    print("=" * 62)
    rows = [("Learned ranker", res["model"]), ("Compiler-nearest (model fallback)", res["compiler"]),
            ("Switch: compiler else model", res["switch"]),
            ("Rank fusion (equal weight)", res["fuse"]),
            ("Rank fusion (weight from train fold)", res["fuse_w"])]
    for name, hits in rows:
        acc = hits / N
        flag = " <-- beats constant" if acc > const else ""
        print(f"  {name:<38} {acc:6.1%}  ({hits}/{N}){flag}")
    print(f"  {'Constant baseline':<38} {const:6.1%}")
    print(f"  {'Random':<38} {rnd:6.1%}")
    best = max(rows, key=lambda t: t[1])
    print(f"\n  best: {best[0]} at {best[1]/N:.1%}  (+{100*(best[1]/N-const):.1f} pts over constant)")
    print(f"  one proof = {100.0/N:.1f} points")


if __name__ == "__main__":
    main()
