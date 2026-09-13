"""
pilot/long_v2.py -- two targeted fixes for line-level localization.

Diagnostics from long_hier.py said the failure is SEGMENTATION, not the ranker:
the indentation-based block holding i* averaged 29.8 of 44.2 lines, so "always
pick the largest block" matched the trained block ranker exactly (10/11), and
handing the model the true block changed line Top-1 not at all (27.3% either
way). A block that contains two thirds of the proof is not a constraint.

FIX 1 -- structural segmentation. Start a new unit at every `have`/`calc`/
`obtain`/... at ANY nesting depth, instead of only at minimum indentation.
Reported against a largest-block baseline so we can tell whether any resulting
block-level accuracy is real or is the same triviality as before.

FIX 2 -- learn from the R(j) curve. Today one proof contributes one positive
line. But we stored (j, K, successes) for every probe, so each proof also gives
5-7 graded observations p_hat_j = successes_j / K. Train a regressor
f(line i) -> p_hat at the prefix ending in line i, then score a line by the
DROP it causes, f(i-1) - f(i): the marginal damage of including that line,
which is exactly what i* marks. This also supplies the graded confidence the
current ranker lacks -- its true-line ranks were 1,1,1,4,7,14,... with nothing
at rank 2-3.
"""
import os
import re
import sys
import json
import collections
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "lean_compiler"))
sys.path.insert(0, CURRENT_DIR)

from exp10 import split_body, segment_blocks, block_of_index, strip_header  # noqa: E402
from exp8_train import alltactics_batch  # noqa: E402
from ds_train import features, bvec, gsizes, topk  # noqa: E402
import xgboost as xgb  # noqa: E402

STRUCT_RE = re.compile(r"^\s*(have|calc|obtain|induction|rcases|cases')\b")


def segment_structural(body):
    """A unit begins at every structural tactic, at ANY indentation depth."""
    starts = sorted({0} | {i for i, l in enumerate(body) if STRUCT_RE.match(l)})
    return [list(range(a, b)) for a, b in zip(starts, starts[1:] + [len(body)])]


def softmax(v):
    v = np.asarray(v, float)
    e = np.exp(v - v.max())
    return e / e.sum()


def fit_rank(X, y, g, depth=2, n_est=60):
    o = np.argsort(g, kind="stable")
    m = xgb.XGBRanker(objective="rank:pairwise", max_depth=depth, n_estimators=n_est,
                      learning_rate=0.1, n_jobs=4, tree_method="hist")
    m.fit(X[o], y[o], group=gsizes(g[o]))
    return m


def main():
    lab = json.load(open("pilot/long_labels.json"))
    at = alltactics_batch([(str(i), strip_header(r["code"])) for i, r in enumerate(lab)], 12, 120)

    proofs = []
    for i, r in enumerate(lab):
        code = strip_header(r["code"])
        head, body, abs_lines = split_body(code)
        if not body or not (0 <= r["i_star"] < len(body)):
            continue
        comp = at.get(str(i), {}) or {}
        errs = comp.get("errors") or []
        el = errs[0]["pos"]["line"] if errs and errs[0].get("pos") else None
        old = segment_blocks(body)
        new = segment_structural(body)
        proofs.append({
            "pid": r["pid"], "n": len(body), "i_star": r["i_star"],
            "old": old, "new": new, "blocks": new,   # bvec() reads p["blocks"]
            "old_b": block_of_index(old, r["i_star"]),
            "new_b": block_of_index(new, r["i_star"]),
            "curve": r.get("R_sequence", []),
            "feats": features(body, new, comp.get("tactics", []) or [], el, abs_lines, None),
        })
    COLS = list(proofs[0]["feats"][0].keys())
    N = len(proofs)

    print("=== FIX 1: segmentation ===")
    for nm, key, bkey in [("indentation (old)", "old", "old_b"), ("structural (new)", "new", "new_b")]:
        nb = [len(p[key]) for p in proofs]
        tb = [len(p[key][p[bkey]]) for p in proofs]
        big = sum(1 for p in proofs if int(np.argmax([len(x) for x in p[key]])) == p[bkey])
        print(f"  {nm:<20} {np.mean(nb):5.1f} units/proof | true unit holds "
              f"{np.mean(tb):5.1f} lines ({100*np.mean([t/p['n'] for t,p in zip(tb,proofs)]):.0f}% of proof) "
              f"| largest-unit baseline {big}/{N} ({100*big/N:.0f}%)")

    R = collections.defaultdict(list)
    pids = [p["pid"] for p in proofs]
    for held in range(N):
        tr = [p for j, p in enumerate(proofs) if pids[j] != pids[held]]
        te = proofs[held]

        X, y, g = [], [], []
        BX, BY, BG, LX, LY, LG = [], [], [], [], [], []
        CX, CY = [], []          # curve-regression rows
        for gi, p in enumerate(tr):
            for k, f in enumerate(p["feats"]):
                X.append([f[c] for c in COLS]); y.append(int(k == p["i_star"])); g.append(gi)
            for bi, blk in enumerate(p["new"]):
                fs = [p["feats"][k] for k in blk]
                BX.append(bvec(fs, bi, p)); BY.append(int(bi == p["new_b"])); BG.append(gi)
                for k in blk:
                    LX.append([p["feats"][k][c] for c in COLS])
                    LY.append(int(k == p["i_star"])); LG.append(gi * 1000 + bi)
            # FIX 2: p_hat at prefix length j is attributed to line j-1
            for pr in p["curve"]:
                j, K, s = pr["j"], pr["K"], pr["successes"]
                idx = j - 1
                if 0 <= idx < len(p["feats"]):
                    CX.append([p["feats"][idx][c] for c in COLS])
                    CY.append(s / max(K, 1))

        m_flat = fit_rank(np.array(X, float), np.array(y), np.array(g))
        m_blk = fit_rank(np.array(BX, float), np.array(BY), np.array(BG))
        m_in = fit_rank(np.array(LX, float), np.array(LY), np.array(LG))
        m_curve = xgb.XGBRegressor(max_depth=2, n_estimators=80, learning_rate=0.1,
                                   n_jobs=4, tree_method="hist")
        m_curve.fit(np.array(CX, float), np.array(CY, float))

        Xte = np.array([[f[c] for c in COLS] for f in te["feats"]], float)
        bte = np.array([f["block_id"] for f in te["feats"]])
        s_flat = m_flat.predict(Xte)
        s_in = m_in.predict(Xte)
        s_blk = m_blk.predict(np.array([bvec([te["feats"][k] for k in blk], bi, te)
                                        for bi, blk in enumerate(te["new"])], float))
        bp = softmax(s_blk)
        lp = np.zeros(len(s_in))
        for bi in range(len(te["new"])):
            m = bte == bi
            if m.sum():
                lp[m] = softmax(s_in[m])
        hier = bp[np.clip(bte, 0, len(bp) - 1)] * lp

        # curve model: score a line by the DROP in predicted recoverability
        phat = m_curve.predict(Xte)
        drop = np.empty_like(phat)
        drop[0] = 1.0 - phat[0]
        drop[1:] = phat[:-1] - phat[1:]

        combo = softmax(hier) + softmax(drop)

        for nm, sc in [("flat", s_flat), ("hier", hier), ("curve", drop), ("combo", combo)]:
            R[nm + "1"].append(topk(sc, te["i_star"], 1))
            R[nm + "3"].append(topk(sc, te["i_star"], 3))
        R["blk1"].append(int(np.argmax(bp) == te["new_b"]))
        R["rand1"].append(1.0 / te["n"]); R["rand3"].append(min(3.0 / te["n"], 1.0))
        R["ranks"].append(int(np.where(np.argsort(-hier) == te["i_star"])[0][0]) + 1)

    def row(nm, k1, k3):
        m1 = np.mean(R[k1]); s1 = np.std(R[k1]) / np.sqrt(len(R[k1]))
        print(f"  {nm:<38} Top-1 {m1:6.1%} +/-{s1:5.1%}   Top-3 {np.mean(R[k3]):6.1%}")

    print(f"\n=== LINE-level (n={N}, structural segmentation) ===")
    row("Flat ranker", "flat1", "flat3")
    row("Hierarchical P(B)*P(i|B)", "hier1", "hier3")
    row("FIX 2: R(j)-curve drop model", "curve1", "curve3")
    row("Combo (hier + curve)", "combo1", "combo3")
    row("Random", "rand1", "rand3")
    print(f"\n  block Top-1 (new segmentation): {np.mean(R['blk1']):.1%}")
    print(f"  hierarchical true-line ranks: {sorted(R['ranks'])}")


if __name__ == "__main__":
    main()
