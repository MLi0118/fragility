"""
pilot/ds_train.py -- ranker on DeepSeek counterfactual labels.

Evaluates localization at TWO granularities, because Stage A showed the peer's
reported 21.6% random Top-1 implies ~4.6 candidates while DeepSeek proofs are
actually 17.4 lines (~10.8% random). Their "step" is therefore a coarser unit
than a proof line -- most likely block-like -- so reporting only one
granularity would compare different tasks:

    line  level -- every body line is a candidate (what i* indexes)
    block level -- indentation-defined top-level blocks (peer-comparable)

CV is grouped by PROBLEM ID. Features: position/structure + Lean proof-state
from allTactics + compiler-error distance + generator logprobs. Compiler
location is a feature and never the label.
"""
import os
import re
import sys
import json
import argparse
import collections
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "lean_compiler"))
sys.path.insert(0, CURRENT_DIR)

from exp10 import split_body, segment_blocks, block_of_index, indent_of, strip_header  # noqa: E402
from exp8_train import alltactics_batch, count_hyps, n_goals  # noqa: E402

KINDS = ["have", "nlinarith", "linarith", "field_simp", "ring", "norm_num", "positivity",
         "simp", "rw", "intro", "apply", "exact", "obtain", "refine", "aesop", "omega", "calc"]


def features(body, blocks, tactics, error_line, abs_lines, lp_lines):
    by_line = collections.defaultdict(list)
    for t in tactics:
        pos = t.get("pos") or {}
        if pos.get("line") is not None:
            by_line[pos["line"]].append(t)
    n = len(body)
    rows, prior_have = [], 0
    for k, line in enumerate(body):
        txt = line.strip()
        ts = by_line.get(abs_lines[k], [])
        goals = [t.get("goals", "") or "" for t in ts]
        b = block_of_index(blocks, k)
        blk = blocks[b] if b is not None else [k]
        lp = (lp_lines or {}).get(str(k)) or (lp_lines or {}).get(k)
        mean_lp, min_lp = (lp if isinstance(lp, (list, tuple)) and len(lp) == 2 else (0.0, 0.0))
        nxt_ind = indent_of(body[k + 1]) if k + 1 < n else -1
        r = {
            "line_idx": k,
            # --- adopted from peer_b/src/lean_engine.py extract_step_features:
            # on a 44-line proof, distance from the END and nesting depth carry
            # information that absolute index and is_last do not.
            "steps_from_end": n - 1 - k,
            "is_penultimate": int(k == n - 2),
            "has_nested_proof": int(nxt_ind > indent_of(line)),
            "has_hypothesis_decl": int(" : " in txt and txt.startswith(("have", "obtain", "let"))),
            "is_focus_dot": int(txt.startswith(("·", "."))),
            "bracket_count": txt.count("[") + txt.count("("),
            "is_at_error_line": int(error_line is not None and abs_lines[k] == error_line),
            "is_before_error_line": int(error_line is not None and abs_lines[k] < error_line),
            "is_after_error_line": int(error_line is not None and abs_lines[k] > error_line),
            "has_err_line": int(error_line is not None), "frac_position": k / max(n - 1, 1), "n_lines": n,
            "indent": indent_of(line), "is_first": int(k == 0), "is_last": int(k == n - 1),
            "word_count": len(txt.split()), "char_len": len(txt),
            "prior_have": prior_have, "has_combinator": int("<;>" in txt),
            "block_id": b if b is not None else -1, "n_blocks": len(blocks),
            "block_size": len(blk),
            "idx_in_block": blk.index(k) if k in blk else 0,
            "n_steps_on_line": len(ts),
            "n_goals": max([n_goals(g) for g in goals], default=0),
            "goal_len": max([len(g) for g in goals], default=0),
            "hyp_count": max([count_hyps(g) for g in goals], default=0),
            "signed_err_dist": (abs_lines[k] - error_line) if error_line else 0,
            "abs_err_dist": abs(abs_lines[k] - error_line) if error_line else -1,
            "mean_logprob": float(mean_lp), "min_logprob": float(min_lp),
        }
        for kd in KINDS:
            r[f"is_{kd}"] = int(bool(re.match(rf"^{kd}\b", txt)))
        if txt.startswith("have"):
            prior_have += 1
        rows.append(r)
    for k in range(n):
        nx = rows[min(k + 1, n - 1)]
        rows[k]["d_goals"] = nx["n_goals"] - rows[k]["n_goals"]
        rows[k]["d_goal_len"] = nx["goal_len"] - rows[k]["goal_len"]
        rows[k]["d_hyps"] = nx["hyp_count"] - rows[k]["hyp_count"]
    return rows


def gsizes(g):
    out, cur, c = [], None, 0
    for x in g:
        if x != cur:
            if cur is not None:
                out.append(c)
            cur, c = x, 1
        else:
            c += 1
    out.append(c)
    return out


def topk(scores, true, k):
    return int(true in list(np.argsort(-np.asarray(scores))[:k]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="pilot/ds_labels.json")
    ap.add_argument("--num_workers", type=int, default=12)
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    lab = json.load(open(args.labels))
    print(f"Loaded {len(lab)} DeepSeek counterfactual labels.")

    at = alltactics_batch([(str(i), strip_header(r["code"])) for i, r in enumerate(lab)],
                          args.num_workers, args.timeout)

    proofs = []
    for i, r in enumerate(lab):
        code = strip_header(r["code"])
        head, body, abs_lines = split_body(code)
        if not body or not (0 <= r["i_star"] < len(body)):
            continue
        blocks = segment_blocks(body)
        comp = at.get(str(i), {}) or {}
        errs = comp.get("errors") or []
        eline = errs[0]["pos"]["line"] if errs and errs[0].get("pos") else None
        proofs.append({
            "pid": r["pid"], "n": len(body), "i_star": r["i_star"],
            "blocks": blocks, "i_star_block": block_of_index(blocks, r["i_star"]),
            "feats": features(body, blocks, comp.get("tactics", []) or [],
                              eline, abs_lines, r.get("lp_lines")),
        })
    print(f"Built {len(proofs)} proofs | mean {np.mean([p['n'] for p in proofs]):.1f} lines, "
          f"{np.mean([len(p['blocks']) for p in proofs]):.1f} blocks")

    COLS = list(proofs[0]["feats"][0].keys())
    import xgboost as xgb

    def fit(X, y, g, depth=2, n_est=60):
        o = np.argsort(g, kind="stable")
        m = xgb.XGBRanker(objective="rank:pairwise", max_depth=depth, n_estimators=n_est,
                          learning_rate=0.1, n_jobs=4, tree_method="hist")
        m.fit(X[o], y[o], group=gsizes(g[o]))
        return m

    res = collections.defaultdict(list)
    pids = [p["pid"] for p in proofs]
    for held in range(len(proofs)):
        tr = [p for j, p in enumerate(proofs) if pids[j] != pids[held]]  # group by problem ID
        te = proofs[held]
        if not tr:
            continue

        X, y, g = [], [], []
        BX, BY, BG = [], [], []
        for gi, p in enumerate(tr):
            for k, f in enumerate(p["feats"]):
                X.append([f[c] for c in COLS]); y.append(int(k == p["i_star"])); g.append(gi)
            for bi, blk in enumerate(p["blocks"]):
                fs = [p["feats"][k] for k in blk]
                BX.append(bvec(fs, bi, p)); BY.append(int(bi == p["i_star_block"])); BG.append(gi)
        X, y, g = np.array(X, float), np.array(y), np.array(g)
        m_line = fit(X, y, g)
        m_blk = fit(np.array(BX, float), np.array(BY), np.array(BG))

        Xte = np.array([[f[c] for c in COLS] for f in te["feats"]], float)
        s_line = m_line.predict(Xte)
        s_blk = m_blk.predict(np.array([bvec([te["feats"][k] for k in blk], bi, te)
                                        for bi, blk in enumerate(te["blocks"])], float))

        res["line_top1"].append(topk(s_line, te["i_star"], 1))
        res["line_top3"].append(topk(s_line, te["i_star"], 3))
        res["line_rand1"].append(1.0 / te["n"])
        res["line_rand3"].append(min(3.0 / te["n"], 1.0))
        res["blk_top1"].append(topk(s_blk, te["i_star_block"], 1))
        res["blk_top3"].append(topk(s_blk, te["i_star_block"], 3))
        nb = len(te["blocks"])
        res["blk_rand1"].append(1.0 / nb)
        res["blk_rand3"].append(min(3.0 / nb, 1.0))
        res["pos_top1"].append(topk(-np.arange(te["n"], dtype=float), te["i_star"], 1))
        res["pos_top3"].append(topk(-np.arange(te["n"], dtype=float), te["i_star"], 3))

    n = len(res["line_top1"])
    print("\n" + "=" * 64)
    print(f"DEEPSEEK-PROVER-V2-7B  |  leave-one-problem-out, n = {n}")
    print("=" * 64)

    def row(name, key, rk=None):
        v = res[key]
        m, se = float(np.mean(v)), float(np.std(v) / np.sqrt(len(v)))
        extra = ""
        if rk:
            r = float(np.mean(res[rk]))
            extra = f"   (random {r:5.1%}, chance-corr {(m-r)/(1-r):6.1%})"
        print(f"  {name:<26} {m:6.1%}  +/-{se:5.1%}{extra}")

    print("\nLINE level (i* granularity):")
    row("Ranker Top-1", "line_top1", "line_rand1")
    row("Ranker Top-3", "line_top3", "line_rand3")
    row("Positional prior Top-1", "pos_top1")
    row("Positional prior Top-3", "pos_top3")
    print("\nBLOCK level (peer-comparable granularity):")
    row("Ranker Top-1", "blk_top1", "blk_rand1")
    row("Ranker Top-3", "blk_top3", "blk_rand3")
    print(f"\nn={n}: one flip = {100.0/n:.1f} points.")


def bvec(fs, bi, p):
    G = lambda c, fn: fn([f[c] for f in fs])
    return [bi, bi / max(len(p["blocks"]) - 1, 1), len(fs), p["n"], len(p["blocks"]),
            G("n_goals", max), G("goal_len", max), G("hyp_count", max),
            G("d_goals", min), G("d_goal_len", max), G("abs_err_dist", min),
            G("word_count", sum), G("is_have", sum), G("is_nlinarith", sum),
            G("indent", min), G("mean_logprob", min), G("min_logprob", min),
            G("frac_position", min)]


if __name__ == "__main__":
    main()
