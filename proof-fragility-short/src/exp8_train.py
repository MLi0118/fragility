"""
pilot/exp8_train.py -- hierarchical fragility ranker on REAL counterfactual labels.

Reads exp20_results.json (real i* per attempt), re-runs allTactics on each
labeled attempt to obtain Lean proof-state at every step, builds line-level
features, and evaluates:

    P(i) = P(B_i) * P(i | B_i)

with leave-one-proof-out CV. At n~8 proofs LOO is the only defensible protocol,
and even then the estimate is rough -- baselines are computed on the SAME
proofs so the comparison is at least internally valid.

Two notes on method:

* Candidates are BODY LINES, not allTactics sub-steps. i* is line-granular
  (a line is the unit you can actually repair), and the peer's ~4.6 candidates
  are line-like too, so line level is the comparable granularity.

* dist_to_error_line is included as a FEATURE here. In the old pipeline that
  was leakage, because the label itself was defined as "nearest step to the
  compiler error". With counterfactual labels the error location is
  independent evidence, so it is legitimate -- and the earlier leak is exactly
  why it has to be re-justified rather than silently reused.
"""
import os
import re
import sys
import json
import argparse
import collections
import multiprocessing as mp
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "lean_compiler"))
sys.path.insert(0, CURRENT_DIR)

import repl_scheduler as rs  # noqa: E402
from exp10 import split_body, segment_blocks, block_of_index, indent_of, strip_header  # noqa: E402

TACTIC_KINDS = ["have", "nlinarith", "linarith", "field_simp", "ring", "norm_num",
                "positivity", "simp", "rw", "calc", "intro", "apply", "exact",
                "constructor", "cases", "obtain", "refine", "aesop", "omega"]


# ---------------------------------------------------------------- allTactics

def _at_worker(wid, q, out, timeout):
    child, _ = rs.initiate_child()
    while True:
        item = q.get()
        if item is None:
            break
        resp = rs.send_command_and_wait(child, item["code"], env=0, allTactics=True,
                                        timeout=timeout, imports=rs.DEFAULT_IMPORTS)
        resp["name"] = item["name"]
        out.append(resp)
        if resp["compilation_result"].get("system_errors") is not None:
            try:
                rs.kill_repl_tree(child)
            except Exception:
                pass
            child, _ = rs.initiate_child()
    try:
        rs.kill_repl_tree(child)
    except Exception:
        pass


def alltactics_batch(tasks, num_workers=12, timeout=120):
    if not tasks:
        return {}
    mgr = mp.Manager()
    q, out = mgr.Queue(), mgr.list()
    for name, code in tasks:
        q.put({"name": name, "code": code})
    n = min(num_workers, len(tasks))
    for _ in range(n):
        q.put(None)
    ps = [mp.Process(target=_at_worker, args=(i, q, out, timeout)) for i in range(n)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    return {r["name"]: r["compilation_result"] for r in out}


def count_hyps(goal):
    return sum(1 for ln in goal.split("\n") if " : " in ln and not ln.strip().startswith("⊢"))


def n_goals(goal):
    return goal.count("⊢")


# ---------------------------------------------------------------- features

def build_line_features(body, blocks, tactics, error_line):
    """One feature row per body line. `tactics` are allTactics entries whose
    pos/endPos line numbers are 1-indexed into the STRIPPED code."""
    n = len(body)
    # map stripped-code line number -> body index
    line_of_body = {}
    stripped_body_lines = {}
    # tactics carry absolute line numbers in stripped code; body[k] sits at
    # stripped line (offset + k') -- recover by matching order of non-blank lines
    # after ":= by" (done by caller via `body_line_no`).
    by_line = collections.defaultdict(list)
    for t in tactics:
        pos = t.get("pos") or {}
        ln = pos.get("line")
        if ln is not None:
            by_line[ln].append(t)

    rows = []
    prior_have = 0
    for k, line in enumerate(body):
        b = block_of_index(blocks, k)
        blk = blocks[b] if b is not None else [k]
        txt = line.strip()
        ts = by_line.get(body_line_no(k), [])
        goals = [t.get("goals", "") or "" for t in ts]
        ng = max([n_goals(g) for g in goals], default=0)
        gl = max([len(g) for g in goals], default=0)
        hc = max([count_hyps(g) for g in goals], default=0)

        row = {
            "line_idx": k,
            "frac_position": k / max(n - 1, 1),
            "n_lines": n,
            "indent": indent_of(line),
            "is_first": int(k == 0),
            "is_last": int(k == n - 1),
            "word_count": len(txt.split()),
            "char_len": len(txt),
            "prior_have_count": prior_have,
            "has_combinator": int("<;>" in txt),
            "block_id": b if b is not None else -1,
            "n_blocks": len(blocks),
            "block_size": len(blk),
            "idx_in_block": blk.index(k) if k in blk else 0,
            "frac_in_block": (blk.index(k) / max(len(blk) - 1, 1)) if k in blk else 0.0,
            "block_frac_position": (b / max(len(blocks) - 1, 1)) if b is not None else 0.0,
            "n_steps_on_line": len(ts),
            "n_goals": ng,
            "goal_len": gl,
            "hyp_count": hc,
            "dist_to_error_line": (body_line_no(k) - error_line) if error_line else 0,
            "abs_dist_to_error": abs(body_line_no(k) - error_line) if error_line else -1,
        }
        for kind in TACTIC_KINDS:
            row[f"is_{kind}"] = int(bool(re.match(rf"^{kind}\b", txt)))
        if txt.startswith("have"):
            prior_have += 1
        rows.append(row)

    # deltas in proof state between consecutive lines
    for k in range(len(rows)):
        nxt = rows[k + 1] if k + 1 < len(rows) else rows[k]
        rows[k]["delta_goals"] = nxt["n_goals"] - rows[k]["n_goals"]
        rows[k]["delta_goal_len"] = nxt["goal_len"] - rows[k]["goal_len"]
        rows[k]["delta_hyps"] = nxt["hyp_count"] - rows[k]["hyp_count"]
    return rows


# body_line_no is rebound per-attempt by the caller (closure over abs_lines)
def body_line_no(k):
    return _BODY_LINES[k]


_BODY_LINES = []


# ---------------------------------------------------------------- evaluation

def topk_from_scores(scores, true_idx, k):
    return int(true_idx in list(np.argsort(-scores)[:k]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="pilot/exp20_results.json")
    ap.add_argument("--minified", required=True)
    ap.add_argument("--num_workers", type=int, default=12)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    lab = json.load(open(args.labels))["results"]
    lab = [r for r in lab if r.get("i_star") is not None]
    data = json.load(open(args.minified))
    print(f"Loaded {len(lab)} counterfactually-labeled attempts.")

    tasks = []
    for r in lab:
        code = data[r["idx"]]["full_code"]
        tasks.append((str(r["idx"]), strip_header(code)))
    print("Running allTactics for proof-state features...", flush=True)
    at = alltactics_batch(tasks, args.num_workers, args.timeout)

    global _BODY_LINES
    proofs = []
    for r in lab:
        code = strip_header(data[r["idx"]]["full_code"])
        head, body, abs_lines = split_body(code)
        if not body:
            continue
        blocks = segment_blocks(body)
        comp = at.get(str(r["idx"]), {}) or {}
        tactics = comp.get("tactics", []) or []
        errs = comp.get("errors") or []
        error_line = errs[0]["pos"]["line"] if errs and errs[0].get("pos") else None
        _BODY_LINES = abs_lines
        feats = build_line_features(body, blocks, tactics, error_line)
        i_star = r["i_star"]
        if not (0 <= i_star < len(body)):
            continue
        proofs.append({"pid": r["pid"], "feats": feats, "blocks": blocks,
                       "i_star": i_star, "n": len(body),
                       "i_star_block": block_of_index(blocks, i_star),
                       "proxy_body_idx": r.get("proxy_body_idx"),
                       "n_tactics": len(tactics)})
    print(f"Built features for {len(proofs)} proofs "
          f"(mean {np.mean([p['n'] for p in proofs]):.1f} lines, "
          f"{np.mean([len(p['blocks']) for p in proofs]):.1f} blocks).")

    FEATS = [k for k in proofs[0]["feats"][0].keys()]
    import xgboost as xgb

    def mat(ps):
        X, y, g, blk = [], [], [], []
        for gi, p in enumerate(ps):
            for k, f in enumerate(p["feats"]):
                X.append([f[c] for c in FEATS])
                y.append(int(k == p["i_star"]))
                g.append(gi)
                blk.append(f["block_id"])
        return np.array(X, float), np.array(y), np.array(g), np.array(blk)

    res = collections.defaultdict(list)
    for held in range(len(proofs)):
        train = [p for i, p in enumerate(proofs) if i != held]
        test = proofs[held]
        Xtr, ytr, gtr, btr = mat(train)

        # --- level 1: which block?
        Bx, By, Bg = [], [], []
        for gi, p in enumerate(train):
            for bi, blk in enumerate(p["blocks"]):
                fs = [p["feats"][k] for k in blk]
                Bx.append(block_vec(fs, bi, p))
                By.append(int(bi == p["i_star_block"]))
                Bg.append(gi)
        Bx, By, Bg = np.array(Bx, float), np.array(By), np.array(Bg)
        bm = xgb.XGBRanker(objective="rank:pairwise", max_depth=2, n_estimators=60,
                           learning_rate=0.1, n_jobs=4, tree_method="hist")
        order = np.argsort(Bg, kind="stable")
        bm.fit(Bx[order], By[order], group=group_sizes(Bg[order]))

        # --- level 2: which line within a block?
        lm = xgb.XGBRanker(objective="rank:pairwise", max_depth=2, n_estimators=60,
                           learning_rate=0.1, n_jobs=4, tree_method="hist")
        key = gtr * 1000 + btr
        o2 = np.argsort(key, kind="stable")
        lm.fit(Xtr[o2], ytr[o2], group=group_sizes(key[o2]))

        # --- flat baseline (no hierarchy)
        fm = xgb.XGBRanker(objective="rank:pairwise", max_depth=2, n_estimators=60,
                           learning_rate=0.1, n_jobs=4, tree_method="hist")
        o3 = np.argsort(gtr, kind="stable")
        fm.fit(Xtr[o3], ytr[o3], group=group_sizes(gtr[o3]))

        # --- score the held-out proof
        Xte = np.array([[f[c] for c in FEATS] for f in test["feats"]], float)
        bte = np.array([f["block_id"] for f in test["feats"]])
        bscore = bm.predict(np.array([block_vec([test["feats"][k] for k in blk], bi, test)
                                      for bi, blk in enumerate(test["blocks"])], float))
        bprob = softmax(bscore)
        lscore = lm.predict(Xte)
        lprob = np.zeros(len(lscore))
        for bi in range(len(test["blocks"])):
            m = bte == bi
            if m.sum():
                lprob[m] = softmax(lscore[m])
        hier = bprob[np.clip(bte, 0, len(bprob) - 1)] * lprob
        flat = fm.predict(Xte)

        ti = test["i_star"]
        for name, sc in [("hierarchical", hier), ("flat", flat)]:
            res[f"{name}_top1"].append(topk_from_scores(sc, ti, 1))
            res[f"{name}_top3"].append(topk_from_scores(sc, ti, 3))
        # baselines
        res["random_top1"].append(1.0 / test["n"])
        res["random_top3"].append(min(3.0 / test["n"], 1.0))
        pos = -np.arange(test["n"], dtype=float)      # positional prior: earlier = likelier
        res["position_top1"].append(topk_from_scores(pos, ti, 1))
        res["position_top3"].append(topk_from_scores(pos, ti, 3))
        if test["proxy_body_idx"] is not None:
            res["proxy_top1"].append(int(test["proxy_body_idx"] == ti))
        # block-level oracle ceiling
        res["blockonly_top1"].append(int(np.argmax(bprob) == test["i_star_block"]))

    n = len(proofs)
    print("\n" + "=" * 62)
    print(f"LEAVE-ONE-OUT RESULTS  (n = {n} proofs)")
    print("=" * 62)

    def rep(name, key):
        v = res.get(key)
        if not v:
            return
        m = float(np.mean(v))
        se = float(np.std(v) / np.sqrt(len(v))) if len(v) > 1 else 0.0
        print(f"  {name:<34} {m:6.1%}   +/- {se:5.1%} (SE)")

    print("\nStep(line)-level localization:")
    rep("Hierarchical  P(B)*P(i|B)  Top-1", "hierarchical_top1")
    rep("Hierarchical  P(B)*P(i|B)  Top-3", "hierarchical_top3")
    rep("Flat ranker                Top-1", "flat_top1")
    rep("Flat ranker                Top-3", "flat_top3")
    print("\nBaselines (same proofs):")
    rep("Random                     Top-1", "random_top1")
    rep("Random                     Top-3", "random_top3")
    rep("Positional prior           Top-1", "position_top1")
    rep("Positional prior           Top-3", "position_top3")
    rep("Compiler proxy             Top-1", "proxy_top1")
    print("\nBlock level:")
    rep("Block ranker               Top-1", "blockonly_top1")

    h1, r1 = np.mean(res["hierarchical_top1"]), np.mean(res["random_top1"])
    h3, r3 = np.mean(res["hierarchical_top3"]), np.mean(res["random_top3"])
    print("\nChance-corrected (acc-rand)/(1-rand):")
    print(f"  Top-1: {(h1-r1)/(1-r1):.1%}")
    print(f"  Top-3: {(h3-r3)/(1-r3):.1%}")
    print(f"\nNOTE: n={n}. SE above is the leave-one-out standard error; with this "
          f"few proofs a single flip moves Top-1 by {100.0/n:.0f} points.")


def block_vec(fs, bi, p):
    g = lambda c, fn: fn([f[c] for f in fs])
    return [bi, bi / max(len(p["blocks"]) - 1, 1), len(fs), p["n"], len(p["blocks"]),
            g("n_goals", max), g("goal_len", max), g("hyp_count", max),
            g("delta_goals", min), g("delta_goal_len", max),
            g("abs_dist_to_error", min), g("word_count", sum),
            g("is_have", sum), g("is_nlinarith", sum), g("is_field_simp", sum),
            g("indent", min), g("n_steps_on_line", sum), g("frac_position", min)]


def group_sizes(g):
    sizes, cur, cnt = [], None, 0
    for x in g:
        if x != cur:
            if cur is not None:
                sizes.append(cnt)
            cur, cnt = x, 1
        else:
            cnt += 1
    sizes.append(cnt)
    return sizes


def softmax(v):
    v = np.asarray(v, float)
    if v.size == 0:
        return v
    e = np.exp(v - v.max())
    return e / e.sum()


if __name__ == "__main__":
    main()
