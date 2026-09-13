"""
pilot/ds_label_more.py -- more counterfactual labels from the EXISTING harvest.

Stage A is skipped entirely: ds_harvest.json already holds 60 problems x 8
DeepSeek attempts with verification results, so no new generation is needed to
find candidates. The previous run labeled only the FIRST failed attempt per
problem (12 labels). Taking several failed attempts per problem multiplies the
label count for labeling GPU only.

Sibling attempts of one problem are correlated, so CV must group by problem ID
(ds_train.py already does). --max_per_problem keeps any single problem from
dominating the set, and --max_candidates bounds wall-clock.
"""
import os
import sys
import json
import time
import argparse

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "lean_compiler"))
sys.path.insert(0, CURRENT_DIR)

from exp10 import split_body  # noqa: E402
from exp_ds import RawTok, find_tokenizer_json, Gen, MODEL  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest", default="pilot/ds_harvest.json")
    ap.add_argument("--out", default="pilot/ds_labels_v2.json")
    ap.add_argument("--max_per_problem", type=int, default=3)
    ap.add_argument("--max_candidates", type=int, default=32)
    ap.add_argument("--target_labels", type=int, default=28)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--k_gate", type=int, default=8)
    ap.add_argument("--k_confirm", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--num_workers", type=int, default=24)
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    t0 = time.time()
    harvest = json.load(open(args.harvest))

    # Build per-problem queues first, then take ROUND-ROBIN. Draining one
    # problem at a time until the cap is hit would spend the whole budget on a
    # handful of problems -- and since CV groups by problem ID, the effective
    # sample size is the number of distinct problems, not the label count.
    n_sorry = 0
    queues = {}
    for pid, h in harvest.items():
        if h["n_ok"] == 0:
            continue
        q = []
        for a in h["attempts"]:
            if a["ok"] or len(q) >= args.max_per_problem:
                continue
            if "sorry" in a["code"]:
                n_sorry += 1
                continue
            hd, body, _ = split_body(a["code"])
            if not hd or not body or len(body) < 2:
                continue
            q.append({"pid": pid, "stmt": h["stmt"], "body": body, "n": len(body),
                      "lp_lines": a["lp_lines"], "code": a["code"]})
        if q:
            queues[pid] = q

    picks = []
    for depth in range(args.max_per_problem):
        for pid, q in queues.items():
            if depth < len(q) and len(picks) < args.max_candidates:
                picks.append(q[depth])
        if len(picks) >= args.max_candidates:
            break
    n_prob = len({p["pid"] for p in picks})
    print(f"{len(picks)} candidate failed attempts from {n_prob} problems "
          f"(<= {args.max_per_problem} each, {n_sorry} skipped for `sorry`)", flush=True)

    tok = RawTok(find_tokenizer_json())
    from vllm import LLM
    llm = LLM(model=MODEL, tensor_parallel_size=1, max_model_len=4096)
    gen = Gen(llm, tok, args.num_workers, args.timeout)

    rec = {i: {} for i in range(len(picks))}

    def note(i, j, s, k):
        if j not in rec[i] or k > rec[i][j]["K"]:
            rec[i][j] = {"j": j, "K": k, "successes": s}

    def probe(items, k):
        reqs = [(str(i), picks[i]["stmt"], picks[i]["body"][:j]) for i, j in items]
        cand = gen.sample(reqs, k, args.max_tokens, want_logprobs=False)
        return {i: sum(1 for d in cand[str(i)] if d["ok"]) for i, _ in items}

    print(f"R(0) gate at K={args.k_gate} for {len(picks)} candidates...", flush=True)
    got = probe([(i, 0) for i in range(len(picks))], args.k_gate)
    live = {}
    for i in range(len(picks)):
        note(i, 0, got[i], args.k_gate)
        if got[i] > 0 and len(live) < args.target_labels:
            live[i] = {"lo": 0, "hi": picks[i]["n"]}
    print(f"    {len(live)} usable ({time.time()-t0:.0f}s so far)", flush=True)

    rnd = 0
    while True:
        items = [(i, (st["lo"] + st["hi"]) // 2) for i, st in live.items()
                 if st["hi"] - st["lo"] > 1]
        if not items:
            break
        rnd += 1
        print(f"round {rnd}: {len(items)} probes at K={args.k}...", flush=True)
        got = probe(items, args.k)
        for i, j in items:
            note(i, j, got[i], args.k)
            live[i]["lo" if got[i] > 0 else "hi"] = j

    items = [(i, st["lo"]) for i, st in live.items()]
    if items:
        print(f"confirming at K={args.k_confirm}...", flush=True)
        got = probe(items, args.k_confirm)
        for i, j in items:
            note(i, j, got[i], args.k_confirm)

    labels = []
    for i, st in live.items():
        p = picks[i]
        labels.append({"pid": p["pid"], "code": p["code"], "n_lines": p["n"],
                       "i_star": st["hi"] - 1, "lp_lines": p["lp_lines"],
                       "R_sequence": sorted(rec[i].values(), key=lambda d: d["j"])})
    json.dump(labels, open(args.out, "w"), indent=2)
    n_uniq = len({l["pid"] for l in labels})
    print(f"\n{len(labels)} labels from {n_uniq} distinct problems -> {args.out}")
    print(f"{gen.calls} generations, {gen.tokens:,} tokens, {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
