"""
pilot/short_pareto.py -- end-to-end (extrinsic) evaluation on SHORT proofs.

Intrinsic accuracy and downstream usefulness are different questions. On this
corpus the ranker beats the constant predictor 31.2% vs 18.8% Top-1 -- but that
says nothing about whether *using* it produces verified proofs more cheaply.
On long proofs it does not: cutting at the model's prediction solved 22/33 at
6,531 tok/solve versus restart's 28/33 at 4,766 (0.73x). Short proofs are where
the model earns its intrinsic margin, so this is where the extrinsic test
matters.

Arms (identical generations, identical K, early-stopping token accounting):

    restart    j = 0        -- WITHOUT the model: regenerate the whole proof
    predicted  j = model    -- WITH the model: cut where it says
    oracle     j = i*       -- ceiling for any fixed-cut policy

Three seeds, because a single run of this comparison reversed itself on the
long-proof corpus (oracle 0.82x then 2.14x on identical inputs). Per-proof
outcomes are logged so restart and predicted are compared PAIRED.
"""
import os
import sys
import json
import time
import argparse
import collections
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "lean_compiler"))
sys.path.insert(0, CURRENT_DIR)

from exp10 import split_body, strip_header, verify_batch, ok  # noqa: E402
from exp_ds import RawTok, find_tokenizer_json, statement_of, build_prompt, MODEL  # noqa: E402
from exp8_train import alltactics_batch  # noqa: E402
from ds_train import features  # noqa: E402
from long_v2 import fit_rank  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="pilot/ds_labels_v2.json")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--num_workers", type=int, default=24)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--out", default="pilot/short_pareto.json")
    args = ap.parse_args()

    t0 = time.time()
    lab = json.load(open(args.labels))

    tok = RawTok(find_tokenizer_json())
    from vllm import LLM, SamplingParams, TokensPrompt
    llm = LLM(model=MODEL, tensor_parallel_size=1, max_model_len=4096)
    print("model loaded", flush=True)

    at = alltactics_batch([(str(i), strip_header(r["code"])) for i, r in enumerate(lab)],
                          12, args.timeout)
    proofs = []
    for i, r in enumerate(lab):
        stmt = statement_of(r["code"])
        code = strip_header(r["code"])
        head, body, abs_lines = split_body(code)
        if not stmt or not body or not (0 <= r["i_star"] < len(body)):
            continue
        comp = at.get(str(i), {}) or {}
        errs = comp.get("errors") or []
        el = errs[0]["pos"]["line"] if errs and errs[0].get("pos") else None
        proofs.append({"pid": r["pid"], "stmt": stmt, "body": body, "n": len(body),
                       "i_star": r["i_star"],
                       "feats": features(body, [list(range(len(body)))],
                                         comp.get("tactics", []) or [], el, abs_lines, None)})
    N = len(proofs)
    COLS = list(proofs[0]["feats"][0].keys())
    print(f"=== {N} short proofs, mean {np.mean([p['n'] for p in proofs]):.1f} lines ===", flush=True)

    # predicted cut: leave-one-PROBLEM-out, deterministic
    pids = [p["pid"] for p in proofs]
    preds = {}
    for h in range(N):
        tr = [p for j, p in enumerate(proofs) if pids[j] != pids[h]]
        X, y, g = [], [], []
        for gi, p in enumerate(tr):
            for k, f in enumerate(p["feats"]):
                X.append([f[c] for c in COLS]); y.append(int(k == p["i_star"])); g.append(gi)
        m = fit_rank(np.array(X, float), np.array(y), np.array(g))
        preds[h] = int(np.argmax(m.predict(
            np.array([[f[c] for c in COLS] for f in proofs[h]["feats"]], float))))
    acc = np.mean([int(preds[i] == proofs[i]["i_star"]) for i in range(N)])
    print(f"    predicted-cut Top-1 accuracy: {acc:.1%}", flush=True)

    arms = {"restart": {i: 0 for i in range(N)},
            "predicted": preds,
            "oracle": {i: proofs[i]["i_star"] for i in range(N)}}

    rec = collections.defaultdict(dict)
    for s in range(args.seeds):
        prompts, meta = [], []
        for arm, cuts in arms.items():
            for i, j in cuts.items():
                ids = tok.encode(build_prompt(proofs[i]["stmt"], proofs[i]["body"][:j]))
                prompts.append(TokensPrompt(prompt_token_ids=ids))
                meta.append((f"{arm}|{i}", i, j))
        sp = SamplingParams(n=args.k, temperature=1.0, top_p=0.95,
                            max_tokens=args.max_tokens, seed=2000 + s)
        outs = llm.generate(prompts, sp, use_tqdm=False)

        per_sample, tasks = {}, []
        for (key, i, j), o in zip(meta, outs):
            per_sample[key] = [len(c.token_ids) for c in o.outputs]
            pre = ("\n" + "\n".join(proofs[i]["body"][:j])) if j else ""
            for ki, c in enumerate(o.outputs):
                body = tok.decode(list(c.token_ids)).split("```")[0]
                code = proofs[i]["stmt"] + pre + ("\n" if not body.startswith("\n") else "") + body
                tasks.append((f"{key}|{ki}", strip_header(code)))
        res = verify_batch(tasks, args.num_workers, args.timeout)
        for arm in arms:
            for i in range(N):
                key = f"{arm}|{i}"
                nt = per_sample.get(key, [])
                hit = next((ki for ki in range(args.k) if ok(res.get(f"{key}|{ki}"))), None)
                rec[(arm, s)][i] = (int(hit is not None),
                                    sum(nt) if hit is None else sum(nt[: hit + 1]))
        print(f"  seed {s} done ({(time.time()-t0)/60:.1f} min)", flush=True)

    print("\n" + "=" * 74)
    print(f"WITH vs WITHOUT THE MODEL -- short proofs (n={N}, K={args.k}, early stop)")
    print("=" * 74)
    print(f"  {'seed':<6}{'without (restart)':>26}{'with (predicted)':>26}{'ratio':>12}")
    ratios = []
    for s in range(args.seeds):
        cells, vals = "", {}
        for arm in ["restart", "predicted"]:
            sol = sum(v[0] for v in rec[(arm, s)].values())
            tk = sum(v[1] for v in rec[(arm, s)].values())
            ps = tk / sol if sol else float("inf")
            vals[arm] = ps
            cells += f"{sol:>4}/{N} {ps:>11,.0f} tok/solve"
        r = vals["restart"] / vals["predicted"] if vals["predicted"] not in (0, float("inf")) else float("nan")
        ratios.append(r)
        print(f"  {s:<6}{cells}{r:>11.2f}x")
    ratios = np.array(ratios, float)
    print(f"\n  with-model vs without: mean {np.nanmean(ratios):.2f}x  "
          f"min {np.nanmin(ratios):.2f}x  max {np.nanmax(ratios):.2f}x  sd {np.nanstd(ratios):.2f}")
    w = int(np.sum(ratios > 1.0))
    print(f"  seeds where the model HELPS: {w}/{args.seeds}")

    for arm in ["oracle"]:
        sol = sum(v[0] for s in range(args.seeds) for v in rec[(arm, s)].values())
        tk = sum(v[1] for s in range(args.seeds) for v in rec[(arm, s)].values())
        base_s = sum(v[0] for s in range(args.seeds) for v in rec[("restart", s)].values())
        base_t = sum(v[1] for s in range(args.seeds) for v in rec[("restart", s)].values())
        print(f"  oracle ceiling: {sol}/{args.seeds*N} @ {tk/max(sol,1):,.0f} tok/solve "
              f"({(base_t/max(base_s,1))/(tk/max(sol,1)):.2f}x vs restart)")

    wins = losses = ties = 0
    for s in range(args.seeds):
        for i in range(N):
            rp, tp = rec[("predicted", s)][i]
            rr, tr_ = rec[("restart", s)][i]
            if rp and not rr:
                wins += 1
            elif rr and not rp:
                losses += 1
            elif rp and rr:
                wins += int(tp < tr_); losses += int(tp > tr_); ties += int(tp == tr_)
            else:
                ties += 1
    print(f"\n  PAIRED per-proof ({wins+losses+ties} comparisons):")
    print(f"    with-model better {wins}  |  without better {losses}  |  tie/both-fail {ties}")
    print(f"\n  wall clock {(time.time()-t0)/60:.1f} min")

    json.dump({str(k): v for k, v in rec.items()}, open(args.out, "w"), default=str)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
