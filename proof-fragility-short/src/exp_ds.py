"""
pilot/exp_ds.py -- DeepSeek-Prover-V2-7B primary head-to-head experiment.

Switches the generator to the peer's model so the candidate-length confound
disappears: Goedel writes ~26-37 line proofs (~30 candidates), DeepSeek non-CoT
writes short ones (~5 expected). That removes the single largest source of the
raw-accuracy gap, at the cost of needing a fresh harvest -- every attempt we
have so far is Goedel's.

Two DeepSeek-specific things this handles:

  * TOKENIZER. transformers' AutoTokenizer for this checkpoint strips ALL
    whitespace on encode ("  have h : 0 < a" -> "haveh:0<a"). vLLM uses it, so
    prompts arrive whitespace-free and the model faithfully continues in kind.
    Fix: encode with the raw `tokenizers.Tokenizer` from tokenizer.json and
    feed vLLM prompt_token_ids; decode outputs from token_ids the same way.

  * PROMPT. DeepSeek-Prover-V2 non-CoT is a raw completion model, not a chat
    model, so no assistant prefill is needed. Prefix conditioning is native:
    put the retained prefix inside the ```lean4 block and let it continue.

Stages (each writes JSON so later stages can be re-run without GPU):
  A harvest   N problems x K samples, verify, find solvable problems
  B label     counterfactual R(j) via binary search on failed attempts
  C train     line features + XGBRanker, CV grouped by PROBLEM ID
"""
import os
import re
import sys
import json
import time
import argparse
import collections
import multiprocessing as mp

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "lean_compiler"))
sys.path.insert(0, CURRENT_DIR)

import repl_scheduler as rs  # noqa: E402
from exp10 import split_body, indent_of, strip_header, verify_batch, ok  # noqa: E402

MODEL = "deepseek-ai/DeepSeek-Prover-V2-7B"


# ------------------------------------------------------------------ tokenizer

def byte_decoder():
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


class RawTok:
    """Byte-level tokenizer that actually round-trips for this checkpoint."""

    def __init__(self, path):
        from tokenizers import Tokenizer
        self.t = Tokenizer.from_file(path)
        self.bd = byte_decoder()

    def encode(self, s):
        return self.t.encode(s, add_special_tokens=False).ids

    def decode(self, ids):
        return self.t.decode(list(ids))

    def token_bytes(self, tid):
        s = self.t.id_to_token(tid)
        if s is None:
            return b""
        try:
            return bytes(self.bd[ch] for ch in s)
        except KeyError:
            return b""


def find_tokenizer_json():
    import glob
    pats = os.path.expanduser(
        "~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-Prover-V2-7B/snapshots/*/tokenizer.json")
    hits = glob.glob(pats)
    if not hits:
        hits = glob.glob(os.path.join(
            os.environ.get("HF_HOME", ""), "hub",
            "models--deepseek-ai--DeepSeek-Prover-V2-7B", "snapshots", "*", "tokenizer.json"))
    if not hits:
        raise FileNotFoundError("tokenizer.json for DeepSeek-Prover-V2-7B not found in HF cache")
    return hits[0]


# ------------------------------------------------------------------ prompting

def statement_of(lean4_code):
    """Theorem statement through ':= by' (DeepSeek non-CoT completion target)."""
    i = lean4_code.find(":= by")
    return None if i < 0 else lean4_code[: i + len(":= by")]


def build_prompt(stmt_header, prefix_lines):
    body = ("\n" + "\n".join(prefix_lines)) if prefix_lines else ""
    return "Complete the following Lean 4 code:\n\n```lean4\n" + stmt_header + body


def logprobs_by_line(tok, token_ids, logprobs):
    """Mean/min logprob per generated LINE. Newline is always its own byte in
    byte-level BPE, so counting 0x0A in each token's bytes gives exact line
    boundaries without re-decoding prefixes."""
    per_line, cur = collections.defaultdict(list), 0
    for tid, lp in zip(token_ids, logprobs):
        per_line[cur].append(lp)
        cur += tok.token_bytes(tid).count(b"\n")
    return {k: (sum(v) / len(v), min(v)) for k, v in per_line.items() if v}


# ------------------------------------------------------------------ generate

class Gen:
    def __init__(self, llm, tok, workers, timeout):
        self.llm, self.tok = llm, tok
        self.workers, self.timeout = workers, timeout
        self.calls = self.tokens = 0

    def _verify_chunked(self, tasks, chunk=600):
        """verify_batch routes every result through ONE multiprocessing.Manager
        proxy. At ~1.6k tasks that is fine; at 8k the manager becomes a
        serialization bottleneck and throughput collapses to a few results per
        minute while timed-out REPL children orphan and accumulate. Chunking
        keeps each Manager short-lived, surfaces progress, and lets orphans be
        reaped between chunks."""
        import subprocess, time as _t
        out = {}
        n = len(tasks)
        for i in range(0, n, chunk):
            part = tasks[i:i + chunk]
            t0 = _t.time()
            out.update(verify_batch(part, self.workers, self.timeout))
            # reap REPL children orphaned by timeouts (ours only -- other users
            # run their own repl binaries from different paths)
            subprocess.run(["pkill", "-f", "melanieli22/.elan.*lake exe repl"],
                           capture_output=True)
            print(f"      verified {min(i+chunk, n)}/{n} ({_t.time()-t0:.0f}s for this chunk)",
                  flush=True)
        return out

    def sample(self, reqs, k, max_tokens, want_logprobs=False):
        """reqs: list of (key, stmt_header, prefix_lines).
        Returns {key: [ {code, text, ntok, lp_lines} ]} plus verification."""
        from vllm import SamplingParams, TokensPrompt
        prompts, meta = [], []
        for key, hdr, pre in reqs:
            ids = self.tok.encode(build_prompt(hdr, pre))
            prompts.append(TokensPrompt(prompt_token_ids=ids))
            meta.append((key, hdr, pre))
        sp = SamplingParams(n=k, temperature=1.0, top_p=0.95, max_tokens=max_tokens,
                            logprobs=0 if want_logprobs else None)
        outs = self.llm.generate(prompts, sp, use_tqdm=False)

        cand, tasks = collections.defaultdict(list), []
        for (key, hdr, pre), o in zip(meta, outs):
            for ki, c in enumerate(o.outputs):
                text = self.tok.decode(list(c.token_ids))
                body = text.split("```")[0]
                pref = ("\n" + "\n".join(pre)) if pre else ""
                code = hdr + pref + ("\n" if not body.startswith("\n") else "") + body
                self.calls += 1
                self.tokens += len(c.token_ids)
                lpl = None
                if want_logprobs and c.logprobs:
                    try:
                        lps = [d[t].logprob for d, t in zip(c.logprobs, c.token_ids)]
                        lpl = logprobs_by_line(self.tok, list(c.token_ids), lps)
                    except Exception:
                        lpl = None
                cand[key].append({"code": code, "ntok": len(c.token_ids), "lp_lines": lpl})
                tasks.append((f"{key}|{ki}", strip_header(code)))

        res = self._verify_chunked(tasks)
        for key, lst in cand.items():
            for ki, d in enumerate(lst):
                d["ok"] = ok(res.get(f"{key}|{ki}"))
        return cand


# ------------------------------------------------------------------ stages

def stage_harvest(args, gen, problems):
    print(f"\n=== STAGE A: harvesting {len(problems)} problems x {args.k_harvest} samples ===",
          flush=True)
    t0 = time.time()
    reqs = [(p["problem_id"], p["stmt"], []) for p in problems]
    cand = gen.sample(reqs, args.k_harvest, args.max_tokens_harvest, want_logprobs=True)

    out = {}
    n_solv = 0
    lens = []
    for p in problems:
        lst = cand.get(p["problem_id"], [])
        good = [d for d in lst if d["ok"]]
        bad = [d for d in lst if not d["ok"] and "sorry" not in d["code"]]
        if good:
            n_solv += 1
        for d in lst:
            _, b, _ = split_body(d["code"])
            if b:
                lens.append(len(b))
        out[p["problem_id"]] = {"stmt": p["stmt"], "n_ok": len(good),
                                "attempts": [{"code": d["code"], "ok": d["ok"],
                                              "lp_lines": d["lp_lines"]} for d in lst]}
    import numpy as np
    print(f"    solvable problems: {n_solv}/{len(problems)}  "
          f"({100.0*n_solv/len(problems):.0f}% pass@{args.k_harvest})")
    if lens:
        print(f"    proof length: mean {np.mean(lens):.1f} lines, median {np.median(lens):.0f} "
              f"-> implied random Top-1 {np.mean([1.0/x for x in lens]):.1%}")
    print(f"    {time.time()-t0:.0f}s, {gen.calls} generations, {gen.tokens:,} tokens")
    json.dump(out, open(args.harvest_out, "w"))
    print(f"    wrote {args.harvest_out}")
    return out


def stage_label(args, gen, harvest):
    print(f"\n=== STAGE B: counterfactual labeling ===", flush=True)
    picks, n_sorry = [], 0
    for pid, h in harvest.items():
        if h["n_ok"] == 0:
            continue
        for a in h["attempts"]:
            if a["ok"]:
                continue
            if "sorry" in a["code"]:
                n_sorry += 1
                continue
            hd, body, abs_lines = split_body(a["code"])
            if not hd or not body or len(body) < 2:
                continue
            picks.append({"pid": pid, "stmt": h["stmt"], "body": body,
                          "n": len(body), "lp_lines": a["lp_lines"], "code": a["code"]})
            break
        if len(picks) >= args.target_labels * 3:
            break
    print(f"    {len(picks)} candidate failed attempts ({n_sorry} skipped for `sorry`)")

    rec = {i: {} for i in range(len(picks))}

    def note(i, j, s, k):
        if j not in rec[i] or k > rec[i][j]["K"]:
            rec[i][j] = {"j": j, "K": k, "successes": s}

    def probe(items, k):
        reqs = [(str(i), picks[i]["stmt"], picks[i]["body"][:j]) for i, j in items]
        cand = gen.sample(reqs, k, args.max_tokens, want_logprobs=False)
        return {i: sum(1 for d in cand[str(i)] if d["ok"]) for i, _ in items}

    print(f"    R(0) gate at K={args.k_gate}...", flush=True)
    got = probe([(i, 0) for i in range(len(picks))], args.k_gate)
    live = {}
    for i in range(len(picks)):
        note(i, 0, got[i], args.k_gate)
        if got[i] > 0 and len(live) < args.target_labels:
            live[i] = {"lo": 0, "hi": picks[i]["n"]}
    print(f"    {len(live)} usable (target {args.target_labels})")

    rnd = 0
    while True:
        items = [(i, (st["lo"] + st["hi"]) // 2) for i, st in live.items()
                 if st["hi"] - st["lo"] > 1]
        if not items:
            break
        rnd += 1
        print(f"    round {rnd}: {len(items)} probes at K={args.k}...", flush=True)
        got = probe(items, args.k)
        for i, j in items:
            note(i, j, got[i], args.k)
            live[i]["lo" if got[i] > 0 else "hi"] = j

    items = [(i, st["lo"]) for i, st in live.items()]
    if items:
        print(f"    confirming at K={args.k_confirm}...", flush=True)
        got = probe(items, args.k_confirm)
        for i, j in items:
            note(i, j, got[i], args.k_confirm)

    labels = []
    for i, st in live.items():
        p = picks[i]
        labels.append({"pid": p["pid"], "code": p["code"], "n_lines": p["n"],
                       "i_star": st["hi"] - 1, "lp_lines": p["lp_lines"],
                       "R_sequence": sorted(rec[i].values(), key=lambda d: d["j"])})
    json.dump(labels, open(args.label_out, "w"), indent=2)
    print(f"    {len(labels)} labels -> {args.label_out}")
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default="pilot/pilot_problems_150.jsonl")
    ap.add_argument("--n_problems", type=int, default=60)
    ap.add_argument("--k_harvest", type=int, default=8)
    ap.add_argument("--target_labels", type=int, default=25)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--k_gate", type=int, default=8)
    ap.add_argument("--k_confirm", type=int, default=8)
    ap.add_argument("--max_tokens_harvest", type=int, default=1024)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--num_workers", type=int, default=24)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--gpu_mem_util", type=float, default=0.90)
    ap.add_argument("--harvest_out", default="pilot/ds_harvest.json")
    ap.add_argument("--label_out", default="pilot/ds_labels.json")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.problems)]
    problems = []
    for r in rows:
        s = statement_of(r["lean4_code"])
        if s:
            problems.append({"problem_id": r["problem_id"], "stmt": s})
    import random
    random.Random(args.seed).shuffle(problems)
    problems = problems[: args.n_problems]

    tokpath = find_tokenizer_json()
    print(f"raw tokenizer: {tokpath}")
    tok = RawTok(tokpath)
    probe_s = "  have h : 0 < a * b := by positivity\n  nlinarith\n"
    assert tok.decode(tok.encode(probe_s)) == probe_s, "raw tokenizer round-trip FAILED"
    print("raw tokenizer round-trip OK (AutoTokenizer would strip whitespace here)")

    from vllm import LLM
    # scai5 GPUs are shared; cap the memory fraction so vLLM does not try to
    # claim memory another user already holds on this card.
    llm = LLM(model=MODEL, tensor_parallel_size=1, max_model_len=4096,
              gpu_memory_utilization=args.gpu_mem_util)
    gen = Gen(llm, tok, args.num_workers, args.timeout)

    harvest = stage_harvest(args, gen, problems)
    labels = stage_label(args, gen, harvest)
    print(f"\nTotal: {gen.calls} generations, {gen.tokens:,} tokens")


if __name__ == "__main__":
    main()
