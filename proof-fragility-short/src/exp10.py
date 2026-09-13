"""
pilot/exp10.py  --  10-sample redesigned fragility experiment.

Implements the three highest-priority items from the revised brief, on a
sample small enough to validate the design before spending GPU hours:

  (1) REAL counterfactual supervision, not compiler proxies.
      R(j) = 1[at least one of K regenerated suffixes succeeds from prefix j].
      Binary search for the recoverability boundary => ~log2(n) probes at K=4
      instead of n probes. Boundary is then re-confirmed at K=8.
      We store the FULL record (j, K, successes, tokens) for every probed j,
      not just the final i*, so p_hat_j = successes_j / K is available later
      for training a recoverability/utility regressor rather than only a hard
      failure index.

  (2) HIERARCHICAL localization. Candidate set is ~29 lines/proof, vs ~4.6 for
      the peer -- that difficulty gap is most of the raw-accuracy gap. We
      segment each proof into top-level semantic blocks and measure how far
      that shrinks the candidate set, so P(i) = P(B_i) * P(i | B_i) becomes
      tractable without changing what "exact step" means at eval time.

  (3) REPAIR-ACTION UTILITY. At the boundary we test the zero-LLM-token
      "auto Lean" action (a fixed closing-tactic portfolio) so the action
      space a in {auto, 1-step, window, suffix} has real P(solve | i, a) data
      at its cheapest rung.

The headline measurement is (4): agreement between our existing compiler-span
proxy label and the real i* on the SAME attempts. The peer reports ~4%
agreement for naive compiler-line labels; our span-containment matching should
beat that, and this is the first direct test of by how much.

Generation uses assistant PREFILL: Goedel is a CoT prover, so we build its
normal chat prompt and prefill the assistant turn with the code block up to
the retained prefix. Prefix retention becomes structural rather than something
we hope for, and the expensive planning phase is skipped.
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

MODEL = "Goedel-LM/Goedel-Prover-V2-8B"

# Zero-LLM-token repair action: a fixed portfolio of closing tactics.
AUTO_TACTICS = [
    "linarith", "nlinarith [sq_nonneg (a - b), sq_nonneg (a + b)]", "norm_num",
    "ring", "field_simp", "positivity", "simp_all", "aesop", "omega",
    "nlinarith", "simp", "decide",
]


# ---------------------------------------------------------------- structure

def split_body(full_code):
    """Return (header_through_':= by', [non-blank body lines], [their 1-indexed
    absolute line numbers in full_code]).

    The absolute numbers are needed to compare against the compiler-proxy
    labels, whose line numbers use a different origin (see calibrate_offset)."""
    idx = full_code.find(":= by")
    if idx < 0:
        return None, None, None
    head = full_code[: idx + len(":= by")]
    rest = full_code[idx + len(":= by"):]
    # line number (1-indexed) of the last line of `head`
    head_line = head.count("\n") + 1
    body, abs_lines = [], []
    for off, ln in enumerate(rest.split("\n")):
        if ln.strip():
            body.append(ln)
            abs_lines.append(head_line + off)
    return head, body, abs_lines


def calibrate_offset(feature_lines, body_abs):
    """The step features number lines from the theorem, not from the top of
    full_code, so a constant offset separates the two coordinate systems.
    Recover it by choosing the shift that lands the most feature lines onto
    real body lines, rather than assuming a convention."""
    if not feature_lines or not body_abs:
        return None, 0
    target = set(body_abs)
    best, best_hits = None, -1
    lo = min(body_abs) - min(feature_lines)
    for off in range(lo - 3, lo + 4):
        hits = sum(1 for l in feature_lines if l + off in target)
        if hits > best_hits:
            best, best_hits = off, hits
    return best, best_hits / len(feature_lines)


def indent_of(line):
    return len(line) - len(line.lstrip())


_DECL_RE = re.compile(r"^\s*(?:theorem|lemma|example)\b", re.MULTILINE)


def strip_header(code):
    """Drop the `import ...` preamble before sending to the REPL.

    send_command_and_wait already establishes the imports in env 0 via its
    `imports=` argument. Leaving them in the code too makes Lean reject the
    whole command with "invalid 'import' command, it must be used in the
    beginning of the file" -- which fails every proof, including correct ones."""
    m = _DECL_RE.search(code)
    return code[m.start():] if m else code


def segment_blocks(body):
    """Top-level semantic blocks: a line at minimum indentation opens a block;
    more-deeply-indented lines belong to the block above them. This groups a
    `have h : T := by` together with its nested proof, and likewise for calc
    and other structured tactics."""
    if not body:
        return []
    base = min(indent_of(ln) for ln in body)
    blocks, cur = [], []
    for i, ln in enumerate(body):
        if indent_of(ln) <= base and cur:
            blocks.append(cur)
            cur = []
        cur.append(i)
    if cur:
        blocks.append(cur)
    return blocks


def block_of_index(blocks, i):
    for b, idxs in enumerate(blocks):
        if i in idxs:
            return b
    return None


# ---------------------------------------------------------------- prompting

def build_prefill(tokenizer, head, body, j):
    """Chat prompt + assistant prefill holding the first j body lines."""
    stmt = head + " sorry"
    user = (
        f"Complete the following Lean 4 code:\n\n```lean4\n{stmt}```\n\n"
        "Before producing the Lean 4 code to formally prove the given theorem, "
        "provide a detailed proof plan outlining the main proof steps and strategies.\n"
        "The plan should highlight key ideas, intermediate lemmas, and proof structures "
        "that will guide the construction of the final formal proof."
    )
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True
    )
    prefix_lines = "\n".join(body[:j])
    prefill_code = head + ("\n" + prefix_lines if prefix_lines else "")
    return chat + "```lean4\n" + prefill_code, prefill_code


# ---------------------------------------------------------------- verifying

def _worker(worker_id, task_queue, result_list, timeout):
    child, _ = rs.initiate_child()
    while True:
        item = task_queue.get()
        if item is None:
            break
        resp = rs.send_command_and_wait(child, item["code"], env=0, timeout=timeout,
                                        imports=rs.DEFAULT_IMPORTS)
        resp["name"] = item["name"]
        result_list.append(resp)
        # A timed-out tactic search leaves the REPL desynced; respawn or this
        # worker hangs forever on its next task.
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


def verify_batch(tasks, num_workers=24, timeout=60):
    if not tasks:
        return {}
    manager = mp.Manager()
    q, out = manager.Queue(), manager.list()
    for name, code in tasks:
        q.put({"name": name, "code": code})
    n = min(num_workers, len(tasks))
    for _ in range(n):
        q.put(None)
    procs = [mp.Process(target=_worker, args=(i, q, out, timeout)) for i in range(n)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    return {r["name"]: r["compilation_result"] for r in out}


def ok(res):
    return bool(res and res.get("pass") and res.get("complete"))


# ---------------------------------------------------------------- probing

class Prober:
    """Batches R(j) probes across all attempts so each binary-search round is
    a single vLLM call."""

    def __init__(self, llm, tok, sp_factory, num_workers, timeout):
        self.llm, self.tok = llm, tok
        self.sp_factory = sp_factory
        self.num_workers, self.timeout = num_workers, timeout
        self.gen_tokens = 0
        self.gen_calls = 0

    def probe(self, reqs, k):
        """reqs: list of (key, head, body, j). Returns {key: (successes, k, tokens)}."""
        if not reqs:
            return {}
        prompts, prefills = [], {}
        for key, head, body, j in reqs:
            prompt, prefill_code = build_prefill(self.tok, head, body, j)
            prompts.append(prompt)
            prefills[key] = prefill_code

        outs = self.llm.generate(prompts, self.sp_factory(k), use_tqdm=False)

        tasks, toks = [], collections.defaultdict(int)
        for (key, _, _, _), o in zip(reqs, outs):
            for ki, comp in enumerate(o.outputs):
                toks[key] += len(comp.token_ids)
                self.gen_tokens += len(comp.token_ids)
                self.gen_calls += 1
                code = prefills[key] + comp.text.split("```")[0]
                tasks.append((f"{key}|{ki}", strip_header(code)))

        res = verify_batch(tasks, self.num_workers, self.timeout)
        agg = {}
        for key, _, _, _ in reqs:
            s = sum(1 for ki in range(k) if ok(res.get(f"{key}|{ki}")))
            agg[key] = (s, k, toks[key])
        return agg


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minified", required=True)
    ap.add_argument("--features", required=True,
                    help="cheap_features_150.json, for the compiler-proxy comparison")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--k_confirm", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=1536)
    ap.add_argument("--num_workers", type=int, default=24)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default="exp10_results.json")
    args = ap.parse_args()

    t_start = time.time()

    data = json.load(open(args.minified))
    byp = collections.defaultdict(list)
    for gi, r in enumerate(data):
        byp[r["origin_problem_id"]].append((gi, r))
    solvable = {k: v for k, v in byp.items() if any(x.get("complete") for _, x in v)}

    # Compiler-span proxy labels keyed by MINIFIED INDEX, so the comparison in
    # [4] is against the very same attempt we counterfactually label -- keying
    # by problem id would silently compare different attempts.
    proxy = {}
    for r in json.load(open(args.features)):
        lab, src = r.get("contrastive_label_idx"), "contrastive"
        if lab is None:
            lab, src = r.get("compiler_label_idx"), "compiler"
        if r.get("index") is not None and lab is not None:
            proxy[r["index"]] = {"label": lab, "source": src,
                                 "feature_lines": [f["line"] for f in r["features"]]}

    # One attempt per problem keeps the 10 samples independent (proofs are the
    # grouping unit), but take the problems in shuffled order -- iterating
    # sorted() biases the sample toward alphabetically-early problems.
    import random
    problem_order = sorted(solvable.items())
    random.Random(args.seed).shuffle(problem_order)

    picks = []
    for pid, rows in problem_order:
        for gi, r in rows:
            if r.get("complete"):
                continue
            code = r.get("full_code")
            if not code or code == "None":
                continue
            head, body, abs_lines = split_body(code)
            if not head or not body or len(body) < 4:
                continue
            picks.append({"pid": pid, "idx": gi, "head": head, "body": body,
                          "abs_lines": abs_lines, "n": len(body),
                          "blocks": segment_blocks(body)})
            break
        if len(picks) >= args.n:
            break

    print(f"=== {len(picks)} attempts selected (failed attempts from solvable problems) ===")
    for p in picks:
        print(f"  {p['pid']}: {p['n']} lines -> {len(p['blocks'])} blocks")
    tot_l = sum(p["n"] for p in picks)
    tot_b = sum(len(p["blocks"]) for p in picks)
    print(f"\n[2] Hierarchical candidate reduction: {tot_l/len(picks):.1f} lines/proof "
          f"-> {tot_b/len(picks):.1f} blocks/proof ({tot_l/max(tot_b,1):.2f}x smaller)")

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(model=MODEL, tensor_parallel_size=1, max_model_len=8192)

    def sp_factory(k):
        return SamplingParams(n=k, temperature=1.0, top_p=0.95, max_tokens=args.max_tokens)

    pr = Prober(llm, tok, sp_factory, args.num_workers, args.timeout)

    # record[key] holds every probed j for this attempt
    record = {i: {} for i in range(len(picks))}

    def note(i, j, s, k, t):
        prev = record[i].get(j)
        if prev is None or k > prev["K"]:
            record[i][j] = {"j": j, "K": k, "successes": s, "tokens": t}

    # ---- round 0: R(0). No prefix retained -- can the prover solve it at all?
    print(f"\n[1] Probing R(0) at K={args.k} for {len(picks)} attempts...", flush=True)
    t0 = time.time()
    agg = pr.probe([(i, p["head"], p["body"], 0) for i, p in enumerate(picks)], args.k)
    for i, (s, k, t) in agg.items():
        note(i, 0, s, k, t)
    print(f"    done in {time.time()-t0:.0f}s", flush=True)

    live = {}
    for i, p in enumerate(picks):
        s = record[i][0]["successes"]
        if s > 0:
            live[i] = {"lo": 0, "hi": p["n"]}   # R(lo)>0, R(hi)=0 (attempt failed)
        print(f"    {p['pid']}: R(0)={s}/{args.k}" + ("" if s > 0 else "   [discarded]"))
    print(f"    {len(live)}/{len(picks)} attempts usable for binary search.")

    # ---- binary search rounds
    rnd = 0
    while True:
        reqs = []
        for i, st in live.items():
            if st["hi"] - st["lo"] > 1:
                mid = (st["lo"] + st["hi"]) // 2
                reqs.append((i, picks[i]["head"], picks[i]["body"], mid))
        if not reqs:
            break
        rnd += 1
        print(f"\n[1] Binary-search round {rnd}: {len(reqs)} probes at K={args.k}...", flush=True)
        t0 = time.time()
        agg = pr.probe(reqs, args.k)
        for (i, _, _, j) in reqs:
            s, k, t = agg[i]
            note(i, j, s, k, t)
            if s > 0:
                live[i]["lo"] = j
            else:
                live[i]["hi"] = j
        print(f"    done in {time.time()-t0:.0f}s", flush=True)

    # ---- confirm boundary at higher K
    conf = [(i, picks[i]["head"], picks[i]["body"], st["lo"]) for i, st in live.items()]
    if conf:
        print(f"\n[1] Confirming boundary at K={args.k_confirm} ({len(conf)} probes)...", flush=True)
        t0 = time.time()
        agg = pr.probe(conf, args.k_confirm)
        for (i, _, _, j) in conf:
            s, k, t = agg[i]
            note(i, j, s, k, t)
        print(f"    done in {time.time()-t0:.0f}s", flush=True)

    # ---- [3] repair-action utility: zero-token auto-Lean at the boundary
    print(f"\n[3] Testing zero-token auto-Lean portfolio at boundary...", flush=True)
    auto_tasks = []
    for i, st in live.items():
        p = picks[i]
        pref = "\n".join(p["body"][: st["lo"]])
        base = p["head"] + ("\n" + pref if pref else "")
        ind = " " * (indent_of(p["body"][0]) if p["body"] else 2)
        for ti, tac in enumerate(AUTO_TACTICS):
            auto_tasks.append((f"{i}|{ti}", strip_header(base + "\n" + ind + tac + "\n")))
    auto_res = verify_batch(auto_tasks, args.num_workers, args.timeout)
    auto_hits = collections.defaultdict(list)
    for i, st in live.items():
        for ti, tac in enumerate(AUTO_TACTICS):
            if ok(auto_res.get(f"{i}|{ti}")):
                auto_hits[i].append(tac)
    n_auto = sum(1 for i in live if auto_hits[i])
    print(f"    {n_auto}/{len(live)} boundaries closed by a zero-token auto tactic "
          f"({len(auto_tasks)} REPL calls, 0 LLM tokens)")

    # ---- results + [4] proxy agreement
    results, agree, comparable = [], 0, 0
    print("\n=== Per-attempt results ===")
    for i, p in enumerate(picks):
        st = live.get(i)
        if st is None:
            results.append({"pid": p["pid"], "idx": p["idx"], "n_lines": p["n"],
                            "n_blocks": len(p["blocks"]), "status": "unsolvable_at_K",
                            "probes": list(record[i].values())})
            print(f"  {p['pid']}: discarded (R(0)=0)")
            continue
        i_star = st["hi"] - 1           # first retained line that kills recovery
        blk = block_of_index(p["blocks"], i_star)
        row = {
            "pid": p["pid"], "idx": p["idx"], "n_lines": p["n"],
            "n_blocks": len(p["blocks"]), "status": "labeled",
            "i_star": i_star, "i_star_block": blk,
            "boundary_lo": st["lo"], "boundary_hi": st["hi"],
            "auto_tactics_that_close": auto_hits[i],
            "probes": sorted(record[i].values(), key=lambda d: d["j"]),
            "fragile_line": p["body"][i_star].strip()[:120],
        }

        # [4] Map the proxy label into body-line space before comparing. The
        # proxy indexes fine-grained allTactics steps; i_star indexes body
        # lines. Comparing the two index spaces directly would be meaningless.
        px = proxy.get(p["idx"])
        proxy_bidx = None
        if px:
            off, fit = calibrate_offset(px["feature_lines"], p["abs_lines"])
            row["proxy_source"] = px["source"]
            row["proxy_calibration_fit"] = round(fit, 3)
            if off is not None and fit >= 0.5 and px["label"] < len(px["feature_lines"]):
                pabs = px["feature_lines"][px["label"]] + off
                if pabs in p["abs_lines"]:
                    proxy_bidx = p["abs_lines"].index(pabs)
            row["proxy_body_idx"] = proxy_bidx
        if proxy_bidx is not None:
            comparable += 1
            row["proxy_agrees"] = (proxy_bidx == i_star)
            if row["proxy_agrees"]:
                agree += 1
        results.append(row)
        verdict = ("MATCH" if row.get("proxy_agrees")
                   else ("differ" if proxy_bidx is not None else "n/a"))
        print(f"  {p['pid']}: i*={i_star} (block {blk}/{len(p['blocks'])}) "
              f"proxy={proxy_bidx} {verdict} | auto={len(auto_hits[i])} "
              f"| {row['fragile_line'][:60]}")

    labeled = [r for r in results if r["status"] == "labeled"]
    elapsed = time.time() - t_start

    print("\n" + "=" * 62)
    print("SUMMARY")
    print("=" * 62)
    print(f"[1] Real counterfactual labels : {len(labeled)}/{len(picks)} attempts")
    print(f"    LLM generations            : {pr.gen_calls} ({pr.gen_tokens:,} output tokens)")
    print(f"[2] Candidate set              : {tot_l/len(picks):.1f} lines -> "
          f"{tot_b/len(picks):.1f} blocks per proof ({tot_l/max(tot_b,1):.2f}x reduction)")
    print(f"[3] Auto-Lean closes boundary  : {n_auto}/{len(live)} (zero LLM tokens)")
    if comparable:
        print(f"[4] Compiler proxy vs real i*  : {agree}/{comparable} exact agreement "
              f"({100.0*agree/comparable:.1f}%)")
    else:
        print("[4] Compiler proxy vs real i*  : no overlapping attempts to compare")
    print(f"\nWall clock: {elapsed/60:.1f} min for {len(picks)} attempts "
          f"({elapsed/max(len(picks),1):.0f}s each)")
    print(f"Projected for 103 attempts: {elapsed/max(len(picks),1)*103/3600:.2f} h")

    json.dump({
        "config": vars(args),
        "summary": {
            "n_attempts": len(picks), "n_labeled": len(labeled),
            "lines_per_proof": tot_l / len(picks), "blocks_per_proof": tot_b / len(picks),
            "auto_close": n_auto, "n_live": len(live),
            "proxy_agree": agree, "proxy_comparable": comparable,
            "gen_calls": pr.gen_calls, "gen_tokens": pr.gen_tokens,
            "elapsed_s": elapsed,
        },
        "results": results,
    }, open(args.out, "w"), indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
