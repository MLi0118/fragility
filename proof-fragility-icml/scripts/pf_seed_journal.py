"""pf_seed_journal.py <run_dir> <out.jsonl> [check_run_dir]: write an output
journal holding <run_dir>'s first prover attempts (round 0 of every iteration),
keyed exactly as prover/query/api.py keys them, so another arm replays them.
With check_run_dir, also report how many of that run's round-0 prompts match."""
import glob
import hashlib
import json
import os
import sys

from transformers import AutoTokenizer

MODEL = "Goedel-LM/Goedel-Prover-V2-8B"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)


def render(messages):
    # api._render_vllm; round-0 prompts carry no prefill
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def h(prompt):
    return hashlib.sha256(f"{MODEL}\0{prompt}".encode()).hexdigest()


def round0(run_dir):
    for f in sorted(glob.glob(os.path.join(run_dir, "round0/prover/iter*/to_inference_codes.json"))):
        yield from json.load(open(f))


src, out = sys.argv[1], sys.argv[2]
keys = set()
with open(out, "w") as f:
    n = 0
    for r in round0(src):
        c = r.get("detailed_cost") or {}
        k = h(render(r["model_input"]))
        keys.add(k)
        f.write(json.dumps({"h": k, "text": r["model_output"],
                            "inp": c.get("input_tokens", 0), "outp": c.get("output_tokens", 0)}) + "\n")
        n += 1
print(f"wrote {n} first attempts ({len(keys)} distinct prompts) to {out}")
if len(sys.argv) > 3:
    other = [h(render(r["model_input"])) for r in round0(sys.argv[3])]
    print(f"{sum(k in keys for k in other)}/{len(other)} round-0 prompts of {sys.argv[3]} are in the journal")
