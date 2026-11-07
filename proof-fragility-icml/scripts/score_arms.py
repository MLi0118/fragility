"""score_arms.py <results_dir> <sim_dir> <out.json> run:sim [run:sim ...]

Score finished ICML-pipeline runs, one per arm:
  solved   problems with a combined proof that has pass and complete both
           true (fully_solved.json comes out empty, so it isn't used)
  tokens   prover input/output tokens summed over every attempt the prover
           sent (round*/prover/iter*/to_inference_codes*.json)
  sim      the arm's proof_simulation sweep: mean problems solved per
           (max_breakdowns, max_corrections), i.e. k = max_corrections + 1
Example:
  python score_arms.py $SCRATCH/results results/simulations scores.json \\
      full85_feedback:pf_fixed_feedback full85s_predict_cut:pf_fixeds_predict_cut
"""
import glob
import json
import os
import sys


def solved(run):
    out = set()
    for r in json.load(open(os.path.join(run, "combined/compilation_results.json"))):
        c = r["compilation_result"]
        if c.get("pass") and c.get("complete"):
            out.add(r["name"].rsplit("_r", 1)[0])
    return sorted(out)


def tokens(run):
    t = {"attempts": 0, "repairs": 0, "input": 0, "output": 0, "repair_input": 0, "repair_output": 0}
    for f in glob.glob(os.path.join(run, "round*/prover/iter*/to_inference_codes*.json")):
        repair = "_corr" in os.path.basename(f)
        for r in json.load(open(f)):
            c = r.get("detailed_cost") or {}
            t["attempts"] += 1
            t["input"] += c.get("input_tokens", 0)
            t["output"] += c.get("output_tokens", 0)
            if repair:
                t["repairs"] += 1
                t["repair_input"] += c.get("input_tokens", 0)
                t["repair_output"] += c.get("output_tokens", 0)
    return t


def sim(path):
    s = json.load(open(os.path.join(path, "sweep_summary.json")))
    rows = []
    for c in s["configs"]:
        p = c["params"]
        rows.append({"breakdowns": p["max_breakdowns"], "k": p["max_corrections"] + 1,
                     "solved": round(c["avg_solve_rate"] * s["n_problems"], 2),
                     "input_tokens": c["avg_total_cost"]["input_tokens"],
                     "output_tokens": c["avg_total_cost"]["output_tokens"]})
    return {"n_problems": s["n_problems"], "n_seeds": s["n_seeds"], "configs": rows}


res, sims, out = sys.argv[1:4]
scores = {}
for spec in sys.argv[4:]:
    run, simname = spec.split(":")
    s = solved(os.path.join(res, run))
    scores[run] = {"n_solved": len(s), "solved": s, "tokens": tokens(os.path.join(res, run)),
                   "simulation": sim(os.path.join(sims, simname))}
    t = scores[run]["tokens"]
    print(f"{run:22s} solved {len(s):3d}  prover tokens {(t['input'] + t['output']) / 1e6:6.2f}M  "
          f"repairs {t['repairs']}")
json.dump(scores, open(out, "w"), indent=1)
