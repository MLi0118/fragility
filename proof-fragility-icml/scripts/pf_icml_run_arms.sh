#!/bin/bash
# pf_icml_run_arms.sh <job> <input.jsonl> <k> <gpu_feedback> <gpu_predict> <gpu_random>
#
# One ICML-pipeline run per arm on the same problems, n_b=4, n_f=4, and a
# chain of k attempts per target (1 fresh + k-1 repairs; k=4 is a prefix of k=8).
#   feedback     their correction: rewrite from the compiler error
#   predict_cut  keep body[:i*_hat], regenerate (fragility/repair.py)
#   random_cut   keep body[:Uniform], regenerate (control)
# The cut arms copy feedback's finished breakdown + formalizer output, so all
# three prove against identical breakdowns; auto_resume restarts them at the
# prover. The two cut arms then run in parallel, one GPU each.
set -euo pipefail
JOB=$1; INPUT=$2; K=$3; G_FB=$4; G_PC=$5; G_RC=$6
R=/space/melanieli22/optimizing-lean-agents
source $R/.venv/bin/activate
export ELAN_HOME=/space/melanieli22/.elan PATH=/space/melanieli22/.elan/bin:$PATH
# hydra_runner resets HF_HOME to $SCRATCH/huggingface, a symlink to this cache
export HF_HOME=/space/melanieli22/.cache/huggingface PIP_CACHE_DIR=/space/melanieli22/.cache/pip
export MPLCONFIGDIR=/space/melanieli22/.cache/matplotlib TOKENIZERS_PARALLELISM=false
export PROOF_TIMEOUT=300  # repl_scheduler default, pinned so all arms match
export CUDA_DEVICE_ORDER=PCI_BUS_ID SCRATCH=/home/melanieli22/pf_icml_scratch PYTHONPATH=$R
cd $R
RES=$SCRATCH/results
LOG=/home/melanieli22/pf_icml_logs; mkdir -p $LOG

run_arm() {  # run_arm <mode> <gpu>
  # vLLM can hang silently mid-batch (full85 breakdown, 2026-10-22: stuck at
  # 167/340 for 6 h, all threads in futex wait). The watchdog kills a tree that
  # used no CPU for 10 min; auto_resume restarts at the unfinished component and
  # the output journal replays gpt-oss generations that already finished.
  local try log=$LOG/${JOB}_$1.log
  for try in 1 2 3 4 5 6; do
    [ -f $log ] && mv $log $LOG/${JOB}_$1.try$((try-1)).log
    CUDA_VISIBLE_DEVICES=$2 PF_VLLM_JOURNAL=$RES/${JOB}_$1.journal.jsonl PF_VLLM_JOURNAL_MODELS=gpt-oss \
    nice -n 10 python seed_prover/hydra_runner.py --config-name=pf_icml \
      components.data_loader.config.input_path=$INPUT \
      components.breakdown_json.config.sample_times=4 \
      components.formalizer.config.sample_times=4 \
      components.formalizer.config.cpu=32 \
      components.proof_builder.config.cpu=32 \
      prover.pipeline.initial_samples=1 \
      prover.pipeline.correction_samples=1 \
      prover.pipeline.max_correction_rounds=$((K-1)) \
      prover.components.prover.config.cpu=32 \
      prover.components.prover.config.model_config=configs/models/local/goedel_prover_8b.yaml \
      +prover.components.prover.config.repair_mode=$1 \
      output.job_name=${JOB}_$1 "${EXTRA[@]}" > $log 2>&1 &
    local pid=$!
    if ! python3 /space/melanieli22/pf_watchdog.py $pid 10 >> $log; then
      echo "HUNG $1 try $try $(date)"; wait $pid 2>/dev/null || true; sleep 30; continue
    fi
    wait $pid || true
    # hydra_runner logs a crashed stage and still reports success
    if grep -q "^Traceback" $log; then echo "ARM_FAILED $1"; return 1; fi
    return 0
  done
  echo "ARM_FAILED $1 (hung 6 times)"; return 1
}
EXTRA=("${@:7}")

run_arm feedback $G_FB
echo FEEDBACK_DONE
pids=()
for pair in predict_cut:$G_PC random_cut:$G_RC; do
  arm=${pair%%:*}; gpu=${pair##*:}
  dst=$RES/${JOB}_$arm/round0
  if [ ! -d "$dst" ]; then
    mkdir -p "$dst"
    for d in $RES/${JOB}_feedback/round0/*; do
      [ "$(basename $d)" = prover ] || cp -r "$d" "$dst/"
    done
    # The formalizer writes its targets into prover/; copy those, not the attempts
    mkdir -p "$dst/prover"
    cp $RES/${JOB}_feedback/round0/prover/{theorems,lemmas}.json "$dst/prover/"
    cp $RES/${JOB}_feedback/dataset.json $RES/${JOB}_$arm/ 2>/dev/null || true
  fi
  # Refuse to start a cut arm that would redo breakdown/formalizer (unshared inputs)
  ( cd seed_prover && python -c "
import sys; from auto_resume import detect_checkpoint
c = detect_checkpoint(sys.argv[1]); print(\"$arm resume:\", c)
sys.exit(0 if c[\"resume_from_component\"] not in (None, \"breakdown\", \"formalizer\") else 1)" $RES/${JOB}_$arm ) || { echo "BAD_RESUME $arm"; exit 1; }
  run_arm $arm $gpu & pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done; [ $rc -eq 0 ] || exit 1
echo ALL_ARMS_DONE
