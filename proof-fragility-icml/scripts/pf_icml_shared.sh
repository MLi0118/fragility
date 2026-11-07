#!/bin/bash
# pf_icml_shared.sh <src_job> <job> <input.jsonl> <k> <gpu_predict> <gpu_random>
#
# Rerun the two cut arms so they share the feedback arm's first attempts, not
# just its breakdowns and formalizations. Each arm's output journal is seeded
# with <src_job>_feedback's round-0 prover outputs (pf_seed_journal.py) and
# PF_VLLM_JOURNAL_ROUND0 limits replay to round 0, so a target the feedback
# arm also attempted starts from the same failed proof; repairs are generated
# fresh. Targets feedback never reached (later iterations) are sampled anew.
set -euo pipefail
SRC=$1; JOB=$2; INPUT=$3; K=$4; G_PC=$5; G_RC=$6
R=/space/melanieli22/optimizing-lean-agents
source $R/.venv/bin/activate
export ELAN_HOME=/space/melanieli22/.elan PATH=/space/melanieli22/.elan/bin:$PATH
export HF_HOME=/space/melanieli22/.cache/huggingface PIP_CACHE_DIR=/space/melanieli22/.cache/pip
export MPLCONFIGDIR=/space/melanieli22/.cache/matplotlib TOKENIZERS_PARALLELISM=false
export PROOF_TIMEOUT=300
export CUDA_DEVICE_ORDER=PCI_BUS_ID SCRATCH=/home/melanieli22/pf_icml_scratch PYTHONPATH=$R
cd $R
RES=$SCRATCH/results
LOG=/home/melanieli22/pf_icml_logs; mkdir -p $LOG
FB=$RES/${SRC}_feedback

run_arm() {  # run_arm <mode> <gpu>; same retry/watchdog loop as pf_icml_run_arms.sh
  local try log=$LOG/${JOB}_$1.log
  for try in 1 2 3 4 5 6; do
    [ -f $log ] && mv $log $LOG/${JOB}_$1.try$((try-1)).log
    CUDA_VISIBLE_DEVICES=$2 PF_VLLM_JOURNAL=$RES/${JOB}_$1.journal.jsonl \
    PF_VLLM_JOURNAL_MODELS=Goedel-Prover PF_VLLM_JOURNAL_ROUND0=1 \
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
      output.job_name=${JOB}_$1 > $log 2>&1 &
    local pid=$!
    if ! python3 /space/melanieli22/pf_watchdog.py $pid 10 >> $log; then
      echo "HUNG $1 try $try $(date)"; wait $pid 2>/dev/null || true; sleep 30; continue
    fi
    wait $pid || true
    if grep -q "^Traceback" $log; then echo "ARM_FAILED $1"; return 1; fi
    return 0
  done
  echo "ARM_FAILED $1 (hung 6 times)"; return 1
}

pids=()
for pair in predict_cut:$G_PC random_cut:$G_RC; do
  arm=${pair%%:*}; gpu=${pair##*:}
  dst=$RES/${JOB}_$arm/round0
  if [ ! -d "$dst" ]; then
    mkdir -p "$dst/prover"
    for d in $FB/round0/*; do
      [ "$(basename $d)" = prover ] || cp -r "$d" "$dst/"
    done
    cp $FB/round0/prover/{theorems,lemmas}.json "$dst/prover/"
    cp $FB/dataset.json $RES/${JOB}_$arm/
    # A restart must not re-seed: the journal already holds this arm's own generations
    python /space/melanieli22/pf_seed_journal.py $FB $RES/${JOB}_$arm.journal.jsonl
  fi
  ( cd seed_prover && python -c "
import sys; from auto_resume import detect_checkpoint
c = detect_checkpoint(sys.argv[1]); print(\"$arm resume:\", c)
sys.exit(0 if c[\"resume_from_component\"] not in (None, \"breakdown\", \"formalizer\") else 1)" $RES/${JOB}_$arm ) || { echo "BAD_RESUME $arm"; exit 1; }
  run_arm $arm $gpu & pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done; [ $rc -eq 0 ] || exit 1
echo ALL_ARMS_DONE
