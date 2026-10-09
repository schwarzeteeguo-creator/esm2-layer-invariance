#!/bin/bash
# R2 chain phase 2b — sharded single-site sensitivity (3 concurrent shards
# per model config; per-dataset resume; merge_shards folds results).
# Phases 0/1 (multisite stats, CKA200) already complete and are skipped.
set -u
source ~/miniforge3/etc/profile.d/conda.sh
conda activate protein

export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6

ROOT=~/AIDD/projects/protein_revision
CODE=$ROOT/code
DATA=$ROOT/data/substitutions
OUT=$ROOT/results
cd $CODE

wait_vram () {
  while true; do
    local used total
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    [ "$((total - used))" -ge "$1" ] && break
    echo "$(date '+%F %T') waiting for ${1}MiB free VRAM (used=${used}MiB)"
    sleep 300
  done
}

count_ok () {
  OUT_JSON="$1" MIN_KEYS="$2" python - <<'EOF'
import json, os, sys
try:
    j = json.load(open(os.environ["OUT_JSON"]))
    sys.exit(0 if len(j) >= int(os.environ["MIN_KEYS"]) else 1)
except Exception:
    sys.exit(1)
EOF
}

# wait until no single-site probing python is running from the old chain
while pgrep -f "probing_v4.*single_site" >/dev/null; do
  echo "old single-site process still alive; waiting"
  sleep 60
done

SPLICES="random modulo contiguous position_groupkfold"
run_config () {  # tag model_dir extra_out_args...
  local tag=$1 model_key=$2 model_dir=$3; shift 3
  local out=$OUT/probing_v4_single_$tag
  local canon=$out/v4__mutant__combined__random.json
  for attempt in 1 2 3 4 5; do
    wait_vram 4000
    pids=""
    for sh in 0 1 2; do
      python -u -m esm_embedding.probing_v4 \
        --data_dir $DATA --model_key $model_key --model_dir $model_dir \
        --arms masked_wt unmasked_wt mutant --splits $SPLICES \
        --single_site_only --batch_size 4 --shard $sh --nshards 3 \
        --output_dir $out "$@" \
        >> $OUT/logs/2_single_${tag}_s${sh}.log 2>&1 &
      pids="$pids $!"
    done
    wait $pids
    python -u -m esm_embedding.merge_shards $out \
      >> $OUT/logs/2_single_${tag}_merge.log 2>&1
    if count_ok "$canon" 55; then echo "config $tag done"; return 0; fi
    echo "$(date '+%F %T') $tag incomplete after attempt $attempt"
    sleep 120
  done
  echo "WARNING: $tag did not reach 55 datasets"
  return 1
}

echo "=== [2b] single-site sharded: esm2 ==="
run_config esm2 esm2_650m $ROOT/models/esm2
echo "=== [2b] single-site sharded: saprot(mask) ==="
run_config saprot saprot_650m $ROOT/models/saprot_hf
echo "=== [2b] single-site sharded: saprot(3di) ==="
run_config saprot_3di saprot_650m $ROOT/models/saprot_hf --struct_dir $ROOT/data/3di

echo "=== [3] Envision baseline (waits for MSA zip) ==="
SCORES=$ROOT/data/pg_scores
for i in $(seq 1 720); do
  [ -f $SCORES/DMS_msa_files.zip ] && break
  echo "$(date '+%F %T') waiting for MSA download"; sleep 120
done
if [ -f $SCORES/DMS_msa_files.zip ]; then
  mkdir -p $SCORES/msa && cd $SCORES/msa && unzip -qn $SCORES/DMS_msa_files.zip
  cd $CODE
  MSA_DIR=$(dirname $(find $SCORES/msa -maxdepth 3 \( -name '*.a2m' -o -name '*.a3m' -o -name '*.gz' \) | head -1) 2>/dev/null)
  python -u -m esm_embedding.envision_baseline \
    --data_dir $DATA --msa_dir "${MSA_DIR:-$SCORES/msa}" \
    --output_dir $OUT/envision \
    >> $OUT/logs/3_envision.log 2>&1
  echo "envision done"
else
  echo "MSA zip never appeared; envision without PSSM"
  python -u -m esm_embedding.envision_baseline \
    --data_dir $DATA --output_dir $OUT/envision \
    >> $OUT/logs/3_envision.log 2>&1
fi

echo "=== R2B CHAIN ALL DONE ==="
