#!/bin/bash
# Chain 8 — training-safe concurrent edition (final pooling phase).
#
# Scope decision 2026-09-17: the 63-dataset pooling extension (an optional
# add-on) is CANCELLED — under the <=7GB training-safe VRAM ceiling it would
# take 3+ days. The ORIGINAL submitted design (20 datasets per model) is
# restored. The two models run CONCURRENTLY (one process each, extraction
# batch 2, OMP 8): ~3.2GB each, ~9.5GB stays free so the user's ~8GB
# training job can start at any moment. First-come-first-served, no
# killing in either direction. until-complete loops; per-dataset resume.
set -u
source ~/miniforge3/etc/profile.d/conda.sh
conda activate protein

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

ROOT=~/AIDD/projects/protein_revision
CODE=$ROOT/code
DATA=$ROOT/data/substitutions
OUT=$ROOT/results
mkdir -p $OUT/logs
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

until_complete () {  # need_MiB max json min cmd...
  local need=$1 max=$2 jp=$3 mn=$4; shift 4
  for i in $(seq 1 "$max"); do
    wait_vram "$need"
    "$@"
    if count_ok "$jp" "$mn"; then return 0; fi
    echo "$(date '+%F %T') incomplete after attempt $i — retry in 10 min"
    sleep 600
  done
  echo "$(date '+%F %T') WARNING: $jp >= $mn not reached after $max attempts"
  return 1
}

echo "=== [5] pooling saprot-20 || [9] pooling esm2-20 (concurrent) ==="
until_complete 9000 60 "$OUT/pooling_v3_saprot/pooling_v3__saprot_650m.json" 19 \
  python -u -m esm_embedding.benchmark_v3 \
    --data_dir $DATA --model_key saprot_650m --model_dir $ROOT/models/saprot_hf \
    --arms mutant masked_wt --max_datasets 20 --extraction_batch 2 \
    --output_dir $OUT/pooling_v3_saprot \
    >> $OUT/logs/5_pooling_saprot.log 2>&1 &
P5=$!
until_complete 9000 60 "$OUT/pooling_v3_esm2/pooling_v3__esm2_650m.json" 19 \
  python -u -m esm_embedding.benchmark_v3 \
    --data_dir $DATA --model_key esm2_650m --model_dir $ROOT/models/esm2 \
    --arms mutant masked_wt --max_datasets 20 --extraction_batch 2 \
    --output_dir $OUT/pooling_v3_esm2 \
    >> $OUT/logs/9_pooling_esm2_full.log 2>&1 &
P9=$!
wait $P5; R5=$?
wait $P9; R9=$?
echo "pooling lanes finished r5=$R5 r9=$R9"

echo "=== [11] final merge ==="
python -u -m esm_embedding.merge_shards $OUT >> $OUT/logs/11_merge_shards.log 2>&1

echo "=== CHAIN2 ALL DONE ==="
echo "=== TAKEOVER_V6 ALL DONE ==="
