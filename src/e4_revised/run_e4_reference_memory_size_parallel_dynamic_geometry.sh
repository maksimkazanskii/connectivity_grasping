#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# Adaptive E4: references = training objects; additional 20% validation objects.
OUT="results/e4_reference_scaling_16"
PROTOTYPES_PER_OBJECT=16
TARGETS=500

if [[ $# -ne 1 || ! "$1" =~ ^[0-4]$ ]]; then
  echo "Usage: bash src/e4_revised/run_e4_reference_memory_size_parallel.sh SEED" >&2
  echo "SEED must be 0, 1, 2, 3, or 4" >&2
  exit 2
fi
seed="$1"
mkdir -p "$OUT/seed_${seed}"

for references in  6 12 25 50 100 200 400 800 ; do
  stem="scale_${references}"
  seed_out="$OUT/seed_${seed}/${stem}.csv"
  log="${seed_out%.csv}.log"
  if [[ -e "$seed_out" || -e "$log" ]]; then
    echo "ERROR: output already exists: $seed_out or $log" >&2
    exit 1
  fi
  echo "===== seed=$seed references=$references train=$references prototypes_per_object=$PROTOTYPES_PER_OBJECT ====="
  python -u -m src.e4_revised.ca_random_connectivity_reference_ablation \
    --grasps-dir grasps \
    --references "$references" --reference-pool 1600 \
    --training-mode reference \
    --targets "$TARGETS" --seeds "$seed" --split-seed 4242 \
    --budget 100 --max-grasps 1000 --prototypes-per-object "$PROTOTYPES_PER_OBJECT" \
    --dynamic-geometry --medoids 8 --neighbors 16 --hidden 64 --epochs 30 \
    --train-queries 128 --train-context 99 --batch-size 256 \
    --val-fraction 0.2 --patience 8 --device mps \
    --out "$seed_out" 2>&1 | tee "$log"
done

echo "========== FINISHED SEED $seed =========="
echo "Merge outputs only after parallel runs have completed."
