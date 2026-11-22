#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# E4 prototype-count ablation: 50 reference/training objects, 20% extra validation objects.
OUT="results/e4_prototype_ablation_ref50_dynamic_geometry"
REFERENCES=50
TARGETS=500

if [[ $# -ne 1 || ! "$1" =~ ^[0-4]$ ]]; then
  echo "Usage: bash src/e4_revised/run_e4_prototype_ablation_50.sh SEED" >&2
  echo "SEED must be 0, 1, 2, 3, or 4" >&2
  exit 2
fi
seed="$1"
mkdir -p "$OUT/seed_${seed}"

for prototypes in 4 8 16 32 64; do
  stem="prototypes_${prototypes}"
  seed_out="$OUT/seed_${seed}/${stem}.csv"
  log="${seed_out%.csv}.log"
  if [[ -e "$seed_out" || -e "$log" ]]; then
    echo "ERROR: output already exists: $seed_out or $log" >&2
    exit 1
  fi
  echo "===== seed=$seed references=$REFERENCES prototypes_per_object=$prototypes total_prototypes=$((REFERENCES * prototypes)) ====="
  python -u -m src.e4_revised.ca_random_connectivity_reference_ablation \
    --grasps-dir grasps \
    --references "$REFERENCES" --reference-pool 1600 \
    --training-mode reference \
    --targets "$TARGETS" --seeds "$seed" --split-seed 4242 \
    --budget 100 --max-grasps 1000 --prototypes-per-object "$prototypes" \
    --dynamic-geometry --medoids 8 --neighbors 16 --hidden 64 --epochs 30 \
    --train-queries 128 --train-context 99 --batch-size 256 \
    --val-fraction 0.2 --patience 8 --device mps \
    --out "$seed_out" 2>&1 | tee "$log"
done

echo "========== FINISHED SEED $seed =========="
echo "Merge outputs only after parallel runs have completed."
