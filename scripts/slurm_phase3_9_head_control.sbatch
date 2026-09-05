#!/usr/bin/env bash
#SBATCH --job-name=solar_p39_head_control
#SBATCH --output=logs/phase3_9/head_control_%j.out
#SBATCH --error=logs/phase3_9/head_control_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate ocqr
OUT=outputs/solar/phase3_9_mechanism_audit/frozen_head_controls
test ! -e "$OUT" || { echo "Refusing to overwrite $OUT" >&2; exit 3; }
mkdir -p logs/phase3_9
python scripts/phase3_9_frozen_head_control.py --method ce --out "$OUT/ce"
python scripts/phase3_9_frozen_head_control.py --method rps --out "$OUT/rps"
