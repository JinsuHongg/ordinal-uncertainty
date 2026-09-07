#!/usr/bin/env bash
#SBATCH --job-name=solar_p41_geometry
#SBATCH --partition=qGPU24
#SBATCH --output=logs/phase4_1/solar_geometry_%j.out
#SBATCH --error=logs/phase4_1/solar_geometry_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ocqr
OUT=outputs/phase4_1_geometric_mechanism/solar
test ! -e "$OUT" || { echo "Refusing to overwrite $OUT" >&2; exit 3; }
mkdir -p logs/phase4_1
echo "hostname=$(hostname) job_id=${SLURM_JOB_ID:-unset} cwd=$PWD"
echo "python=$(command -v python) conda_env=${CONDA_DEFAULT_ENV:-unset}"
python scripts/run_phase4_1_geometric_mechanism.py solar \
  --solar-feature-root outputs/solar/phase3_15_direction_scale_mechanism_confirmation_retry3_qgpu24 \
  --solar-centroids outputs/solar/phase3_9_mechanism_audit/geometry/geometry.json
