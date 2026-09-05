"""Static contract checks for the SLURM-only Phase 3.8 solar protocol."""
import ast
from pathlib import Path


def assignments(path):
    tree = ast.parse(Path(path).read_text())
    return {target.id: ast.literal_eval(node.value)
            for node in tree.body if isinstance(node, ast.Assign)
            for target in node.targets if isinstance(target, ast.Name)
            and target.id in {"CHANNELS", "CHANNEL_INDICES", "MAX_EPOCHS"}}


def test_frozen_channel_contract_is_imported_from_audited_implementation():
    text = Path("scripts/phase3_8_solar_confirmation.py").read_text()
    assert "CHANNELS, CHANNEL_INDICES" in text
    assert "phase3_7a_solar_3ch" in text
    assert "Zarr time ns - 8 hours" in text
    assert "minimum validation CE" in text
    assert "minimum validation RPS" in text


def test_training_jobs_are_separate_gpu_sbatch_jobs_with_no_overwrite_guard():
    for method in ("ce", "rps"):
        text = Path(f"scripts/slurm_phase3_8_{method}.sbatch").read_text()
        assert "#SBATCH --gres=gpu:1" in text
        assert "#SBATCH --cpus-per-task=8" in text
        assert "#SBATCH --mem=64G" in text
        assert "#SBATCH --time=24:00:00" in text
        assert "test ! -e \"$OUT\"" in text
        assert f"--method {method}" in text
