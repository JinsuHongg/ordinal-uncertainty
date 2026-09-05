from pathlib import Path

def test_phase39_preserves_frozen_contract():
 text=Path("scripts/phase3_9_solar_mechanism_audit.py").read_text()
 assert "register_forward_pre_hook" in text
 assert "class_centroids(train" in text
 assert "l2_normalize" in text and "cosine_distances" in text
 assert "phase3_8_replay_max_abs_logit_error" in text
 assert "'backbone_updates':False" in text

def test_phase39_has_separate_no_overwrite_jobs():
 for name in ("ce_features", "rps_features", "geometry"):
  text=Path(f"scripts/slurm_phase3_9_{name}.sbatch").read_text()
  assert 'test ! -e "$OUT"' in text
