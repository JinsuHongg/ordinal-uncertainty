"""Guardrails for the frozen UTKFace direction/scale confirmation."""
from pathlib import Path


def test_phase3_13_protocol_is_frozen_and_gpu_extraction_is_mandatory() -> None:
    source = Path("scripts/run_phase3_13_utkface_direction_scale.py").read_text()
    assert 'for n in ("train","validation")' in source
    assert 'torch.device("cuda:0")' in source
    assert "torch.inference_mode()" in source
    assert "num_workers=0" in source
    assert 'device="cuda:0"' in source
    assert '"test_loaded":False' in source
    assert "rop\":False" in source
    assert "target=.5*w.norm(dim=1)+.5*bw.norm(dim=1)" in source
    assert 'for n,norm in (("C",w.norm(dim=1)),("D",target))' in source
    assert "fithead(h,x,y,0)" in source
    assert "E=100" in source
    assert "BS=64" in source


def test_phase3_13_never_requests_the_test_manifest_partition() -> None:
    source = Path("scripts/run_phase3_13_utkface_direction_scale.py").read_text()
    assert 'records_for_split(rs,"test")' not in source
    assert "records_for_split(rs, 'test')" not in source
