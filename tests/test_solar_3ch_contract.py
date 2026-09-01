"""Frozen Phase-3.7A channel contract without requiring cluster data access."""
import ast
from pathlib import Path

def constants():
    tree=ast.parse(Path("scripts/phase3_7a_solar_3ch.py").read_text())
    values={}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"CHANNELS", "CHANNEL_INDICES"}:
                    values[target.id]=ast.literal_eval(node.value)
    return values

def test_frozen_source_indices_preserve_requested_order():
    values=constants()
    source=("aia94","aia131","aia171","aia193","aia211","aia304","aia335","aia1600","hmi_m","hmi_bx","hmi_by","hmi_bz","hmi_v")
    assert values["CHANNEL_INDICES"] == (8,7,1)
    assert tuple(source[i] for i in values["CHANNEL_INDICES"]) == values["CHANNELS"] == ("hmi_m","aia1600","aia131")
