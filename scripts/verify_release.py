#!/usr/bin/env python3
"""Read-only checks for the included paper-frozen HJTM artifact and source."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
model = ROOT / "results/hjtm/final_hjtm_v1/hjtm_final_v1.json"
freeze = ROOT / "configs/hpcd/FINAL_METHOD_FREEZE.json"
artifact_hash = hashlib.sha256(model.read_bytes()).hexdigest()
metadata = json.loads(freeze.read_text())
expected_hash = metadata["method"]["human_process"]["canonical_hjtm_sha256"]
assert artifact_hash == expected_hash, (artifact_hash, expected_hash)
artifact = json.loads(model.read_text())
assert artifact["model_name"] == "HJTM_FINAL_V1"
assert artifact["state_definition"]["B"] == 6
assert artifact["state_definition"]["E"] == 3
assert len(artifact["state_definition"]["mass_boundaries"]) == 5
assert len(artifact["state_definition"]["entropy_boundaries"]) == 2
assert abs(metadata["method"]["beta"] - 0.22) < 1e-12
assert metadata["method"]["controller_order"] == 0
assert metadata["decoding"]["K"] == 4
assert metadata["decoding"]["attempts_per_slot"] == 2
names = []
for p in (ROOT / "code").rglob("*.py"):
    ast.parse(p.read_text(encoding="utf8"), filename=str(p))
    names.append(p)
for p in (ROOT / "results").rglob("*.py"):
    ast.parse(p.read_text(encoding="utf8"), filename=str(p))
    names.append(p)
print("PASS: HJTM artifact SHA256 verified:", artifact_hash)
print("PASS: 6 mass bins, 3 entropy regions, frozen decoder settings")
print("PASS: Python syntax checks", len(names), "source files")
print("NOTE: No inference, generation, external detector or end-to-end reproducibility tests run.")
