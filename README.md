# Human-Process Control Decoding (HPCD)

This repository provides the **source-code release** for *Human-Process Control Decoding: Steering Language Models Toward Human-Written Token-Choice Distributions*, prepared for anonymous peer review.

**Release status:** frozen scientific source and small non-identifying artifacts have been collected, sanitized for review, and checked offline. This package **has not been independently reproduced end-to-end on the published RAID evaluation splits**. Do not interpret a passing offline check as a reproduction of paper tables.

## Method overview

HPCD estimates human token-choice occupancy using cumulative probability-mass states under a fixed reference language model. It steers a paraphrase decoding distribution toward domain-calibrated human typicality, without detector feedback, then applies detector-free semantic/factual validity checks before process-aware candidate selection. The final method uses order `n=0`, beta `0.22`, temperature `1.0`, top-p `0.95`, four candidate slots, two attempts per slot and the unchanged-Source fallback when no candidate passes.

## Essential files

| Component | Source path |
|---|---|
| Human occupancy model artifact | `results/hjtm/final_hjtm_v1/hjtm_final_v1.json` |
| Frozen scientific parameters | `configs/hpcd/FINAL_METHOD_FREEZE.json` |
| Baseline HJTM and controller | `code/hpcd/hpcd_controller_v1.py` |
| Reference-aligned projected-typicality controller | `code/hpcd/hpcd_reference_aligned_v2.py` |
| Primary candidate-generation driver | `code/hpcd/run_fresh95_final_candidates_v1.py` |
| Replication candidate-generation driver | `code/confirm48/run_confirm48_final_candidates_v1.py` |
| Development provenance-pinned driver | `code/hpcd/run_t2c4_dev111_threegen_candidates_v3.py` |
| Frozen development selection scripts | `code/hpcd/analyze_t2c4_dev111_quality_selection_v1.py`, `code/hpcd/finalize_t2c4_dev111_surface_entity_selection_v2.py` |
| Replication quality and scoring | `code/confirm48/run_confirm48_quality_selection_v1.py`, `code/confirm48/run_confirm48_primary_t0_v1.py` |
| NLI and entity evaluation | `code/final_analysis/run_frozen_bidirectional_nli_v1.py`, `code/final_analysis/run_frozen_entity_preservation_v1.py` |
| Statistical analysis | `code/final_analysis_v2/finalize_experimental_package_v1.py` |

Files retain experiment-era names (`fresh95` = Primary, `confirm48` = Replication) to preserve traceability. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for prerequisites, limits, and example commands.

## Installation and offline verification

Python 3.11 was used in the recorded macOS environment. The published generation drivers may require additional packages and hardware-specific versions.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-minimal.txt
python scripts/verify_release.py
python -m unittest discover -s tests -v
```

`configs/environment.yml` is an archived environment record, **not guaranteed to solve on all platforms**. Model weights are **not bundled**. Obtain required checkpoints from their providers subject to their licenses and access terms.

## Generation and frozen input requirements

For a held-out run, you must supply the original frozen split manifest, its SHA-256 verification file, and licensed model checkpoints. Both held-out drivers read environment variables. Example invocation:

```bash
export HPCD_PROJECT_ROOT="$PWD"
export HPCD_MODEL_PATH="/your/local/Qwen2.5-3B-Instruct"
export HPCD_MANIFEST="/your/local/frozen_primary_manifest.json"
export HPCD_MANIFEST_SHA_FILE="/your/local/frozen_primary_manifest.sha256"
export HPCD_OUT_DIR="/your/local/primary_outputs"
python code/hpcd/run_fresh95_final_candidates_v1.py
```

Use `code/confirm48/run_confirm48_final_candidates_v1.py` for Replication with the matching Replication manifest. **Do not substitute an arbitrary new split if your goal is to reproduce the paper's numbers.** Before expensive generation, inspect the manifest schema and execute a smoke run. The model locations embedded in older experimental utilities are placeholders and require local configuration. Some scripts depend on locally generated intermediate outputs not included here.

## External baselines and detector integrations

This release includes the integration and evaluation scripts written for CoPA, DIPPER, MAGE, Fast-DetectGPT, RADAR, and SurpMark. **It does not redistribute third-party repositories, model checkpoints, or their weights.** Obtain upstream source code and follow the original licenses, then configure model paths and working directories locally. Baseline wrappers are historical experimental scripts and may require source-specific directory adaptation. See [`docs/EXTERNAL_DEPENDENCIES.md`](docs/EXTERNAL_DEPENDENCIES.md).

## Reproducibility, anonymity, and data

* The frozen HJTM model is bundled, along with the small freeze manifests and figure-data tables. Source and dataset files are excluded pending license/redistribution checks.
* The release snapshots are path-sanitized. As a result, published source-file hashes differ from the original frozen-code SHA-256 values. The original hashes are retained in the freeze manifest; see [`docs/PROVENANCE.md`](docs/PROVENANCE.md).
* This repository should be linked in an anonymous submission **only through a checked anonymous mirror**. Do not include direct personal GitHub URLs in the submission materials.
* The reader must not infer that the 20-evaluator human study can be reproduced from this source snapshot: raw individual evaluation forms are not included.

## License

**License to be selected by the code owners before public release.** Until then, no general permission to reuse or redistribute the original HPCD source is granted by this repository. Third-party source remains subject to its respective licensing terms. Do not mistake a missing license for open-source permission.
