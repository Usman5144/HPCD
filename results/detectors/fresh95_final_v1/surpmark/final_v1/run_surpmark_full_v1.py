#!/usr/bin/env python3
from __future__ import annotations

import os

import hashlib
import json
import math
import pickle
import time
from collections import Counter
from pathlib import Path

import torch

from surpmark.compute_gjs import (
    get_token_features_from_input,
    discretize_sequence,
    calculate_single_sequence_transition_counts,
    convert_counts_to_probabilities,
    calculate_average_gutman_h_score,
)


ROOT = Path(os.environ.get("HPCD_PROJECT_ROOT", str(Path(__file__).resolve().parents[5])))

INPUT = ROOT / (
    "results/detectors/fresh95_final_v1/inputs/"
    "fresh95_detector_input_v1.jsonl"
)

CALIB = ROOT / (
    "results/detectors/fresh95_final_v1/surpmark/reference_v1/"
    "human_threshold_calibration_v1.jsonl"
)

STATE = ROOT / (
    "results/detectors/fresh95_final_v1/surpmark/smoke_v1/"
    "reference_state_v1.pkl"
)

OUTDIR = ROOT / (
    "results/detectors/fresh95_final_v1/surpmark/final_v1"
)

FINAL_OUT = OUTDIR / "scores_v1.jsonl"
CALIB_OUT = OUTDIR / "human_calibration_scores_v1.jsonl"
MANIFEST_OUT = OUTDIR / "run_manifest_v1.json"


EXPECTED_INPUT_SHA = (
    "5e5bc0f60923b43675be252f6c3651e7"
    "0338c46ac4ee1a6ba3d211e8f0cdfcb5"
)

EXPECTED_CALIB_SHA = (
    "ee54e61cacdc7d06742aadd93438078fa"
    "7f60c053c69131bac116ea911161615"
)

EXPECTED_STATE_SHA = (
    "22daab3137e6b018cbeb7ad680ca0a77e"
    "e93e6acacf94002029c49d241f5cd7d"
)

MODEL_NAME = "gpt2-large"
MAX_TEXT_LENGTH = 1024
BATCH_SIZE = 1
TRANSITION_ORDER = 1


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def scalar(x):
    return float(x)


print("=" * 96)
print("SURPMARK FULL FROZEN SCORING")
print("=" * 96)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

for path in (
    FINAL_OUT,
    CALIB_OUT,
    MANIFEST_OUT,
):
    if path.exists():
        raise RuntimeError(
            f"Refusing overwrite: {path}"
        )


# =====================================================================
# 1. Frozen artifact verification
# =====================================================================

assert sha256_file(INPUT) == EXPECTED_INPUT_SHA
assert sha256_file(CALIB) == EXPECTED_CALIB_SHA
assert sha256_file(STATE) == EXPECTED_STATE_SHA

fresh = load_jsonl(INPUT)
calib = load_jsonl(CALIB)

assert len(fresh) == 523
assert len(calib) == 280

expected_conditions = Counter({
    "human": 48,
    "original": 95,
    "vanilla_t10": 95,
    "vanilla_t11": 95,
    "copa3b": 95,
    "hpcd_n0": 95,
})

assert Counter(
    r["condition"]
    for r in fresh
) == expected_conditions

print()
print("===== FROZEN INPUTS =====")
print("Fresh95 rows       :", len(fresh))
print("calibration humans :", len(calib))
print("input SHA          :", sha256_file(INPUT))
print("calibration SHA    :", sha256_file(CALIB))
print("reference state SHA:", sha256_file(STATE))
print("FROZEN_INPUTS=PASS")


# =====================================================================
# 2. Frozen SurpMark state
# =====================================================================

with STATE.open("rb") as f:
    state = pickle.load(f)

assert state["model_name"] == MODEL_NAME
assert state["max_text_length"] == MAX_TEXT_LENGTH
assert state["transition_order"] == TRANSITION_ORDER
assert state["k_means"] == 8
assert state["seed"] == 25

bin_boundaries = state["bin_boundaries"]

p_transition_prob = state[
    "p_transition_prob"
]

q_transition_prob = state[
    "q_transition_prob"
]

length_N_human = state[
    "length_N_human"
]

length_N_machine = state[
    "length_N_machine"
]

print()
print("===== FROZEN SURPMARK CONTRACT =====")
print("model              :", MODEL_NAME)
print("max text length    :", MAX_TEXT_LENGTH)
print("KMeans states      :", state["k_means"])
print("transition order   :", TRANSITION_ORDER)
print("seed               :", state["seed"])
print("human ref tokens   :", length_N_human)
print("machine ref tokens :", length_N_machine)
print("boundaries         :", bin_boundaries)
print("REFERENCE_STATE=PASS")


# =====================================================================
# 3. CUDA / local model
# =====================================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA unavailable"
    )

print()
print("===== DEVICE =====")
print("GPU:", torch.cuda.get_device_name(0))


# =====================================================================
# 4. Extract surprisal features once for all 803 documents
# =====================================================================

rows = fresh + calib

texts = [
    r["text"]
    for r in rows
]

print()
print("=" * 96)
print("EXTRACT FEATURES FOR 523 FINAL + 280 CALIBRATION")
print("=" * 96)

start = time.perf_counter()

features, feature_info = (
    get_token_features_from_input(
        None,
        None,
        texts,
        MODEL_NAME,
        MAX_TEXT_LENGTH,
        0,
        name="fresh95_plus_human_calibration",
        verbose=True,
        batch_size=BATCH_SIZE,
        use_float64=False,
    )
)

assert len(features) == 803

feature_seconds = (
    time.perf_counter()
    - start
)

print()
print("feature docs       :", len(features))
print(
    "feature extraction:",
    round(feature_seconds, 3),
    "sec",
)


# =====================================================================
# 5. Exact native SurpMark document score
#
# Native orientation frozen before outcomes:
#
#   score =
#       distance(test, human reference)
#       -
#       distance(test, machine reference)
#
# Higher = more machine-like.
# =====================================================================

score_start = time.perf_counter()

scored = []

for idx, (src, feat) in enumerate(
    zip(rows, features)
):

    states = discretize_sequence(
        feat,
        bin_boundaries,
    )

    transition_counts = (
        calculate_single_sequence_transition_counts(
            states,
            TRANSITION_ORDER,
        )
    )

    transition_prob = (
        convert_counts_to_probabilities(
            transition_counts
        )
    )

    distance_machine = (
        calculate_average_gutman_h_score(
            q_transition_prob,
            transition_prob,
            length_N_machine,
            len(states),
        )
    )

    distance_human = (
        calculate_average_gutman_h_score(
            p_transition_prob,
            transition_prob,
            length_N_human,
            len(states),
        )
    )

    score = (
        distance_human
        -
        distance_machine
    )

    vals = [
        distance_machine,
        distance_human,
        score,
    ]

    if not all(
        math.isfinite(float(x))
        for x in vals
    ):
        raise RuntimeError(
            f"Non-finite score at row {idx}"
        )

    record = {
        "input_index":
            idx,

        "base_id":
            src.get("base_id"),

        "sample_id":
            src.get("sample_id"),

        "source_sample_id":
            src.get("source_sample_id"),

        "source_id":
            src.get("source_id"),

        "domain":
            src.get("domain"),

        "source_generator":
            src.get("source_generator"),

        "generator_key":
            src.get("generator_key"),

        "condition":
            src.get(
                "condition",
                "human_threshold_calibration",
            ),

        "text_sha256":
            src.get(
                "text_sha256"
            )
            or sha256_text(src["text"]),

        "state_length":
            len(states),

        "distance_to_machine_reference":
            scalar(distance_machine),

        "distance_to_human_reference":
            scalar(distance_human),

        "surpmark_score":
            scalar(score),

        "analysis_score":
            scalar(score),

        "orientation":
            "higher = more machine-like",

        "status":
            "ok",
    }

    scored.append(record)


score_seconds = (
    time.perf_counter()
    - score_start
)

assert len(scored) == 803


# =====================================================================
# 6. Split canonical evaluation and independent calibration
# =====================================================================

fresh_scored = scored[:523]
calib_scored = scored[523:]

assert len(fresh_scored) == 523
assert len(calib_scored) == 280

assert Counter(
    r["condition"]
    for r in fresh_scored
) == expected_conditions

assert all(
    r["status"] == "ok"
    for r in fresh_scored
)

assert all(
    r["status"] == "ok"
    for r in calib_scored
)


with FINAL_OUT.open(
    "w",
    encoding="utf-8",
) as f:

    for r in fresh_scored:
        f.write(
            json.dumps(
                r,
                ensure_ascii=False,
            )
            + "\n"
        )


with CALIB_OUT.open(
    "w",
    encoding="utf-8",
) as f:

    for r in calib_scored:
        f.write(
            json.dumps(
                r,
                ensure_ascii=False,
            )
            + "\n"
        )


# =====================================================================
# 7. Alignment checks
# =====================================================================

for src, out in zip(
    fresh,
    fresh_scored,
):
    expected_text_sha = (
        src.get("text_sha256")
        or sha256_text(src["text"])
    )

    assert (
        out["text_sha256"]
        ==
        expected_text_sha
    )


for src, out in zip(
    calib,
    calib_scored,
):
    expected_text_sha = (
        src.get("text_sha256")
        or sha256_text(src["text"])
    )

    assert (
        out["text_sha256"]
        ==
        expected_text_sha
    )


# Exact source-fallback diagnostic only.
# This does NOT alter or exclude any sample.

fresh_by_key = {}

for r in fresh:
    if r["condition"] == "original":
        fresh_by_key[
            r["source_sample_id"]
        ] = r

fallback_counts = Counter()

for r in fresh:
    if r["condition"] not in {
        "vanilla_t10",
        "vanilla_t11",
        "copa3b",
        "hpcd_n0",
    }:
        continue

    source = fresh_by_key.get(
        r["source_sample_id"]
    )

    if source is None:
        continue

    if r["text"] == source["text"]:
        fallback_counts[
            r["condition"]
        ] += 1


print()
print("===== COMPLETION =====")
print("canonical scored    :", len(fresh_scored))
print("calibration scored  :", len(calib_scored))
print("failures            : 0")
print("scoring seconds     :", round(score_seconds, 3))

print()
print(
    "exact source-text matches "
    "(diagnostic only):",
    dict(fallback_counts),
)


# =====================================================================
# 8. Manifest
# =====================================================================

manifest = {
    "status":
        "PASS",

    "detector":
        "SurpMark",

    "implementation":
        "authors' released SurpMark code with thin per-document scoring wrapper",

    "canonical_input_rows":
        523,

    "independent_human_calibration_rows":
        280,

    "canonical_input_sha256":
        sha256_file(INPUT),

    "human_calibration_input_sha256":
        sha256_file(CALIB),

    "reference_state_sha256":
        sha256_file(STATE),

    "model_name":
        MODEL_NAME,

    "model_resolution":
        (
            "local offline symlink to verified "
            "/path/to/models/gpt2-large-surpmark"
        ),

    "max_text_length":
        MAX_TEXT_LENGTH,

    "k_means":
        8,

    "transition_order":
        1,

    "seed":
        25,

    "batch_size":
        1,

    "score_definition":
        (
            "distance_to_human_reference "
            "- distance_to_machine_reference"
        ),

    "orientation":
        "higher = more machine-like",

    "threshold_tuned_on_fresh95":
        False,

    "reference_fit_reused":
        True,

    "reference_refit":
        False,

    "outcome_based_selection":
        False,

    "feature_extraction_seconds":
        feature_seconds,

    "score_computation_seconds":
        score_seconds,

    "canonical_scores_file":
        str(
            FINAL_OUT.relative_to(ROOT)
        ),

    "canonical_scores_sha256":
        sha256_file(FINAL_OUT),

    "calibration_scores_file":
        str(
            CALIB_OUT.relative_to(ROOT)
        ),

    "calibration_scores_sha256":
        sha256_file(CALIB_OUT),

    "exact_source_text_match_counts":
        dict(fallback_counts),

    "gpu":
        torch.cuda.get_device_name(0),
}


MANIFEST_OUT.write_text(
    json.dumps(
        manifest,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)


print()
print("===== FINAL ARTIFACTS =====")
print(
    "canonical SHA :",
    sha256_file(FINAL_OUT),
)
print(
    "calibration SHA:",
    sha256_file(CALIB_OUT),
)
print(
    "manifest SHA  :",
    sha256_file(MANIFEST_OUT),
)

print()
print("=" * 96)
print("SURPMARK_FULL_SCORING=PASS")
print("=" * 96)
