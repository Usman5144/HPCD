#!/usr/bin/env python3
from __future__ import annotations

import os

import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(os.environ.get("HPCD_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))

sys.path.insert(
    0,
    str(ROOT / "code/hpcd"),
)

from hpcd_controller_v1 import HJTMModel
from reanalyze_t2b7_common_metrics_v1 import score_tokens


# ==============================================================================
# Frozen inputs
# ==============================================================================

RAW = (
    ROOT
    / "results/hpcd/"
      "t2c4_dev111_threegen_candidates_v3/"
      "candidates.jsonl"
)

SCORED = (
    ROOT
    / "results/hpcd/"
      "t2c4_dev111_quality_selection_v1/"
      "scored_candidates.jsonl"
)

RESCORED = (
    ROOT
    / "results/hpcd/"
      "t2c4_dev111_surface_entity_selection_v2/"
      "rescored_candidates.jsonl"
)

SELECTED = (
    ROOT
    / "results/hpcd/"
      "t2c4_dev111_surface_entity_selection_v2/"
      "selected_outputs.jsonl"
)

COMMON_SELECTED = (
    ROOT
    / "results/hpcd/"
      "t2c4_dev111_common_rescore_v1.json"
)

MODEL_PATH = Path(
    "/path/to/huggingface/hub/"
    "models--Qwen--Qwen2.5-3B-Instruct/"
    "snapshots/"
    "aa8e72537993ba99e69dfaafa59ed015b17504d1"
)

HJTM_PATH = (
    ROOT
    / "results/hjtm/final_hjtm_v1/"
      "hjtm_final_v1.json"
)


EXPECTED_RAW_SHA = (
    "9027ef59f6c1cb3dc70e1f6155871ba"
    "46b3fdedf9dd4df0ddaa46d101763eb07"
)

EXPECTED_SCORED_SHA = (
    "5ca94afa38220a1d6b02df5b02708127"
    "7231b15f4756bc10c90c578cc88fd1e0"
)

EXPECTED_COMMON_SELECTED_SHA = (
    "2510585baacc293959677d85089cd02ba"
    "49d337f4a651ae63de533eb89726c46"
)


# ==============================================================================
# Outputs
# ==============================================================================

OUT = (
    ROOT
    / "results/hpcd/factorial_dev111"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

FIG_DIR = (
    ROOT
    / "figures/hpcd"
)

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

COMMON_CANDIDATES = (
    OUT
    / "candidate_common_n0_scores_v1.jsonl"
)

COMMON_CANDIDATE_SUMMARY = (
    OUT
    / "candidate_common_n0_summary_v1.json"
)

PAIR_CSV = (
    OUT
    / "candidate_pair_analysis.csv"
)

FOUR_CSV = (
    OUT
    / "four_condition_outputs.csv"
)

SUMMARY_JSON = (
    OUT
    / "factorial_summary.json"
)

BOOT_CSV = (
    OUT
    / "cluster_bootstrap.csv"
)

AUDIT_JSON = (
    OUT
    / "artifact_audit.json"
)

FIG_PDF = (
    FIG_DIR
    / "decoding_vs_selection_factorial.pdf"
)


# ==============================================================================
# Helpers
# ==============================================================================

def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def text_sha(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def load_jsonl(path):
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                rows.append(
                    json.loads(line)
                )
            except Exception as exc:
                raise RuntimeError(
                    f"{path}: bad JSON line {ln}: {exc}"
                )

    return rows


def stable_key(row):
    return (
        str(row["instance_id"]),
        int(row["slot"]),
        int(row["attempt"]),
        str(row["condition"]),
    )


def pair_key(row):
    return (
        str(row["instance_id"]),
        int(row["slot"]),
        int(row["attempt"]),
    )


def base_id(row):
    return str(row["sample_id"])


def safe_float(x):
    if x is None:
        return None

    try:
        x = float(x)
    except Exception:
        return None

    if not math.isfinite(x):
        return None

    return x


def mean(vals):
    vals = [
        float(x)
        for x in vals
        if x is not None
    ]

    if not vals:
        return None

    return float(
        statistics.fmean(vals)
    )


def median(vals):
    vals = [
        float(x)
        for x in vals
        if x is not None
    ]

    if not vals:
        return None

    return float(
        statistics.median(vals)
    )


def bootstrap_cluster(
    pair_rows,
    delta_field,
    *,
    seed=2027,
    B=20000,
):
    groups = defaultdict(list)

    for r in pair_rows:
        d = r.get(delta_field)

        if d is None:
            continue

        groups[
            r["base_sample_id"]
        ].append(
            float(d)
        )

    cluster_effects = {
        bid:
            float(
                statistics.fmean(vals)
            )
        for bid, vals
        in groups.items()
    }

    ids = sorted(
        cluster_effects
    )

    values = np.asarray(
        [
            cluster_effects[i]
            for i in ids
        ],
        dtype=float,
    )

    if len(values) == 0:
        return {
            "n_clusters": 0,
            "point_estimate": None,
            "ci": [None, None],
        }

    rng = np.random.default_rng(
        seed
    )

    draws = np.empty(
        B,
        dtype=float,
    )

    for b in range(B):
        idx = rng.integers(
            0,
            len(values),
            size=len(values),
        )

        draws[b] = (
            values[idx].mean()
        )

    lo, hi = np.quantile(
        draws,
        [0.025, 0.975],
    )

    return {
        "n_clusters":
            len(values),

        "point_estimate":
            float(values.mean()),

        "ci":
            [
                float(lo),
                float(hi),
            ],

        "p_effect_lt_zero":
            float(
                np.mean(
                    draws < 0
                )
            ),

        "bootstrap_replicates":
            B,
    }


# ==============================================================================
# Verify frozen inputs
# ==============================================================================

print("=" * 100)
print("DEV111 CANDIDATE-LEVEL COMMON n0 RESCORE + FACTORIAL")
print("=" * 100)

for p in (
    RAW,
    SCORED,
    RESCORED,
    SELECTED,
    COMMON_SELECTED,
    MODEL_PATH,
    HJTM_PATH,
):
    if not p.exists():
        raise RuntimeError(
            f"Missing required input: {p}"
        )


raw_sha = sha256_file(RAW)
scored_sha = sha256_file(SCORED)
common_selected_sha = sha256_file(
    COMMON_SELECTED
)

print()
print("raw SHA            :", raw_sha)
print("scored SHA         :", scored_sha)
print(
    "selected-common SHA:",
    common_selected_sha,
)

if raw_sha != EXPECTED_RAW_SHA:
    raise RuntimeError(
        "Frozen raw SHA mismatch"
    )

if scored_sha != EXPECTED_SCORED_SHA:
    raise RuntimeError(
        "Frozen scored SHA mismatch"
    )

if (
    common_selected_sha
    != EXPECTED_COMMON_SELECTED_SHA
):
    raise RuntimeError(
        "Frozen selected-common SHA mismatch"
    )


rows = load_jsonl(
    RESCORED
)

selected_rows = load_jsonl(
    SELECTED
)

if len(rows) != 7992:
    raise RuntimeError(
        f"Expected 7992 corrected candidates, "
        f"found {len(rows)}"
    )

if len(selected_rows) != 999:
    raise RuntimeError(
        f"Expected 999 frozen selections, "
        f"found {len(selected_rows)}"
    )


cond_counts = Counter(
    r["condition"]
    for r in rows
)

print()
print(
    "corrected candidate condition counts:",
    dict(cond_counts),
)

for cond in (
    "vanilla",
    "hpcd_n0",
    "hpcd_n1",
):
    if cond_counts[cond] != 2664:
        raise RuntimeError(
            f"{cond}: expected 2664"
        )


instances = sorted({
    r["instance_id"]
    for r in rows
})

base_ids = sorted({
    r["sample_id"]
    for r in rows
})

if len(instances) != 333:
    raise RuntimeError(
        f"Expected 333 instances, "
        f"found {len(instances)}"
    )

if len(base_ids) != 111:
    raise RuntimeError(
        f"Expected 111 base IDs, "
        f"found {len(base_ids)}"
    )


# ==============================================================================
# Candidate target set
#
# Score ONLY corrected-quality-valid Vanilla and HPCD-n0 candidates.
# ==============================================================================
target = [
    r
    for r in rows
    if (
        r["condition"]
        in (
            "vanilla",
            "hpcd_n0",
        )
        and bool(
            r["quality_valid_v2"]
        )
    )
]


target_keys = {
    stable_key(r)
    for r in target
}

if len(target_keys) != len(target):
    raise RuntimeError(
        "Duplicate target candidate keys"
    )


target_counts = Counter(
    r["condition"]
    for r in target
)

print()
print("=" * 100)
print("COMMON n0 TARGET")
print("=" * 100)
print(
    "quality-valid candidate attempts:",
    len(target),
)
print(
    "by condition:",
    dict(target_counts),
)


# ==============================================================================
# Resume existing candidate-common scores
# ==============================================================================
existing = []

if COMMON_CANDIDATES.exists():
    existing = load_jsonl(
        COMMON_CANDIDATES
    )


existing_map = {}

for r in existing:
    k = (
        r["instance_id"],
        int(r["slot"]),
        int(r["attempt"]),
        r["condition"],
    )

    if k in existing_map:
        raise RuntimeError(
            f"Duplicate existing score key: {k}"
        )

    if k not in target_keys:
        raise RuntimeError(
            f"Existing score key not in current target: {k}"
        )

    existing_map[k] = r


print(
    "already scored:",
    len(existing_map),
)

print(
    "remaining:",
    len(target)
    - len(existing_map),
)


# Build cache from previous completed rows.
score_cache = {}

for r in existing:
    score_cache[
        (
            r["domain"],
            r["text_sha256"],
        )
    ] = {
        "nll":
            float(
                r["common_nll_n0"]
            ),

        "typicality":
            float(
                r["common_typicality_n0"]
            ),

        "scored_tokens":
            int(
                r["common_scored_tokens_n0"]
            ),
    }


# ==============================================================================
# Load frozen reference scorer only if work remains
# ==============================================================================
remaining = [
    r
    for r in target
    if stable_key(r)
    not in existing_map
]


if remaining:

    if torch.backends.mps.is_available():
        device = torch.device(
            "mps"
        )
    else:
        device = torch.device(
            "cpu"
        )

    print()
    print("device:", device)

    print(
        "Loading frozen Qwen2.5-3B "
        "reference scorer..."
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
        )
    )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    )

    model.to(device)
    model.eval()

    hjtm = HJTMModel(
        HJTM_PATH
    )


    done = len(existing_map)
    total = len(target)

    with COMMON_CANDIDATES.open(
        "a",
        encoding="utf-8",
    ) as out:

        for r in sorted(
            remaining,
            key=lambda x: (
                x["instance_id"],
                x["condition"],
                int(x["slot"]),
                int(x["attempt"]),
            ),
        ):
            text = r["text"]
            domain = r["domain"]

            tsha = text_sha(text)

            ckey = (
                domain,
                tsha,
            )

            if ckey in score_cache:
                result = (
                    score_cache[
                        ckey
                    ]
                )

            else:
                token_ids = (
                    tokenizer.encode(
                        text,
                        add_special_tokens=False,
                    )
                )

                if len(token_ids) < 2:
                    raise RuntimeError(
                        "Candidate tokenized to "
                        "fewer than 2 tokens"
                    )

                result = score_tokens(
                    token_ids,
                    model=model,
                    hjtm=hjtm,
                    device=device,
                    domain=domain,
                    order=0,
                    eos_token_id=
                        tokenizer.eos_token_id,
                )

                result = {
                    "nll":
                        float(
                            result["nll"]
                        ),

                    "typicality":
                        float(
                            result[
                                "typicality"
                            ]
                        ),

                    "scored_tokens":
                        int(
                            result[
                                "scored_tokens"
                            ]
                        ),
                }

                score_cache[
                    ckey
                ] = result


            rec = {
                "instance_id":
                    r["instance_id"],

                "sample_id":
                    r["sample_id"],

                "generator_key":
                    r["generator_key"],

                "domain":
                    domain,

                "condition":
                    r["condition"],

                "slot":
                    int(
                        r["slot"]
                    ),

                "attempt":
                    int(
                        r["attempt"]
                    ),

                "seed":
                    int(
                        r["seed"]
                    ),

                "text_sha256":
                    tsha,

                "historical_order":
                    int(
                        r["order"]
                    ),

                "historical_trajectory_typicality":
                    float(
                        r[
                            "trajectory_typicality"
                        ]
                    ),

                "common_nll_n0":
                    result["nll"],

                "common_typicality_n0":
                    result[
                        "typicality"
                    ],

                "common_scored_tokens_n0":
                    result[
                        "scored_tokens"
                    ],
            }


            out.write(
                json.dumps(
                    rec,
                    ensure_ascii=False,
                )
                + "\n"
            )

            out.flush()

            done += 1

            print(
                f"[{done}/{total}] "
                f"{r['condition']:8s} "
                f"{domain:7s} "
                f"slot={r['slot']} "
                f"att={r['attempt']} "
                f"T0={result['typicality']:.6f}",
                flush=True,
            )


# ==============================================================================
# Reload + validate complete score artifact
# ==============================================================================
common_rows = load_jsonl(
    COMMON_CANDIDATES
)

common_map = {}

for r in common_rows:
    k = (
        r["instance_id"],
        int(r["slot"]),
        int(r["attempt"]),
        r["condition"],
    )

    if k in common_map:
        raise RuntimeError(
            f"Duplicate candidate common score: {k}"
        )

    common_map[k] = r


if set(common_map) != target_keys:
    missing = (
        target_keys
        - set(common_map)
    )

    extra = (
        set(common_map)
        - target_keys
    )

    raise RuntimeError(
        "Candidate common-n0 score coverage mismatch. "
        f"missing={len(missing)} "
        f"extra={len(extra)}"
    )


# ==============================================================================
# Check whether historical HPCD-n0 score and exact common-n0 score
# induce the same selected output.
#
# Do NOT require numeric equality.
# Selection equivalence is the important invariant.
# ==============================================================================
n0_score_diffs = []

for r in common_rows:
    if r["condition"] != "hpcd_n0":
        continue

    n0_score_diffs.append(
        abs(
            float(
                r[
                    "common_typicality_n0"
                ]
            )
            -
            float(
                r[
                    "historical_trajectory_typicality"
                ]
            )
        )
    )


print()
print("=" * 100)
print("HISTORICAL n0 SCORE vs COMMON n0 SCORE")
print("=" * 100)

print(
    "n:",
    len(n0_score_diffs),
)

print(
    "mean abs difference:",
    float(
        np.mean(
            n0_score_diffs
        )
    ),
)

print(
    "max abs difference:",
    float(
        np.max(
            n0_score_diffs
        )
    ),
)


candidate_summary = {
    "status":
        "COMPLETE",

    "input":
        str(
            RESCORED.relative_to(
                ROOT
            )
        ),

    "input_sha256":
        sha256_file(
            RESCORED
        ),

    "conditions":
        [
            "vanilla",
            "hpcd_n0",
        ],

    "quality_rule":
        "quality_valid_v2",

    "score_order":
        0,

    "target_rows":
        len(target),

    "condition_counts":
        dict(target_counts),

    "output_sha256":
        sha256_file(
            COMMON_CANDIDATES
        ),

    "hpcd_n0_historical_vs_common_n0": {
        "n":
            len(
                n0_score_diffs
            ),

        "mean_abs_difference":
            float(
                np.mean(
                    n0_score_diffs
                )
            ),

        "max_abs_difference":
            float(
                np.max(
                    n0_score_diffs
                )
            ),
    },
}


COMMON_CANDIDATE_SUMMARY.write_text(
    json.dumps(
        candidate_summary,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)


# ==============================================================================
# Build corrected slot-wise candidate structure
#
# Historical rule:
#   within each slot -> first quality-valid attempt
#   then:
#       FV   -> first valid slot
#       HJTM -> minimum T0 across valid slots
# ==============================================================================
row_map = {
    stable_key(r): r
    for r in rows
}


common_t0 = {
    k:
        float(
            r[
                "common_typicality_n0"
            ]
        )
    for k, r
    in common_map.items()
}


by_instance_condition_slot = (
    defaultdict(list)
)

for r in rows:
    if r["condition"] not in (
        "vanilla",
        "hpcd_n0",
    ):
        continue

    by_instance_condition_slot[
        (
            r["instance_id"],
            r["condition"],
            int(r["slot"]),
        )
    ].append(r)


def slot_valid_candidates(
    iid,
    condition,
):
    result = []

    for slot_id in range(4):

        attempts = sorted(
            by_instance_condition_slot[
                (
                    iid,
                    condition,
                    slot_id,
                )
            ],
            key=lambda r:
                int(
                    r["attempt"]
                ),
        )

        if len(attempts) != 2:
            raise RuntimeError(
                "Expected exactly two attempts "
                f"for {iid}/{condition}/slot={slot_id}"
            )

        chosen = next(
            (
                r
                for r in attempts
                if bool(
                    r[
                        "quality_valid_v2"
                    ]
                )
            ),
            None,
        )

        if chosen is not None:

            k = stable_key(
                chosen
            )

            if k not in common_t0:
                raise RuntimeError(
                    f"Missing common T0: {k}"
                )

            result.append(
                chosen
            )

    return result


def first_valid(
    iid,
    condition,
):
    vals = slot_valid_candidates(
        iid,
        condition,
    )

    if not vals:
        return None

    return sorted(
        vals,
        key=lambda r:
            int(
                r["slot"]
            ),
    )[0]


def hjtm_valid(
    iid,
    condition,
):
    vals = slot_valid_candidates(
        iid,
        condition,
    )

    if not vals:
        return None

    return min(
        vals,
        key=lambda r: (
            common_t0[
                stable_key(r)
            ],
            int(r["slot"]),
            int(r["attempt"]),
        ),
    )


# ==============================================================================
# Source/metadata map
# ==============================================================================
instance_template = {}

for r in rows:
    instance_template.setdefault(
        r["instance_id"],
        r,
    )


def factorial_record(
    iid,
    name,
    selected_candidate,
):
    template = instance_template[
        iid
    ]

    if selected_candidate is None:
        return {
            "instance_id":
                iid,

            "base_sample_id":
                template[
                    "sample_id"
                ],

            "generator_key":
                template[
                    "generator_key"
                ],

            "domain":
                template[
                    "domain"
                ],

            "factorial_condition":
                name,

            "abstained":
                True,

            "slot":
                None,

            "attempt":
                None,

            "seed":
                None,

            "common_t0":
                None,

            "output_text":
                template[
                    "source_text"
                ],
        }


    k = stable_key(
        selected_candidate
    )

    return {
        "instance_id":
            iid,

        "base_sample_id":
            selected_candidate[
                "sample_id"
            ],

        "generator_key":
            selected_candidate[
                "generator_key"
            ],

        "domain":
            selected_candidate[
                "domain"
            ],

        "factorial_condition":
            name,

        "abstained":
            False,

        "slot":
            int(
                selected_candidate[
                    "slot"
                ]
            ),

        "attempt":
            int(
                selected_candidate[
                    "attempt"
                ]
            ),

        "seed":
            int(
                selected_candidate[
                    "seed"
                ]
            ),

        "common_t0":
            float(
                common_t0[k]
            ),

        "output_text":
            selected_candidate[
                "text"
            ],
    }


# ==============================================================================
# Four conditions
# ==============================================================================
four_rows = []

for iid in instances:

    conditions = (
        (
            "V_FV",
            first_valid(
                iid,
                "vanilla",
            ),
        ),

        (
            "V_HJTM",
            hjtm_valid(
                iid,
                "vanilla",
            ),
        ),

        (
            "H_FV",
            first_valid(
                iid,
                "hpcd_n0",
            ),
        ),

        (
            "H_HJTM",
            hjtm_valid(
                iid,
                "hpcd_n0",
            ),
        ),
    )


    for name, chosen in conditions:
        four_rows.append(
            factorial_record(
                iid,
                name,
                chosen,
            )
        )


if len(four_rows) != 1332:
    raise RuntimeError(
        f"Expected 1332 factorial rows, "
        f"found {len(four_rows)}"
    )


four_lookup = {
    (
        r["instance_id"],
        r[
            "factorial_condition"
        ],
    ): r
    for r in four_rows
}


# ==============================================================================
# Frozen-selection reconstruction invariants
# ==============================================================================
frozen = {
    (
        r["instance_id"],
        r["condition"],
    ):
        r["output_text"]
    for r in selected_rows
}


v_mismatch = []
h_mismatch = []


for iid in instances:

    if (
        four_lookup[
            (
                iid,
                "V_FV",
            )
        ]["output_text"]
        !=
        frozen[
            (
                iid,
                "vanilla",
            )
        ]
    ):
        v_mismatch.append(
            iid
        )


    if (
        four_lookup[
            (
                iid,
                "H_HJTM",
            )
        ]["output_text"]
        !=
        frozen[
            (
                iid,
                "hpcd_n0",
            )
        ]
    ):
        h_mismatch.append(
            iid
        )


print()
print("=" * 100)
print("FROZEN SELECTION RECONSTRUCTION")
print("=" * 100)

print(
    "V_FV vs frozen Vanilla mismatches:",
    len(v_mismatch),
)

print(
    "H_HJTM(common n0) vs frozen n0 mismatches:",
    len(h_mismatch),
)


# Vanilla MUST reproduce exactly.
if v_mismatch:
    raise RuntimeError(
        "V_FV failed to reproduce frozen Vanilla. "
        f"Examples={v_mismatch[:5]}"
    )


# If common n0 changes historical n0 selection,
# STOP before claiming a factorial.
if h_mismatch:
    print()
    print(
        "STOP: common-n0 selector does not "
        "exactly reproduce historical HPCD-n0 "
        "selection."
    )

    print(
        "Example mismatches:",
        h_mismatch[:10],
    )

    raise RuntimeError(
        "H_HJTM selection-equivalence gate failed"
    )


# ==============================================================================
# Raw matched-candidate decoder effect
#
# All same source / slot / attempt keys.
# Require BOTH candidate attempts to pass corrected quality.
# ==============================================================================
vanilla_rows = {
    pair_key(r): r
    for r in rows
    if r[
        "condition"
    ] == "vanilla"
}

hpcd_rows = {
    pair_key(r): r
    for r in rows
    if r[
        "condition"
    ] == "hpcd_n0"
}


if len(vanilla_rows) != 2664:
    raise RuntimeError(
        "Expected 2664 Vanilla candidate keys"
    )

if len(hpcd_rows) != 2664:
    raise RuntimeError(
        "Expected 2664 HPCD candidate keys"
    )

if set(vanilla_rows) != set(hpcd_rows):
    raise RuntimeError(
        "Vanilla/HPCD candidate keys differ"
    )


pair_rows = []

seed_mismatch = 0


for pk in sorted(
    vanilla_rows
):
    v = vanilla_rows[
        pk
    ]

    h = hpcd_rows[
        pk
    ]

    if (
        int(v["seed"])
        !=
        int(h["seed"])
    ):
        seed_mismatch += 1


    v_valid = bool(
        v[
            "quality_valid_v2"
        ]
    )

    h_valid = bool(
        h[
            "quality_valid_v2"
        ]
    )


    delta = None

    if (
        v_valid
        and h_valid
    ):
        vt0 = common_t0[
            stable_key(v)
        ]

        ht0 = common_t0[
            stable_key(h)
        ]

        delta = (
            ht0
            -
            vt0
        )


    pair_rows.append({
        "instance_id":
            v["instance_id"],

        "base_sample_id":
            v["sample_id"],

        "generator_key":
            v["generator_key"],

        "domain":
            v["domain"],

        "slot":
            int(
                v["slot"]
            ),

        "attempt":
            int(
                v["attempt"]
            ),

        "seed":
            int(
                v["seed"]
            ),

        "vanilla_valid":
            v_valid,

        "hpcd_valid":
            h_valid,

        "both_valid":
            bool(
                v_valid
                and h_valid
            ),

        "vanilla_common_t0":
            (
                common_t0[
                    stable_key(v)
                ]
                if v_valid
                else None
            ),

        "hpcd_common_t0":
            (
                common_t0[
                    stable_key(h)
                ]
                if h_valid
                else None
            ),

        "delta_t0_hpcd_minus_vanilla":
            delta,
    })


if seed_mismatch:
    raise RuntimeError(
        f"Matched seed mismatches: {seed_mismatch}"
    )


with PAIR_CSV.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            pair_rows[0]
        ),
    )

    writer.writeheader()
    writer.writerows(
        pair_rows
    )


# ==============================================================================
# Factorial CSV
# ==============================================================================
with FOUR_CSV.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            four_rows[0]
        ),
    )

    writer.writeheader()
    writer.writerows(
        four_rows
    )


# ==============================================================================
# Summaries
# ==============================================================================
raw_deltas = [
    r[
        "delta_t0_hpcd_minus_vanilla"
    ]
    for r in pair_rows
    if r[
        "delta_t0_hpcd_minus_vanilla"
    ] is not None
]


raw_summary = {
    "all_matched_seed_pairs":
        len(pair_rows),

    "both_quality_valid_pairs":
        len(raw_deltas),

    "mean_delta":
        mean(raw_deltas),

    "median_delta":
        median(raw_deltas),

    "improved_negative":
        sum(
            x < 0
            for x in raw_deltas
        ),

    "worsened_positive":
        sum(
            x > 0
            for x in raw_deltas
        ),

    "ties":
        sum(
            x == 0
            for x in raw_deltas
        ),
}


factorial_map = {
    name: {}
    for name in (
        "V_FV",
        "V_HJTM",
        "H_FV",
        "H_HJTM",
    )
}


for r in four_rows:
    factorial_map[
        r[
            "factorial_condition"
        ]
    ][
        r["instance_id"]
    ] = r


contrasts = {
    "decoder_effect_first_valid":
        (
            "H_FV",
            "V_FV",
        ),

    "selector_effect_under_vanilla":
        (
            "V_HJTM",
            "V_FV",
        ),

    "decoder_effect_same_hjtm_selector":
        (
            "H_HJTM",
            "V_HJTM",
        ),

    "full_pipeline":
        (
            "H_HJTM",
            "V_FV",
        ),
}


contrast_summary = {}
contrast_pair_rows = {}


for name, (
    a,
    b,
) in contrasts.items():

    vals = []
    rows_here = []

    for iid in instances:

        ra = factorial_map[
            a
        ][iid]

        rb = factorial_map[
            b
        ][iid]

        if (
            ra["abstained"]
            or rb["abstained"]
        ):
            continue

        delta = (
            float(
                ra["common_t0"]
            )
            -
            float(
                rb["common_t0"]
            )
        )

        vals.append(
            delta
        )

        rows_here.append({
            "instance_id":
                iid,

            "base_sample_id":
                ra[
                    "base_sample_id"
                ],

            "generator_key":
                ra[
                    "generator_key"
                ],

            "domain":
                ra[
                    "domain"
                ],

            "delta":
                delta,
        })


    contrast_pair_rows[
        name
    ] = rows_here


    contrast_summary[
        name
    ] = {
        "A":
            a,

        "B":
            b,

        "n":
            len(vals),

        "mean_delta":
            mean(vals),

        "median_delta":
            median(vals),

        "improved_negative":
            sum(
                x < 0
                for x in vals
            ),

        "worsened_positive":
            sum(
                x > 0
                for x in vals
            ),

        "ties":
            sum(
                x == 0
                for x in vals
            ),
    }


# ==============================================================================
# Cluster bootstrap for Step-1 contrasts
#
# This is a Step-1 robustness calculation.
# The formal domain-stratified ITT bootstrap remains Step 2.
# ==============================================================================
boot_records = []


raw_boot = bootstrap_cluster(
    pair_rows,
    "delta_t0_hpcd_minus_vanilla",
)


boot_records.append({
    "contrast":
        "raw_candidate_decoder",

    **raw_boot,
})


for name, rows_here in (
    contrast_pair_rows.items()
):

    tmp = [
        {
            **r,
            "delta_value":
                r["delta"],
        }
        for r in rows_here
    ]

    b = bootstrap_cluster(
        tmp,
        "delta_value",
    )

    boot_records.append({
        "contrast":
            name,
        **b,
    })


with BOOT_CSV.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:

    fields = [
        "contrast",
        "n_clusters",
        "point_estimate",
        "ci",
        "p_effect_lt_zero",
        "bootstrap_replicates",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fields,
    )

    writer.writeheader()

    for r in boot_records:
        rr = dict(r)

        rr["ci"] = (
            json.dumps(
                rr["ci"]
            )
        )

        writer.writerow(
            rr
        )


# ==============================================================================
# Coverage
# ==============================================================================
coverage = {}

for name in (
    "V_FV",
    "V_HJTM",
    "H_FV",
    "H_HJTM",
):
    rr = list(
        factorial_map[
            name
        ].values()
    )

    rewritten = sum(
        not r["abstained"]
        for r in rr
    )

    coverage[name] = {
        "rewritten":
            rewritten,

        "abstained":
            333
            -
            rewritten,

        "coverage":
            rewritten
            / 333.0,

        "mean_common_t0_nonabstained":
            mean([
                r["common_t0"]
                for r in rr
                if not r[
                    "abstained"
                ]
            ]),
    }


# ==============================================================================
# Audit
# ==============================================================================
audit = {
    "status":
        "COMPLETE",

    "raw_sha256":
        raw_sha,

    "scored_sha256":
        scored_sha,

    "corrected_candidate_sha256":
        sha256_file(
            RESCORED
        ),

    "frozen_selected_sha256":
        sha256_file(
            SELECTED
        ),

    "selected_common_sha256":
        common_selected_sha,

    "candidate_common_n0_sha256":
        sha256_file(
            COMMON_CANDIDATES
        ),

    "candidate_common_n0_rows":
        len(
            common_rows
        ),

    "source_instances":
        len(
            instances
        ),

    "base_ids":
        len(
            base_ids
        ),

    "candidate_pair_rows":
        len(
            pair_rows
        ),

    "factorial_rows":
        len(
            four_rows
        ),

    "seed_mismatches":
        seed_mismatch,

    "V_FV_frozen_vanilla_mismatches":
        len(
            v_mismatch
        ),

    "H_HJTM_frozen_n0_mismatches":
        len(
            h_mismatch
        ),
}


AUDIT_JSON.write_text(
    json.dumps(
        audit,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)


summary = {
    "status":
        "COMPLETE",

    "metric":
        "common HJTM n0 typicality",

    "interpretation":
        "negative delta means first named condition is more human-typical",

    "raw_candidate_decoder":
        raw_summary,

    "coverage":
        coverage,

    "factorial_contrasts":
        contrast_summary,

    "cluster_bootstrap":
        boot_records,
}


SUMMARY_JSON.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)


# ==============================================================================
# Figure
# ==============================================================================
try:
    import matplotlib.pyplot as plt

    complete = [
        iid
        for iid in instances
        if all(
            not factorial_map[
                c
            ][iid][
                "abstained"
            ]
            for c in (
                "V_FV",
                "V_HJTM",
                "H_FV",
                "H_HJTM",
            )
        )
    ]

    labels = [
        "V_FV",
        "V_HJTM",
        "H_FV",
        "H_HJTM",
    ]

    means = [
        statistics.fmean(
            factorial_map[
                c
            ][iid][
                "common_t0"
            ]
            for iid
            in complete
        )
        for c in labels
    ]

    fig, ax = plt.subplots(
        figsize=(
            7.2,
            4.8,
        )
    )

    ax.bar(
        labels,
        means,
    )

    ax.set_ylabel(
        "Common human-process typicality T0 ↓"
    )

    ax.set_title(
        "DEV111 Decoder × Selector Factorial"
    )

    ax.text(
        0.02,
        0.98,
        f"Common complete cases: n={len(complete)}",
        transform=ax.transAxes,
        va="top",
    )

    fig.tight_layout()

    fig.savefig(
        FIG_PDF,
        bbox_inches="tight",
    )

    plt.close(fig)

except Exception as exc:
    print(
        "WARNING: figure failed:",
        repr(exc),
    )


# ==============================================================================
# Final report
# ==============================================================================
print()
print("=" * 100)
print("RAW MATCHED-CANDIDATE DECODER EFFECT")
print("=" * 100)
print(
    json.dumps(
        raw_summary,
        indent=2,
    )
)

print()
print("=" * 100)
print("FOUR-CONDITION COVERAGE")
print("=" * 100)
print(
    json.dumps(
        coverage,
        indent=2,
    )
)

print()
print("=" * 100)
print("FACTORIAL CONTRASTS")
print("=" * 100)
print(
    json.dumps(
        contrast_summary,
        indent=2,
    )
)

print()
print("=" * 100)
print("CLUSTER BOOTSTRAP")
print("=" * 100)

for r in boot_records:
    print(r)


print()
print("=" * 100)
print("INVARIANTS")
print("=" * 100)

print(
    "source instances                 :",
    len(instances),
)

print(
    "base sample IDs                  :",
    len(base_ids),
)

print(
    "all matched slot/attempt pairs   :",
    len(pair_rows),
)

print(
    "factorial rows                   :",
    len(four_rows),
)

print(
    "seed mismatches                  :",
    seed_mismatch,
)

print(
    "V_FV frozen Vanilla mismatches   :",
    len(v_mismatch),
)

print(
    "H_HJTM frozen n0 mismatches      :",
    len(h_mismatch),
)


print()
print("=" * 100)
print("ARTIFACT HASHES")
print("=" * 100)

for p in (
    COMMON_CANDIDATES,
    COMMON_CANDIDATE_SUMMARY,
    PAIR_CSV,
    FOUR_CSV,
    BOOT_CSV,
    AUDIT_JSON,
    SUMMARY_JSON,
):
    print(
        p.relative_to(ROOT),
        sha256_file(p),
    )


print()
print("=" * 100)
print("STEP 1 COMPLETE — STOP HERE")
print("=" * 100)
