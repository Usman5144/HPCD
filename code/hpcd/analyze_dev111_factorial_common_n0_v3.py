#!/usr/bin/env python3
from __future__ import annotations

import os

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(os.environ.get("HPCD_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))

RESCORED = (
    ROOT
    / "results/hpcd/"
      "t2c4_dev111_surface_entity_selection_v2/"
      "rescored_candidates.jsonl"
)

FROZEN_SELECTED = (
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

COMMON_CANDIDATES = (
    ROOT
    / "results/hpcd/factorial_dev111/"
      "candidate_common_n0_scores_v1.jsonl"
)

OUT = (
    ROOT
    / "results/hpcd/factorial_dev111"
)

FIG_DIR = (
    ROOT
    / "figures/hpcd"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


PAIR_CSV = (
    OUT
    / "candidate_pair_analysis_v3.csv"
)

FOUR_CSV = (
    OUT
    / "four_condition_outputs_v3.csv"
)

MIGRATION_CSV = (
    OUT
    / "historical_vs_common_selector_v3.csv"
)

BOOT_CSV = (
    OUT
    / "cluster_bootstrap_v3.csv"
)

SUMMARY_JSON = (
    OUT
    / "factorial_summary_v3.json"
)

AUDIT_JSON = (
    OUT
    / "factorial_audit_v3.json"
)

FIG_PDF = (
    FIG_DIR
    / "decoding_vs_selection_factorial_v3.pdf"
)


EXPECTED_COMMON_SELECTED_SHA = (
    "2510585baacc293959677d85089cd02ba"
    "49d337f4a651ae63de533eb89726c46"
)

EXPECTED_CANDIDATE_COMMON_ROWS = 2279
EXPECTED_INSTANCES = 333
EXPECTED_BASE_IDS = 111
EXPECTED_PAIR_KEYS = 2664


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
                    f"{path}: bad JSON line "
                    f"{ln}: {exc}"
                )

    return rows


def stable_key(r):
    return (
        r["instance_id"],
        int(r["slot"]),
        int(r["attempt"]),
        r["condition"],
    )


def pair_key(r):
    return (
        r["instance_id"],
        int(r["slot"]),
        int(r["attempt"]),
    )


def mean(xs):
    xs = [
        float(x)
        for x in xs
        if x is not None
        and math.isfinite(float(x))
    ]

    if not xs:
        return None

    return float(
        statistics.fmean(xs)
    )


def median(xs):
    xs = [
        float(x)
        for x in xs
        if x is not None
        and math.isfinite(float(x))
    ]

    if not xs:
        return None

    return float(
        statistics.median(xs)
    )


def summarize(values):
    values = [
        float(x)
        for x in values
        if x is not None
        and math.isfinite(float(x))
    ]

    return {
        "n":
            len(values),

        "mean_delta":
            mean(values),

        "median_delta":
            median(values),

        "improved_negative":
            sum(
                x < 0
                for x in values
            ),

        "worsened_positive":
            sum(
                x > 0
                for x in values
            ),

        "ties":
            sum(
                x == 0
                for x in values
            ),
    }


def cluster_bootstrap(
    rows,
    delta_field,
    *,
    seed=2027,
    B=20000,
):
    by_base = defaultdict(list)

    for r in rows:
        d = r.get(
            delta_field
        )

        if d is None:
            continue

        by_base[
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
        in by_base.items()
    }

    bids = sorted(
        cluster_effects
    )

    x = np.asarray(
        [
            cluster_effects[b]
            for b in bids
        ],
        dtype=float,
    )

    if len(x) == 0:
        return {
            "n_clusters": 0,
            "point_estimate": None,
            "ci_low": None,
            "ci_high": None,
            "p_effect_lt_zero": None,
            "B": B,
        }

    rng = np.random.default_rng(
        seed
    )

    draws = np.empty(
        B,
        dtype=float,
    )

    for i in range(B):
        idx = rng.integers(
            0,
            len(x),
            size=len(x),
        )

        draws[i] = (
            x[idx].mean()
        )

    lo, hi = np.quantile(
        draws,
        [0.025, 0.975],
    )

    return {
        "n_clusters":
            len(x),

        "point_estimate":
            float(
                x.mean()
            ),

        "ci_low":
            float(lo),

        "ci_high":
            float(hi),

        "p_effect_lt_zero":
            float(
                np.mean(
                    draws < 0
                )
            ),

        "B":
            B,
    }


# ==============================================================================
# Load frozen artifacts
# ==============================================================================

for p in (
    RESCORED,
    FROZEN_SELECTED,
    COMMON_SELECTED,
    COMMON_CANDIDATES,
):
    if not p.exists():
        raise RuntimeError(
            f"Missing required artifact: {p}"
        )


if (
    sha256_file(COMMON_SELECTED)
    != EXPECTED_COMMON_SELECTED_SHA
):
    raise RuntimeError(
        "Frozen selected-output common "
        "rescore SHA mismatch"
    )


candidates = load_jsonl(
    RESCORED
)

frozen_selected = load_jsonl(
    FROZEN_SELECTED
)

candidate_scores = load_jsonl(
    COMMON_CANDIDATES
)

common_selected_obj = json.loads(
    COMMON_SELECTED.read_text(
        encoding="utf-8"
    )
)

common_selected = (
    common_selected_obj[
        "rescored_selected_outputs"
    ]
)


print("=" * 100)
print("DEV111 DECODER × SELECTOR FACTORIAL — V3")
print("EXACT COMMON-n0 METRIC")
print("=" * 100)

print()
print(
    "corrected candidates:",
    len(candidates),
)

print(
    "frozen selected:",
    len(frozen_selected),
)

print(
    "candidate common-n0 scores:",
    len(candidate_scores),
)

print(
    "selected common-rescore rows:",
    len(common_selected),
)


if len(candidates) != 7992:
    raise RuntimeError(
        "Expected 7992 corrected candidates"
    )

if len(frozen_selected) != 999:
    raise RuntimeError(
        "Expected 999 frozen selections"
    )

if (
    len(candidate_scores)
    != EXPECTED_CANDIDATE_COMMON_ROWS
):
    raise RuntimeError(
        f"Expected "
        f"{EXPECTED_CANDIDATE_COMMON_ROWS} "
        f"candidate common scores, "
        f"found {len(candidate_scores)}"
    )

if len(common_selected) != 999:
    raise RuntimeError(
        "Expected 999 selected common scores"
    )


# ==============================================================================
# Core IDs
# ==============================================================================

instances = sorted({
    r["instance_id"]
    for r in candidates
})

base_ids = sorted({
    r["sample_id"]
    for r in candidates
})


if len(instances) != EXPECTED_INSTANCES:
    raise RuntimeError(
        f"Expected 333 instances, "
        f"found {len(instances)}"
    )

if len(base_ids) != EXPECTED_BASE_IDS:
    raise RuntimeError(
        f"Expected 111 base IDs, "
        f"found {len(base_ids)}"
    )


# ==============================================================================
# Common candidate score map
# ==============================================================================

score_map = {}

for r in candidate_scores:

    k = (
        r["instance_id"],
        int(r["slot"]),
        int(r["attempt"]),
        r["condition"],
    )

    if k in score_map:
        raise RuntimeError(
            f"Duplicate common score key: {k}"
        )

    score_map[k] = float(
        r[
            "common_typicality_n0"
        ]
    )


target_keys = {
    stable_key(r)
    for r in candidates
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
}


if set(score_map) != target_keys:

    missing = (
        target_keys
        - set(score_map)
    )

    extra = (
        set(score_map)
        - target_keys
    )

    raise RuntimeError(
        "Common candidate score-set mismatch. "
        f"missing={len(missing)} "
        f"extra={len(extra)}"
    )


print()
print(
    "common-n0 target coverage:",
    len(score_map),
    "/",
    len(target_keys),
    "PASS",
)


# ==============================================================================
# Candidate lookup
# ==============================================================================

candidate_map = {
    stable_key(r): r
    for r in candidates
}


if len(candidate_map) != 7992:
    raise RuntimeError(
        "Duplicate corrected candidate keys"
    )


template = {}

for r in candidates:
    template.setdefault(
        r["instance_id"],
        r,
    )


# ==============================================================================
# Historical selected-output maps
# ==============================================================================

frozen_map = {
    (
        r["instance_id"],
        r["condition"],
    ):
        r
    for r in frozen_selected
}


common_selected_map = {
    (
        r["instance_id"],
        r["condition"],
    ):
        r
    for r in common_selected
}


if len(frozen_map) != 999:
    raise RuntimeError(
        "Frozen selected map != 999"
    )

if len(common_selected_map) != 999:
    raise RuntimeError(
        "Common selected map != 999"
    )


# ==============================================================================
# Validate candidate-level common scores against already completed
# selected-output common scores.
# ==============================================================================

score_differences = []

for iid in instances:

    for cond in (
        "vanilla",
        "hpcd_n0",
    ):

        fr = frozen_map[
            (
                iid,
                cond,
            )
        ]

        cr = common_selected_map[
            (
                iid,
                cond,
            )
        ]

        if (
            fr["output_text"]
            != cr["output_text"]
        ):
            raise RuntimeError(
                "Frozen/common-selected text mismatch: "
                f"{iid}/{cond}"
            )


        if (
            fr["status"]
            == "REWRITE"
        ):

            k = (
                iid,
                int(
                    fr[
                        "selected_slot"
                    ]
                ),
                int(
                    fr[
                        "selected_attempt"
                    ]
                ),
                cond,
            )

            if k not in score_map:
                raise RuntimeError(
                    f"Frozen selected candidate "
                    f"missing common score: {k}"
                )


            if (
                candidate_map[k]["text"]
                !=
                fr["output_text"]
            ):
                raise RuntimeError(
                    f"Frozen candidate text mismatch: "
                    f"{k}"
                )


            diff = abs(
                score_map[k]
                -
                float(
                    cr[
                        "common_typicality_n0"
                    ]
                )
            )

            score_differences.append(
                diff
            )


print()
print("=" * 100)
print("COMMON-SCORE CROSS-ARTIFACT VALIDATION")
print("=" * 100)

print(
    "n selected rewrites checked:",
    len(score_differences),
)

print(
    "mean abs difference:",
    mean(
        score_differences
    ),
)

print(
    "max abs difference:",
    max(
        score_differences
    ),
)


if max(score_differences) > 1e-3:
    raise RuntimeError(
        "Candidate-level common score does not "
        "agree with frozen selected common score"
    )


# ==============================================================================
# Historical slot semantics:
#
# 1. For every slot, use first quality-valid attempt.
# 2. First-valid selector -> earliest valid slot.
# 3. Common selector -> minimum exact common-n0 T0 across valid slots.
# ==============================================================================

grouped = defaultdict(list)

for r in candidates:

    if r["condition"] not in (
        "vanilla",
        "hpcd_n0",
    ):
        continue

    grouped[
        (
            r["instance_id"],
            r["condition"],
            int(r["slot"]),
        )
    ].append(r)


def valid_slot_candidates(
    iid,
    cond,
):
    vals = []

    for slot_id in range(4):

        attempts = sorted(
            grouped[
                (
                    iid,
                    cond,
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
                f"Expected 2 attempts: "
                f"{iid}/{cond}/{slot_id}"
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

            if (
                stable_key(chosen)
                not in score_map
            ):
                raise RuntimeError(
                    "Missing common score for "
                    f"quality-valid candidate: "
                    f"{stable_key(chosen)}"
                )

            vals.append(
                chosen
            )

    return vals


def select_first_valid(
    iid,
    cond,
):
    vals = valid_slot_candidates(
        iid,
        cond,
    )

    if not vals:
        return None

    return min(
        vals,
        key=lambda r:
            int(
                r["slot"]
            ),
    )


def select_common_n0(
    iid,
    cond,
):
    vals = valid_slot_candidates(
        iid,
        cond,
    )

    if not vals:
        return None

    return min(
        vals,
        key=lambda r: (
            score_map[
                stable_key(r)
            ],
            int(
                r["slot"]
            ),
        ),
    )


def make_row(
    iid,
    condition_name,
    chosen,
):
    t = template[iid]

    if chosen is None:

        return {
            "instance_id":
                iid,

            "base_sample_id":
                t["sample_id"],

            "generator_key":
                t["generator_key"],

            "domain":
                t["domain"],

            "factorial_condition":
                condition_name,

            "abstained":
                True,

            "slot":
                None,

            "attempt":
                None,

            "common_t0":
                None,

            "output_text":
                t["source_text"],
        }


    return {
        "instance_id":
            iid,

        "base_sample_id":
            chosen["sample_id"],

        "generator_key":
            chosen["generator_key"],

        "domain":
            chosen["domain"],

        "factorial_condition":
            condition_name,

        "abstained":
            False,

        "slot":
            int(
                chosen["slot"]
            ),

        "attempt":
            int(
                chosen["attempt"]
            ),

        "common_t0":
            float(
                score_map[
                    stable_key(
                        chosen
                    )
                ]
            ),

        "output_text":
            chosen["text"],
    }


# ==============================================================================
# Symmetric 2x2 factorial
# ==============================================================================

four_rows = []

for iid in instances:

    specs = (
        (
            "V_FV",
            select_first_valid(
                iid,
                "vanilla",
            ),
        ),

        (
            "V_C0",
            select_common_n0(
                iid,
                "vanilla",
            ),
        ),

        (
            "H_FV",
            select_first_valid(
                iid,
                "hpcd_n0",
            ),
        ),

        (
            "H_C0",
            select_common_n0(
                iid,
                "hpcd_n0",
            ),
        ),
    )


    for name, chosen in specs:
        four_rows.append(
            make_row(
                iid,
                name,
                chosen,
            )
        )


if len(four_rows) != 1332:
    raise RuntimeError(
        f"Expected 1332 four-condition rows, "
        f"found {len(four_rows)}"
    )


fmap = {
    name: {}
    for name in (
        "V_FV",
        "V_C0",
        "H_FV",
        "H_C0",
    )
}


for r in four_rows:
    fmap[
        r[
            "factorial_condition"
        ]
    ][
        r["instance_id"]
    ] = r


# ==============================================================================
# Frozen Vanilla reconstruction
# ==============================================================================

v_mismatch = []

for iid in instances:

    fr = frozen_map[
        (
            iid,
            "vanilla",
        )
    ]

    if (
        fmap[
            "V_FV"
        ][iid][
            "output_text"
        ]
        !=
        fr[
            "output_text"
        ]
    ):
        v_mismatch.append(
            iid
        )


if v_mismatch:
    raise RuntimeError(
        "V_FV failed to reproduce frozen "
        f"Vanilla: {v_mismatch[:5]}"
    )


# ==============================================================================
# Historical HPCD vs common-n0 selector migration
# ==============================================================================

migration_rows = []

h_common_mismatch = []


for iid in instances:

    hist = frozen_map[
        (
            iid,
            "hpcd_n0",
        )
    ]

    c0 = fmap[
        "H_C0"
    ][iid]


    changed = (
        c0[
            "output_text"
        ]
        !=
        hist[
            "output_text"
        ]
    )


    if changed:
        h_common_mismatch.append(
            iid
        )


    hist_common = (
        common_selected_map[
            (
                iid,
                "hpcd_n0",
            )
        ][
            "common_typicality_n0"
        ]
    )


    migration_rows.append({
        "instance_id":
            iid,

        "base_sample_id":
            template[
                iid
            ][
                "sample_id"
            ],

        "generator_key":
            template[
                iid
            ][
                "generator_key"
            ],

        "domain":
            template[
                iid
            ][
                "domain"
            ],

        "historical_status":
            hist[
                "status"
            ],

        "historical_slot":
            hist.get(
                "selected_slot"
            ),

        "historical_attempt":
            hist.get(
                "selected_attempt"
            ),

        "historical_common_t0":
            float(
                hist_common
            ),

        "common_selector_abstained":
            c0[
                "abstained"
            ],

        "common_selector_slot":
            c0[
                "slot"
            ],

        "common_selector_attempt":
            c0[
                "attempt"
            ],

        "common_selector_t0":
            c0[
                "common_t0"
            ],

        "selection_changed":
            changed,

        "delta_common_selector_minus_historical":
            (
                float(
                    c0[
                        "common_t0"
                    ]
                )
                -
                float(
                    hist_common
                )
                if (
                    not c0[
                        "abstained"
                    ]
                    and
                    hist[
                        "status"
                    ]
                    == "REWRITE"
                )
                else None
            ),
    })


print()
print("=" * 100)
print("SELECTOR MIGRATION DIAGNOSTIC")
print("=" * 100)

print(
    "H_C0 vs historical n0 "
    "selection changes:",
    len(
        h_common_mismatch
    ),
    "/333",
)

print(
    "change rate:",
    len(
        h_common_mismatch
    )
    / 333.0,
)


# This should reproduce the already-observed diagnostic.
if len(
    h_common_mismatch
) != 22:
    raise RuntimeError(
        "Expected the previously observed "
        "22 selector migrations; found "
        f"{len(h_common_mismatch)}"
    )


# ==============================================================================
# Raw matched slot/attempt decoder effect
# ==============================================================================

vanilla_attempts = {
    pair_key(r): r
    for r in candidates
    if r[
        "condition"
    ]
    == "vanilla"
}

hpcd_attempts = {
    pair_key(r): r
    for r in candidates
    if r[
        "condition"
    ]
    == "hpcd_n0"
}


if len(
    vanilla_attempts
) != EXPECTED_PAIR_KEYS:

    raise RuntimeError(
        "Expected 2664 Vanilla "
        "slot/attempt keys"
    )


if len(
    hpcd_attempts
) != EXPECTED_PAIR_KEYS:

    raise RuntimeError(
        "Expected 2664 HPCD "
        "slot/attempt keys"
    )


if (
    set(
        vanilla_attempts
    )
    !=
    set(
        hpcd_attempts
    )
):
    raise RuntimeError(
        "Matched candidate key sets differ"
    )


pair_rows = []
seed_mismatch = 0


for pk in sorted(
    vanilla_attempts
):

    v = vanilla_attempts[
        pk
    ]

    h = hpcd_attempts[
        pk
    ]


    if (
        int(
            v["seed"]
        )
        !=
        int(
            h["seed"]
        )
    ):
        seed_mismatch += 1


    vv = bool(
        v[
            "quality_valid_v2"
        ]
    )

    hv = bool(
        h[
            "quality_valid_v2"
        ]
    )


    delta = None
    vt0 = None
    ht0 = None


    if vv:
        vt0 = score_map[
            stable_key(v)
        ]


    if hv:
        ht0 = score_map[
            stable_key(h)
        ]


    if (
        vv
        and hv
    ):
        delta = (
            ht0
            -
            vt0
        )


    pair_rows.append({
        "instance_id":
            v[
                "instance_id"
            ],

        "base_sample_id":
            v[
                "sample_id"
            ],

        "generator_key":
            v[
                "generator_key"
            ],

        "domain":
            v[
                "domain"
            ],

        "slot":
            int(
                v["slot"]
            ),

        "attempt":
            int(
                v[
                    "attempt"
                ]
            ),

        "seed":
            int(
                v["seed"]
            ),

        "vanilla_valid":
            vv,

        "hpcd_valid":
            hv,

        "both_valid":
            (
                vv
                and hv
            ),

        "vanilla_common_t0":
            vt0,

        "hpcd_common_t0":
            ht0,

        "delta":
            delta,
    })


if seed_mismatch:
    raise RuntimeError(
        f"Seed mismatches: "
        f"{seed_mismatch}"
    )


validity_transition = Counter()

for r in pair_rows:

    validity_transition[
        (
            r[
                "vanilla_valid"
            ],
            r[
                "hpcd_valid"
            ],
        )
    ] += 1


raw_values = [
    r["delta"]
    for r in pair_rows
    if r[
        "delta"
    ] is not None
]


raw_summary = summarize(
    raw_values
)

raw_summary[
    "all_seed_matched_pairs"
] = len(
    pair_rows
)

raw_summary[
    "validity_transition"
] = {
    "both_valid":
        validity_transition[
            (
                True,
                True,
            )
        ],

    "vanilla_only_valid":
        validity_transition[
            (
                True,
                False,
            )
        ],

    "hpcd_only_valid":
        validity_transition[
            (
                False,
                True,
            )
        ],

    "neither_valid":
        validity_transition[
            (
                False,
                False,
            )
        ],
}


# ==============================================================================
# Pairwise factorial contrasts
# ==============================================================================

def paired_contrast(
    a_name,
    b_name,
):
    values = []
    rows_out = []

    for iid in instances:

        a = fmap[
            a_name
        ][iid]

        b = fmap[
            b_name
        ][iid]


        if (
            a[
                "abstained"
            ]
            or
            b[
                "abstained"
            ]
        ):
            continue


        d = (
            float(
                a[
                    "common_t0"
                ]
            )
            -
            float(
                b[
                    "common_t0"
                ]
            )
        )


        values.append(
            d
        )

        rows_out.append({
            "instance_id":
                iid,

            "base_sample_id":
                a[
                    "base_sample_id"
                ],

            "generator_key":
                a[
                    "generator_key"
                ],

            "domain":
                a[
                    "domain"
                ],

            "delta":
                d,
        })


    return (
        summarize(
            values
        ),
        rows_out,
    )


contrast_specs = {
    "decoder_effect_first_valid":
        (
            "H_FV",
            "V_FV",
        ),

    "selector_effect_under_vanilla":
        (
            "V_C0",
            "V_FV",
        ),

    "selector_effect_under_hpcd":
        (
            "H_C0",
            "H_FV",
        ),

    "decoder_effect_same_common_selector":
        (
            "H_C0",
            "V_C0",
        ),

    "symmetric_full_pipeline":
        (
            "H_C0",
            "V_FV",
        ),
}


contrast_summary = {}
contrast_rows = {}


for name, (
    a,
    b,
) in contrast_specs.items():

    s, rr = paired_contrast(
        a,
        b,
    )

    s[
        "condition_a"
    ] = a

    s[
        "condition_b"
    ] = b

    contrast_summary[
        name
    ] = s

    contrast_rows[
        name
    ] = rr


# ==============================================================================
# Frozen historical pipeline contrast
# ==============================================================================
hist_values = []
hist_rows = []


for iid in instances:

    v = common_selected_map[
        (
            iid,
            "vanilla",
        )
    ]

    h = common_selected_map[
        (
            iid,
            "hpcd_n0",
        )
    ]


    if (
        v[
            "status"
        ]
        != "REWRITE"
        or
        h[
            "status"
        ]
        != "REWRITE"
    ):
        continue


    d = (
        float(
            h[
                "common_typicality_n0"
            ]
        )
        -
        float(
            v[
                "common_typicality_n0"
            ]
        )
    )


    hist_values.append(
        d
    )

    hist_rows.append({
        "instance_id":
            iid,

        "base_sample_id":
            template[
                iid
            ][
                "sample_id"
            ],

        "generator_key":
            template[
                iid
            ][
                "generator_key"
            ],

        "domain":
            template[
                iid
            ][
                "domain"
            ],

        "delta":
            d,
    })


historical_pair_summary = summarize(
    hist_values
)


# ==============================================================================
# Reproduce the already frozen all-three 214 result exactly
# ==============================================================================
matched_all_three = set(
    common_selected_obj[
        "matched_all_three_ids"
    ]
)


if len(
    matched_all_three
) != 214:

    raise RuntimeError(
        "Expected 214 frozen all-three IDs"
    )


replication_values = []

for iid in sorted(
    matched_all_three
):

    v = common_selected_map[
        (
            iid,
            "vanilla",
        )
    ]

    h = common_selected_map[
        (
            iid,
            "hpcd_n0",
        )
    ]


    replication_values.append(
        float(
            h[
                "common_typicality_n0"
            ]
        )
        -
        float(
            v[
                "common_typicality_n0"
            ]
        )
    )


replication = summarize(
    replication_values
)


print()
print("=" * 100)
print("FROZEN RESULT REPRODUCTION")
print("=" * 100)

print(
    json.dumps(
        replication,
        indent=2,
    )
)


if (
    replication[
        "n"
    ] != 214
):

    raise RuntimeError(
        "Frozen result n != 214"
    )


if abs(
    replication[
        "mean_delta"
    ]
    -
    (
        -0.533541
    )
) > 5e-6:

    raise RuntimeError(
        "Frozen -0.533541 result "
        "was not reproduced"
    )


# ==============================================================================
# Cluster bootstrap
# ==============================================================================

bootstrap_rows = []


raw_boot = cluster_bootstrap(
    pair_rows,
    "delta",
)

bootstrap_rows.append({
    "contrast":
        "raw_candidate_decoder",

    **raw_boot,
})


for name, rr in (
    contrast_rows.items()
):

    tmp = [
        {
            **r,
            "d":
                r[
                    "delta"
                ],
        }
        for r in rr
    ]

    b = cluster_bootstrap(
        tmp,
        "d",
    )

    bootstrap_rows.append({
        "contrast":
            name,
        **b,
    })


hist_tmp = [
    {
        **r,
        "d":
            r[
                "delta"
            ],
    }
    for r in hist_rows
]


hist_boot = cluster_bootstrap(
    hist_tmp,
    "d",
)


bootstrap_rows.append({
    "contrast":
        "historical_frozen_pipeline",
    **hist_boot,
})


# ==============================================================================
# Coverage
# ==============================================================================

coverage = {}

for name in (
    "V_FV",
    "V_C0",
    "H_FV",
    "H_C0",
):

    rr = list(
        fmap[
            name
        ].values()
    )

    rewrites = sum(
        not x[
            "abstained"
        ]
        for x in rr
    )

    coverage[
        name
    ] = {
        "rewrites":
            rewrites,

        "abstentions":
            333
            -
            rewrites,

        "coverage":
            rewrites
            / 333.0,

        "mean_common_t0_rewrites":
            mean([
                x[
                    "common_t0"
                ]
                for x in rr
                if not x[
                    "abstained"
                ]
            ]),
    }


# ==============================================================================
# Write artifacts
# ==============================================================================

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


with MIGRATION_CSV.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            migration_rows[0]
        ),
    )

    writer.writeheader()
    writer.writerows(
        migration_rows
    )


with BOOT_CSV.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:

    fields = [
        "contrast",
        "n_clusters",
        "point_estimate",
        "ci_low",
        "ci_high",
        "p_effect_lt_zero",
        "B",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fields,
    )

    writer.writeheader()
    writer.writerows(
        bootstrap_rows
    )


audit = {
    "status":
        "COMPLETE",

    "candidate_common_n0_file":
        str(
            COMMON_CANDIDATES
            .relative_to(
                ROOT
            )
        ),

    "candidate_common_n0_sha256":
        sha256_file(
            COMMON_CANDIDATES
        ),

    "candidate_common_n0_rows":
        len(
            candidate_scores
        ),

    "candidate_score_crosscheck": {
        "n":
            len(
                score_differences
            ),

        "mean_abs_diff":
            mean(
                score_differences
            ),

        "max_abs_diff":
            max(
                score_differences
            ),
    },

    "source_instances":
        len(
            instances
        ),

    "base_ids":
        len(
            base_ids
        ),

    "matched_seed_pairs":
        len(
            pair_rows
        ),

    "seed_mismatches":
        seed_mismatch,

    "factorial_rows":
        len(
            four_rows
        ),

    "V_FV_frozen_vanilla_mismatches":
        len(
            v_mismatch
        ),

    "H_C0_vs_historical_n0_selection_changes":
        len(
            h_common_mismatch
        ),

    "frozen_214_result_reproduced":
        replication,
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
        "exact common HJTM n0 typicality",

    "important_distinction":
        {
            "historical_hpcd_selector":
                "generation-time trajectory_typicality",

            "symmetric_factorial_selector":
                "exact post-hoc common n0 score_tokens metric",

            "selection_migrations":
                len(
                    h_common_mismatch
                ),
        },

    "raw_candidate_decoder":
        raw_summary,

    "coverage":
        coverage,

    "factorial_contrasts":
        contrast_summary,

    "historical_frozen_pipeline_pairwise":
        historical_pair_summary,

    "frozen_214_replication":
        replication,

    "cluster_bootstrap":
        bootstrap_rows,
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
            not fmap[
                c
            ][iid][
                "abstained"
            ]
            for c in (
                "V_FV",
                "V_C0",
                "H_FV",
                "H_C0",
            )
        )
    ]


    labels = [
        "V_FV",
        "V_C0",
        "H_FV",
        "H_C0",
    ]


    means = [
        statistics.fmean(
            fmap[
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
        "Common HJTM n0 typicality T0 ↓"
    )

    ax.set_title(
        "DEV111 Decoder × Selector Factorial"
    )

    ax.text(
        0.02,
        0.98,
        (
            f"Common complete cases: "
            f"n={len(complete)}"
        ),
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
        "WARNING figure:",
        repr(exc),
    )


# ==============================================================================
# Report
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
print("SYMMETRIC FACTORIAL COVERAGE")
print("=" * 100)

print(
    json.dumps(
        coverage,
        indent=2,
    )
)


print()
print("=" * 100)
print("SYMMETRIC FACTORIAL CONTRASTS")
print("=" * 100)

print(
    json.dumps(
        contrast_summary,
        indent=2,
    )
)


print()
print("=" * 100)
print("HISTORICAL FROZEN PIPELINE")
print("=" * 100)

print(
    json.dumps(
        historical_pair_summary,
        indent=2,
    )
)


print()
print("=" * 100)
print("CLUSTER BOOTSTRAP")
print("=" * 100)

for r in bootstrap_rows:
    print(r)


print()
print("=" * 100)
print("FINAL INVARIANTS")
print("=" * 100)

print(
    "source instances                   :",
    len(instances),
)

print(
    "base IDs                           :",
    len(base_ids),
)

print(
    "candidate score rows               :",
    len(candidate_scores),
)

print(
    "matched slot/attempt pairs         :",
    len(pair_rows),
)

print(
    "seed mismatches                    :",
    seed_mismatch,
)

print(
    "factorial rows                     :",
    len(four_rows),
)

print(
    "V_FV frozen Vanilla mismatches     :",
    len(v_mismatch),
)

print(
    "H_C0 vs historical n0 migrations   :",
    len(h_common_mismatch),
)

print(
    "frozen 214-result mean             :",
    replication[
        "mean_delta"
    ],
)


print()
print("=" * 100)
print("ARTIFACT HASHES")
print("=" * 100)

for p in (
    PAIR_CSV,
    FOUR_CSV,
    MIGRATION_CSV,
    BOOT_CSV,
    AUDIT_JSON,
    SUMMARY_JSON,
):
    print(
        p.relative_to(
            ROOT
        ),
        sha256_file(p),
    )


if FIG_PDF.exists():

    print(
        FIG_PDF.relative_to(
            ROOT
        ),
        sha256_file(
            FIG_PDF
        ),
    )


print()
print("=" * 100)
print("STEP 1 COMPLETE — STOP HERE")
print("=" * 100)
