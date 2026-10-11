#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


# =============================================================================
# GLOBAL FROZEN ANALYSIS CONTRACT
# =============================================================================

ROOT = Path(".").resolve()

B = 50_000
SEED = 2027

DATASETS = [
    ("Overall", None),
    ("News", "news"),
    ("Reddit", "reddit"),
    ("Reviews", "reviews"),
    ("Wiki", "wiki"),
]

GENERATORS = [
    "Qwen2.5-3B-Instruct",
    "DeepSeek-R1-Distill-Qwen-7B",
]

METHOD_ORDER = [
    "Human",
    "Original",
    "Vanilla",
    "Temperature Control",
    "CoPA-3B",
    "HPCD",
]

COND_DISPLAY = {
    "human": "Human",
    "original": "Original",
    "vanilla_t10": "Vanilla",
    "vanilla_t11": "Temperature Control",
    "copa3b": "CoPA-3B",
    "copa_3b": "CoPA-3B",
    "hpcd_n0": "HPCD",
}

DISPLAY_COND = {
    "Human": "human",
    "Original": "original",
    "Vanilla": "vanilla_t10",
    "Temperature Control": "vanilla_t11",
    "CoPA-3B": "copa_3b",
    "HPCD": "hpcd_n0",
}


# =============================================================================
# FROZEN INPUTS
# =============================================================================

T0_FILE = Path(
    "results/hpcd/final_copa_t0_eval_v1/"
    "primary_scores_all_methods.jsonl"
)

QUALITY_FILE = Path(
    "results/quality/fresh95_common_quality_v1/"
    "per_instance.jsonl"
)

NLI_FILE = Path(
    "results/quality/final_semantic_v1/nli/"
    "nli_scores_v1.jsonl"
)

ENTITY_FILE = Path(
    "results/quality/final_semantic_v1/entity/"
    "entity_scores_v1.jsonl"
)

FAST_FILE = Path(
    "results/detectors/fresh95_final_v1/"
    "fast_detectgpt/scores_v1.jsonl"
)

RADAR_FILE = Path(
    "results/detectors/fresh95_final_v1/"
    "radar/scores_v1.jsonl"
)

SURPMARK_FILE = Path(
    "results/detectors/fresh95_final_v1/"
    "surpmark/final_v1/scores_v1.jsonl"
)

SIMPLE_FILE = Path(
    "results/detectors/fresh95_final_v1/"
    "simple_scores/document_scores_v1.jsonl"
)

THRESHOLD_FILE = Path(
    "results/detectors/fresh95_final_v1/"
    "common_analysis_v1/calibration_thresholds_v1.csv"
)

ASR_SOURCE = Path(
    "results/final_analysis_v1/asr/"
    "detector_asr_by_dataset_v1.csv"
)

ASR_PAIR_SOURCE = Path(
    "results/final_analysis_v1/asr/"
    "detector_asr_pairwise_by_dataset_v1.csv"
)


EXPECTED_SHA = {
    T0_FILE:
        "f0ce714c7c2d616125e48bf7e7a1598e3ca0176a46fe393626550c6f696e0f20",

    QUALITY_FILE:
        "935ad844fa0c189080b947b8d1d8959c685b30911aa7b6e34e71a4a8d7057513",

    NLI_FILE:
        "d0e4ab27af8f83c42b318fb83020ee2ccb2e4ed57fce3138591ad5be849c0e88",

    ENTITY_FILE:
        "ab06655c4dd5679eb36c7d16f4f915514ba530f2a4d69208afded72bc548c8eb",

    FAST_FILE:
        "0088c81388c3c9729f7b9c8a77957ec13e5e6b3ba3c3883eb6fcd796045699e8",

    RADAR_FILE:
        "6df9b68bfc196f9dae527b1a815eae2b81a820816ab937c1799b5a2b01fda6b5",

    SURPMARK_FILE:
        "72d0c5e7548e48b9991d8eb772014f103e008199ee819c308fb39afb3468b90a",

    SIMPLE_FILE:
        "8c5bd0b9b92fb69741eb9971adb63fd191f5ddcaf981aa044c9777a4942e5502",

    THRESHOLD_FILE:
        "de6fe70e9f0195f01674fd4b03209ae6721a7b2d0c5816f2767fd984c566be13",
}


# =============================================================================
# OUTPUTS
# =============================================================================

QUALITY_OUT = Path(
    "results/final_analysis_v1/quality/"
    "quality_by_dataset_v1.csv"
)

QUALITY_PAIR_OUT = Path(
    "results/final_analysis_v1/quality/"
    "quality_pairwise_by_dataset_v1.csv"
)

QUALITY_GEN_OUT = Path(
    "results/final_analysis_v1/quality/"
    "quality_by_generator_v1.csv"
)

QUALITY_PAIR_GEN_OUT = Path(
    "results/final_analysis_v1/quality/"
    "quality_pairwise_by_generator_v1.csv"
)

QUALITY_DG_OUT = Path(
    "results/final_analysis_v1/quality/"
    "quality_by_dataset_generator_v1.csv"
)


T0_OUT = Path(
    "results/final_analysis_v1/t0/"
    "t0_by_dataset_v1.csv"
)

T0_PAIR_OUT = Path(
    "results/final_analysis_v1/t0/"
    "t0_pairwise_by_dataset_v1.csv"
)

T0_GEN_OUT = Path(
    "results/final_analysis_v1/t0/"
    "t0_by_generator_v1.csv"
)

T0_PAIR_GEN_OUT = Path(
    "results/final_analysis_v1/t0/"
    "t0_pairwise_by_generator_v1.csv"
)

T0_DG_OUT = Path(
    "results/final_analysis_v1/t0/"
    "t0_by_dataset_generator_v1.csv"
)


DET_METRICS_OUT = Path(
    "results/final_analysis_v1/detectors/"
    "detector_metrics_by_dataset_v1.csv"
)

DET_PAIR_OUT = Path(
    "results/final_analysis_v1/detectors/"
    "detector_pairwise_by_dataset_v1.csv"
)

DET_GEN_OUT = Path(
    "results/final_analysis_v1/detectors/"
    "detector_metrics_by_generator_v1.csv"
)

DET_PAIR_GEN_OUT = Path(
    "results/final_analysis_v1/detectors/"
    "detector_pairwise_by_generator_v1.csv"
)

DET_DG_OUT = Path(
    "results/final_analysis_v1/detectors/"
    "detector_metrics_by_dataset_generator_v1.csv"
)

DET_ASR_OUT = Path(
    "results/final_analysis_v1/detectors/"
    "detector_asr_by_dataset_v1.csv"
)

DET_ASR_PAIR_OUT = Path(
    "results/final_analysis_v1/detectors/"
    "detector_asr_pairwise_by_dataset_v1.csv"
)


SIMPLE_OUT = Path(
    "results/final_analysis_v1/simple_scores/"
    "simple_scores_by_dataset_v1.csv"
)

SIMPLE_GEN_OUT = Path(
    "results/final_analysis_v1/simple_scores/"
    "simple_scores_by_generator_v1.csv"
)

SIMPLE_DG_OUT = Path(
    "results/final_analysis_v1/simple_scores/"
    "simple_scores_by_dataset_generator_v1.csv"
)


MAIN_DOMAIN_TABLE = Path(
    "results/final_analysis_v1/tables/"
    "domain_generalization_main_v1.csv"
)

DETECTOR_DOMAIN_TABLE = Path(
    "results/final_analysis_v1/tables/"
    "detector_tpr5_by_dataset_v1.csv"
)

MANIFEST_OUT = Path(
    "results/final_analysis_v1/"
    "domain_generator_aggregation_manifest_v1.json"
)


# =============================================================================
# HELPERS
# =============================================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def load_jsonl(path: Path):
    with path.open(
        encoding="utf-8",
    ) as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def write_csv(path: Path, rows):
    rows = list(rows)

    if not rows:
        raise RuntimeError(
            f"No rows for {path}"
        )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )
        w.writeheader()
        w.writerows(rows)


def finite_or_none(x):
    if x is None:
        return None

    try:
        x = float(x)
    except Exception:
        return None

    if not math.isfinite(x):
        return None

    return x


def quantile_ci(x):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    x = x[
        np.isfinite(x)
    ]

    if len(x) == 0:
        return (
            float("nan"),
            float("nan"),
        )

    lo, hi = np.quantile(
        x,
        [0.025, 0.975],
    )

    return (
        float(lo),
        float(hi),
    )


def base_key(row):
    return row["base_id"]


def pair_key(row):
    for key in [
        "instance_id",
        "source_sample_id",
    ]:
        value = row.get(key)

        if value:
            return str(value)

    return (
        f"{row['base_id']}::"
        f"{row.get('source_generator')}"
    )


def bootstrap_mean_base(
    base_values: dict,
    base_domains: dict | None = None,
    stratified: bool = False,
):
    """
    base_values:
      base_id -> scalar

    For T0 pooled comparisons only, use domain-stratified
    cluster bootstrap to match the frozen primary protocol.
    """

    if not base_values:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
        )

    rng = np.random.default_rng(
        SEED
    )

    ids = sorted(
        base_values
    )

    point = float(
        np.mean(
            [
                base_values[i]
                for i in ids
            ]
        )
    )

    if (
        stratified
        and
        base_domains is not None
    ):
        domains = sorted(
            {
                base_domains[i]
                for i in ids
            }
        )

        pieces = []

        weights = []

        for domain in domains:

            dids = [
                i
                for i in ids
                if base_domains[i]
                == domain
            ]

            vals = np.asarray(
                [
                    base_values[i]
                    for i in dids
                ],
                dtype=np.float64,
            )

            draws = rng.integers(
                0,
                len(vals),
                size=(
                    B,
                    len(vals),
                ),
            )

            means = vals[
                draws
            ].mean(
                axis=1
            )

            pieces.append(
                means
            )

            weights.append(
                len(vals)
            )

        weights = np.asarray(
            weights,
            dtype=np.float64,
        )

        weights = (
            weights
            /
            weights.sum()
        )

        boot = np.zeros(
            B,
            dtype=np.float64,
        )

        for w, p in zip(
            weights,
            pieces,
        ):
            boot += w * p

    else:
        vals = np.asarray(
            [
                base_values[i]
                for i in ids
            ],
            dtype=np.float64,
        )

        draws = rng.integers(
            0,
            len(vals),
            size=(
                B,
                len(vals),
            ),
        )

        boot = vals[
            draws
        ].mean(
            axis=1
        )

    lo, hi = quantile_ci(
        boot
    )

    return (
        point,
        lo,
        hi,
    )


def absolute_cluster_summary(
    records,
    value_field="value",
):
    usable = []

    for r in records:
        v = finite_or_none(
            r.get(
                value_field
            )
        )

        if v is not None:
            usable.append(
                (
                    r,
                    v,
                )
            )

    if not usable:
        return {
            "n_base_ids": 0,
            "n_instances": len(records),
            "n_metric_instances": 0,
            "mean": "",
            "median": "",
            "std": "",
            "cluster_mean": "",
            "cluster_ci_low": "",
            "cluster_ci_high": "",
        }

    values = np.asarray(
        [
            v
            for _, v in usable
        ],
        dtype=np.float64,
    )

    by_base = defaultdict(
        list
    )

    for r, v in usable:
        by_base[
            r["base_id"]
        ].append(
            v
        )

    base_values = {
        b:
            float(
                np.mean(vals)
            )
        for b, vals
        in by_base.items()
    }

    point, lo, hi = (
        bootstrap_mean_base(
            base_values
        )
    )

    return {
        "n_base_ids":
            len(base_values),

        "n_instances":
            len(records),

        "n_metric_instances":
            len(usable),

        "mean":
            float(
                values.mean()
            ),

        "median":
            float(
                np.median(
                    values
                )
            ),

        "std":
            float(
                values.std(
                    ddof=1
                )
            )
            if len(values) > 1
            else 0.0,

        "cluster_mean":
            point,

        "cluster_ci_low":
            lo,

        "cluster_ci_high":
            hi,
    }


def paired_cluster_summary(
    lhs,
    rhs,
    *,
    stratified=False,
):
    lmap = {
        r["pair_key"]: r
        for r in lhs
        if finite_or_none(
            r.get("value")
        ) is not None
    }

    rmap = {
        r["pair_key"]: r
        for r in rhs
        if finite_or_none(
            r.get("value")
        ) is not None
    }

    common = sorted(
        set(lmap)
        &
        set(rmap)
    )

    if not common:
        return None

    by_base = defaultdict(
        list
    )

    base_domains = {}

    instance_diffs = []

    for key in common:

        a = lmap[key]
        b = rmap[key]

        if (
            a["base_id"]
            !=
            b["base_id"]
        ):
            raise RuntimeError(
                "Paired base mismatch"
            )

        diff = (
            float(
                a["value"]
            )
            -
            float(
                b["value"]
            )
        )

        instance_diffs.append(
            diff
        )

        by_base[
            a["base_id"]
        ].append(
            diff
        )

        base_domains[
            a["base_id"]
        ] = a["domain"]

    base_values = {
        b:
            float(
                np.mean(vals)
            )
        for b, vals
        in by_base.items()
    }

    point, lo, hi = (
        bootstrap_mean_base(
            base_values,
            base_domains,
            stratified=stratified,
        )
    )

    rng = np.random.default_rng(
        SEED
    )

    # Probability sign from same cluster bootstrap form.
    ids = sorted(
        base_values
    )

    vals = np.asarray(
        [
            base_values[i]
            for i in ids
        ],
        dtype=np.float64,
    )

    if (
        stratified
        and
        len(set(base_domains.values()))
        > 1
    ):
        domains = sorted(
            set(
                base_domains.values()
            )
        )

        boot = np.zeros(
            B,
            dtype=np.float64,
        )

        total_n = len(ids)

        for d in domains:
            dids = [
                i
                for i in ids
                if base_domains[i] == d
            ]

            dvals = np.asarray(
                [
                    base_values[i]
                    for i in dids
                ]
            )

            draws = rng.integers(
                0,
                len(dvals),
                size=(
                    B,
                    len(dvals),
                ),
            )

            boot += (
                len(dids)
                /
                total_n
            ) * dvals[
                draws
            ].mean(
                axis=1
            )
    else:
        draws = rng.integers(
            0,
            len(vals),
            size=(
                B,
                len(vals),
            ),
        )

        boot = vals[
            draws
        ].mean(
            axis=1
        )

    return {
        "n_instances":
            len(common),

        "n_base_ids":
            len(base_values),

        "instance_mean_delta":
            float(
                np.mean(
                    instance_diffs
                )
            ),

        "cluster_mean_delta":
            point,

        "cluster_ci_low":
            lo,

        "cluster_ci_high":
            hi,

        "bootstrap_prob_delta_lt_0":
            float(
                np.mean(
                    boot < 0
                )
            ),

        "bootstrap_prob_delta_gt_0":
            float(
                np.mean(
                    boot > 0
                )
            ),
    }


def group_filter(
    rows,
    domain=None,
    generator=None,
):
    out = rows

    if domain is not None:
        out = [
            r
            for r in out
            if r["domain"]
            == domain
        ]

    if generator is not None:
        out = [
            r
            for r in out
            if r.get(
                "source_generator"
            )
            == generator
        ]

    return out


def auc_point(
    human_scores,
    machine_scores,
):
    h = np.asarray(
        human_scores,
        dtype=np.float64,
    )

    m = np.asarray(
        machine_scores,
        dtype=np.float64,
    )

    comp = (
        m[:, None]
        >
        h[None, :]
    ).astype(
        np.float64
    )

    ties = (
        m[:, None]
        ==
        h[None, :]
    )

    comp += (
        0.5
        *
        ties
    )

    return float(
        comp.mean()
    )


def cluster_auc(
    human_rows,
    machine_rows,
):
    """
    Joint base-ID cluster bootstrap:
    resampled base carries human plus all available source-generator
    machine variants for the condition.
    """

    human_by_base = {}

    for r in human_rows:
        human_by_base[
            r["base_id"]
        ] = float(
            r["analysis_score"]
        )

    machine_by_base = defaultdict(
        list
    )

    for r in machine_rows:
        machine_by_base[
            r["base_id"]
        ].append(
            float(
                r["analysis_score"]
            )
        )

    bases = sorted(
        set(
            human_by_base
        )
        &
        set(
            machine_by_base
        )
    )

    if not bases:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            0,
        )

    human_scores = [
        human_by_base[b]
        for b in bases
    ]

    machine_scores = [
        s
        for b in bases
        for s in machine_by_base[b]
    ]

    point = auc_point(
        human_scores,
        machine_scores,
    )

    n = len(
        bases
    )

    # K[i,j] =
    # sum of human_i vs all machine variants in base_j.
    K = np.zeros(
        (
            n,
            n,
        ),
        dtype=np.float64,
    )

    nvar = np.zeros(
        n,
        dtype=np.float64,
    )

    for i, bh in enumerate(
        bases
    ):
        hs = human_by_base[
            bh
        ]

        for j, bm in enumerate(
            bases
        ):
            ms = np.asarray(
                machine_by_base[
                    bm
                ],
                dtype=np.float64,
            )

            nvar[j] = len(
                ms
            )

            K[i, j] = float(
                np.sum(
                    ms > hs
                )
                +
                0.5
                *
                np.sum(
                    ms == hs
                )
            )

    rng = np.random.default_rng(
        SEED
    )

    W = rng.multinomial(
        n,
        np.repeat(
            1.0 / n,
            n,
        ),
        size=B,
    ).astype(
        np.float64
    )

    numerator = np.einsum(
        "bi,ij,bj->b",
        W,
        K,
        W,
        optimize=True,
    )

    denominator = (
        W.sum(
            axis=1
        )
        *
        (
            W
            @
            nvar
        )
    )

    boot = numerator / denominator

    lo, hi = quantile_ci(
        boot
    )

    return (
        point,
        lo,
        hi,
        n,
    )


def cluster_tpr(
    machine_rows,
    threshold,
):
    by_base_total = defaultdict(
        int
    )

    by_base_detected = defaultdict(
        int
    )

    for r in machine_rows:
        b = r["base_id"]

        by_base_total[b] += 1

        if (
            float(
                r["analysis_score"]
            )
            >
            threshold
        ):
            by_base_detected[
                b
            ] += 1

    bases = sorted(
        by_base_total
    )

    total = np.asarray(
        [
            by_base_total[b]
            for b in bases
        ],
        dtype=np.float64,
    )

    detected = np.asarray(
        [
            by_base_detected[b]
            for b in bases
        ],
        dtype=np.float64,
    )

    point = float(
        detected.sum()
        /
        total.sum()
    )

    rng = np.random.default_rng(
        SEED
    )

    W = rng.multinomial(
        len(bases),
        np.repeat(
            1.0
            /
            len(bases),
            len(bases),
        ),
        size=B,
    ).astype(
        np.float64
    )

    boot = (
        W
        @
        detected
    ) / (
        W
        @
        total
    )

    lo, hi = quantile_ci(
        boot
    )

    return (
        point,
        lo,
        hi,
    )


# =============================================================================
# VERIFY INPUT HASHES
# =============================================================================

print("=" * 100)
print("FINAL DOMAIN / GENERATOR AGGREGATION")
print("=" * 100)

for path, expected in EXPECTED_SHA.items():

    if not path.exists():
        raise RuntimeError(
            f"Missing frozen input: {path}"
        )

    observed = sha256_file(
        path
    )

    print(
        f"{path}: {observed}"
    )

    if observed != expected:
        raise RuntimeError(
            f"SHA mismatch: {path}"
        )

print("ALL_FROZEN_INPUT_SHA=PASS")


# =============================================================================
# LOAD DATA
# =============================================================================

t0_raw = load_jsonl(
    T0_FILE
)

quality_raw = load_jsonl(
    QUALITY_FILE
)

nli_raw = load_jsonl(
    NLI_FILE
)

entity_raw = load_jsonl(
    ENTITY_FILE
)

fast_raw = load_jsonl(
    FAST_FILE
)

radar_raw = load_jsonl(
    RADAR_FILE
)

surp_raw = load_jsonl(
    SURPMARK_FILE
)

simple_raw = load_jsonl(
    SIMPLE_FILE
)

if len(t0_raw) != 475:
    raise RuntimeError(
        "T0 row count != 475"
    )

if len(quality_raw) != 475:
    raise RuntimeError(
        "Quality row count != 475"
    )

if len(nli_raw) != 380:
    raise RuntimeError(
        "NLI row count != 380"
    )

if len(entity_raw) != 475:
    raise RuntimeError(
        "Entity row count != 475"
    )

for rows, name in [
    (fast_raw, "Fast"),
    (radar_raw, "RADAR"),
    (surp_raw, "SurpMark"),
    (simple_raw, "Simple"),
]:
    if len(rows) != 523:
        raise RuntimeError(
            f"{name} row count != 523"
        )

print("ROW_COUNTS=PASS")


# =============================================================================
# QUALITY — BUILD ONE TIDY METRIC UNIVERSE
# =============================================================================

quality_metric_records = []


def add_quality_record(
    row,
    condition,
    metric,
    value,
    source,
):
    quality_metric_records.append({
        "base_id":
            row["base_id"],

        "domain":
            row["domain"],

        "source_generator":
            row.get(
                "source_generator"
            ),

        "condition":
            condition,

        "method":
            COND_DISPLAY[
                condition
            ],

        "pair_key":
            pair_key(
                row
            ),

        "metric":
            metric,

        "value":
            value,

        "source":
            source,
    })


# Common quality
for r in quality_raw:

    cond = r["condition"]

    common_metrics = {
        "MPNet":
            r["mpnet_cosine"],

        "edit_fraction":
            r["token_edit_fraction"],

        "length_ratio":
            r["word_length_ratio"],

        "absolute_length_deviation":
            abs(
                float(
                    r["word_length_ratio"]
                )
                -
                1.0
            ),
    }

    source_numbers = (
        r.get(
            "source_number_strings"
        )
        or
        []
    )

    source_dates = (
        r.get(
            "source_date_strings"
        )
        or
        []
    )

    if source_numbers:
        common_metrics[
            "number_preservation"
        ] = (
            1.0
            if r[
                "all_numbers_preserved"
            ]
            else 0.0
        )

    if source_dates:
        common_metrics[
            "date_preservation"
        ] = (
            1.0
            if r[
                "all_dates_preserved"
            ]
            else 0.0
        )

    if (
        source_numbers
        or
        source_dates
    ):
        common_metrics[
            "number_date_preservation"
        ] = (
            1.0
            if r[
                "all_number_date_strings_preserved"
            ]
            else 0.0
        )

    for metric, value in (
        common_metrics.items()
    ):
        add_quality_record(
            r,
            cond,
            metric,
            value,
            "frozen_common_quality",
        )


# NLI — Original intentionally absent.
# No model inference is fabricated for self-vs-self.
for r in nli_raw:

    cond = r["condition"]

    for metric, field in [
        (
            "NLI-forward",
            "forward_entailment",
        ),
        (
            "NLI-reverse",
            "reverse_entailment",
        ),
        (
            "NLI-min",
            "nli_min",
        ),
        (
            "NLI-mean",
            "nli_mean",
        ),
    ]:
        add_quality_record(
            r,
            cond,
            metric,
            r[field],
            "frozen_bidirectional_nli",
        )


# Entity
for r in entity_raw:

    cond = r["condition"]

    if (
        r[
            "entity_preservation_rate"
        ]
        is not None
    ):
        add_quality_record(
            r,
            cond,
            "entity_preservation",
            r[
                "entity_preservation_rate"
            ],
            "frozen_source_only_entity",
        )

    add_quality_record(
        r,
        cond,
        "all_entities_preserved_rate",
        (
            1.0
            if r[
                "all_entities_preserved"
            ]
            else 0.0
        ),
        "frozen_source_only_entity",
    )


# =============================================================================
# QUALITY BY DATASET
# =============================================================================

quality_rows = []

metrics = sorted(
    {
        r["metric"]
        for r in quality_metric_records
    }
)

for dataset_label, domain in DATASETS:

    scoped = group_filter(
        quality_metric_records,
        domain=domain,
    )

    for method in [
        "Original",
        "Vanilla",
        "Temperature Control",
        "CoPA-3B",
        "HPCD",
    ]:

        for metric in metrics:

            subset = [
                r
                for r in scoped
                if (
                    r["method"]
                    == method
                    and
                    r["metric"]
                    == metric
                )
            ]

            if not subset:
                continue

            stats = absolute_cluster_summary(
                subset
            )

            quality_rows.append({
                "dataset":
                    dataset_label,

                "method":
                    method,

                "n_base_ids":
                    stats[
                        "n_base_ids"
                    ],

                "n_instances":
                    stats[
                        "n_instances"
                    ],

                "n_metric_instances":
                    stats[
                        "n_metric_instances"
                    ],

                "metric":
                    metric,

                "mean":
                    stats[
                        "mean"
                    ],

                "median":
                    stats[
                        "median"
                    ],

                "std":
                    stats[
                        "std"
                    ],

                "cluster_mean":
                    stats[
                        "cluster_mean"
                    ],

                "cluster_ci_low":
                    stats[
                        "cluster_ci_low"
                    ],

                "cluster_ci_high":
                    stats[
                        "cluster_ci_high"
                    ],
            })


write_csv(
    QUALITY_OUT,
    quality_rows,
)


# =============================================================================
# QUALITY PAIRWISE BY DATASET
# =============================================================================

quality_pair_rows = []

QUALITY_PAIRS = [
    (
        "HPCD - Vanilla",
        "HPCD",
        "Vanilla",
    ),
    (
        "HPCD - CoPA-3B",
        "HPCD",
        "CoPA-3B",
    ),
    (
        "HPCD - Temperature Control",
        "HPCD",
        "Temperature Control",
    ),
]

for dataset_label, domain in DATASETS:

    scoped = group_filter(
        quality_metric_records,
        domain=domain,
    )

    for metric in metrics:

        metric_rows = [
            r
            for r in scoped
            if r["metric"]
            == metric
        ]

        for (
            comparison,
            lhs_method,
            rhs_method,
        ) in QUALITY_PAIRS:

            lhs = [
                r
                for r in metric_rows
                if r["method"]
                == lhs_method
            ]

            rhs = [
                r
                for r in metric_rows
                if r["method"]
                == rhs_method
            ]

            result = paired_cluster_summary(
                lhs,
                rhs,
            )

            if result is None:
                continue

            quality_pair_rows.append({
                "dataset":
                    dataset_label,

                "comparison":
                    comparison,

                "metric":
                    metric,

                **result,
            })


write_csv(
    QUALITY_PAIR_OUT,
    quality_pair_rows,
)


# =============================================================================
# QUALITY BY GENERATOR + DATASET×GENERATOR
# =============================================================================

quality_gen_rows = []
quality_pair_gen_rows = []
quality_dg_rows = []

for generator in GENERATORS:

    scoped = group_filter(
        quality_metric_records,
        generator=generator,
    )

    for method in [
        "Original",
        "Vanilla",
        "Temperature Control",
        "CoPA-3B",
        "HPCD",
    ]:
        for metric in metrics:

            subset = [
                r
                for r in scoped
                if (
                    r["method"]
                    == method
                    and
                    r["metric"]
                    == metric
                )
            ]

            if not subset:
                continue

            s = absolute_cluster_summary(
                subset
            )

            quality_gen_rows.append({
                "source_generator":
                    generator,

                "method":
                    method,

                "metric":
                    metric,

                **s,
            })

    for metric in metrics:

        metric_rows = [
            r
            for r in scoped
            if r["metric"]
            == metric
        ]

        for (
            comparison,
            lhs_method,
            rhs_method,
        ) in QUALITY_PAIRS:

            result = paired_cluster_summary(
                [
                    r
                    for r in metric_rows
                    if r["method"]
                    == lhs_method
                ],
                [
                    r
                    for r in metric_rows
                    if r["method"]
                    == rhs_method
                ],
            )

            if result is None:
                continue

            quality_pair_gen_rows.append({
                "source_generator":
                    generator,

                "comparison":
                    comparison,

                "metric":
                    metric,

                **result,
            })


for dataset_label, domain in DATASETS[1:]:

    for generator in GENERATORS:

        scoped = group_filter(
            quality_metric_records,
            domain=domain,
            generator=generator,
        )

        for method in [
            "Original",
            "Vanilla",
            "Temperature Control",
            "CoPA-3B",
            "HPCD",
        ]:

            for metric in metrics:

                subset = [
                    r
                    for r in scoped
                    if (
                        r["method"]
                        == method
                        and
                        r["metric"]
                        == metric
                    )
                ]

                if not subset:
                    continue

                s = absolute_cluster_summary(
                    subset
                )

                quality_dg_rows.append({
                    "dataset":
                        dataset_label,

                    "source_generator":
                        generator,

                    "method":
                        method,

                    "metric":
                        metric,

                    **s,
                })


write_csv(
    QUALITY_GEN_OUT,
    quality_gen_rows,
)

write_csv(
    QUALITY_PAIR_GEN_OUT,
    quality_pair_gen_rows,
)

write_csv(
    QUALITY_DG_OUT,
    quality_dg_rows,
)


# =============================================================================
# T0 TIDY RECORDS
# =============================================================================

t0_records = []

for r in t0_raw:

    method = COND_DISPLAY[
        r["method"]
    ]

    t0_records.append({
        "base_id":
            r["base_id"],

        "domain":
            r["domain"],

        "source_generator":
            r["source_generator"],

        "condition":
            r["method"],

        "method":
            method,

        "pair_key":
            r["instance_id"],

        "value":
            float(
                r["T0"]
            ),
    })


# =============================================================================
# T0 BY DATASET
# =============================================================================

t0_rows = []

for dataset_label, domain in DATASETS:

    scoped = group_filter(
        t0_records,
        domain=domain,
    )

    for method in [
        "Original",
        "Vanilla",
        "Temperature Control",
        "CoPA-3B",
        "HPCD",
    ]:

        subset = [
            r
            for r in scoped
            if r["method"]
            == method
        ]

        s = absolute_cluster_summary(
            subset
        )

        t0_rows.append({
            "dataset":
                dataset_label,

            "method":
                method,

            "n_base_ids":
                s[
                    "n_base_ids"
                ],

            "n_instances":
                s[
                    "n_instances"
                ],

            "mean":
                s[
                    "mean"
                ],

            "median":
                s[
                    "median"
                ],

            "std":
                s[
                    "std"
                ],

            "cluster_mean":
                s[
                    "cluster_mean"
                ],

            "cluster_ci_low":
                s[
                    "cluster_ci_low"
                ],

            "cluster_ci_high":
                s[
                    "cluster_ci_high"
                ],

            "orientation":
                "lower = more human-process typical",
        })


write_csv(
    T0_OUT,
    t0_rows,
)


# =============================================================================
# T0 PAIRWISE BY DATASET
# =============================================================================

T0_PAIRS = [
    (
        "HPCD - Vanilla",
        "HPCD",
        "Vanilla",
    ),
    (
        "HPCD - Temperature Control",
        "HPCD",
        "Temperature Control",
    ),
    (
        "HPCD - CoPA-3B",
        "HPCD",
        "CoPA-3B",
    ),
    (
        "CoPA-3B - Vanilla",
        "CoPA-3B",
        "Vanilla",
    ),
]

t0_pair_rows = []

for dataset_label, domain in DATASETS:

    scoped = group_filter(
        t0_records,
        domain=domain,
    )

    for (
        comparison,
        lhs_method,
        rhs_method,
    ) in T0_PAIRS:

        result = paired_cluster_summary(
            [
                r
                for r in scoped
                if r["method"]
                == lhs_method
            ],
            [
                r
                for r in scoped
                if r["method"]
                == rhs_method
            ],
            stratified=(
                dataset_label
                ==
                "Overall"
            ),
        )

        if result is None:
            continue

        t0_pair_rows.append({
            "dataset":
                dataset_label,

            "comparison":
                comparison,

            **result,

            "orientation":
                "negative = lhs more human-process typical",
        })


write_csv(
    T0_PAIR_OUT,
    t0_pair_rows,
)


# =============================================================================
# T0 BY GENERATOR + DATASET×GENERATOR
# =============================================================================

t0_gen_rows = []
t0_pair_gen_rows = []
t0_dg_rows = []

for generator in GENERATORS:

    scoped = group_filter(
        t0_records,
        generator=generator,
    )

    for method in [
        "Original",
        "Vanilla",
        "Temperature Control",
        "CoPA-3B",
        "HPCD",
    ]:

        subset = [
            r
            for r in scoped
            if r["method"]
            == method
        ]

        s = absolute_cluster_summary(
            subset
        )

        t0_gen_rows.append({
            "source_generator":
                generator,

            "method":
                method,

            **s,
        })

    for (
        comparison,
        lhs_method,
        rhs_method,
    ) in T0_PAIRS:

        result = paired_cluster_summary(
            [
                r
                for r in scoped
                if r["method"]
                == lhs_method
            ],
            [
                r
                for r in scoped
                if r["method"]
                == rhs_method
            ],
        )

        if result is None:
            continue

        t0_pair_gen_rows.append({
            "source_generator":
                generator,

            "comparison":
                comparison,

            **result,
        })


for dataset_label, domain in DATASETS[1:]:

    for generator in GENERATORS:

        scoped = group_filter(
            t0_records,
            domain=domain,
            generator=generator,
        )

        for method in [
            "Original",
            "Vanilla",
            "Temperature Control",
            "CoPA-3B",
            "HPCD",
        ]:

            subset = [
                r
                for r in scoped
                if r["method"]
                == method
            ]

            if not subset:
                continue

            s = absolute_cluster_summary(
                subset
            )

            t0_dg_rows.append({
                "dataset":
                    dataset_label,

                "source_generator":
                    generator,

                "method":
                    method,

                **s,
            })


write_csv(
    T0_GEN_OUT,
    t0_gen_rows,
)

write_csv(
    T0_PAIR_GEN_OUT,
    t0_pair_gen_rows,
)

write_csv(
    T0_DG_OUT,
    t0_dg_rows,
)


# =============================================================================
# DETECTOR INPUTS
# =============================================================================

thresholds = {}

with THRESHOLD_FILE.open(
    encoding="utf-8",
) as f:

    for r in csv.DictReader(f):
        thresholds[
            r["metric"]
        ] = float(
            r["threshold"]
        )


detector_sets = {}


def normalize_detector_rows(
    rows,
    detector_name,
    score_fn,
):
    out = []

    for r in rows:

        condition = r[
            "condition"
        ]

        method = COND_DISPLAY[
            condition
        ]

        out.append({
            "base_id":
                r["base_id"],

            "domain":
                r["domain"],

            "source_generator":
                (
                    r.get(
                        "source_generator"
                    )
                    if condition
                    !=
                    "human"
                    else
                    "human"
                ),

            "condition":
                condition,

            "method":
                method,

            "pair_key":
                (
                    pair_key(r)
                    if condition
                    !=
                    "human"
                    else
                    r["base_id"]
                ),

            "analysis_score":
                float(
                    score_fn(r)
                ),

            "detector":
                detector_name,
        })

    return out


detector_sets[
    "Fast-DetectGPT"
] = normalize_detector_rows(
    fast_raw,
    "Fast-DetectGPT",
    lambda r:
        r["analysis_score"],
)

detector_sets[
    "RADAR"
] = normalize_detector_rows(
    radar_raw,
    "RADAR",
    lambda r:
        r["analysis_score"],
)

detector_sets[
    "SurpMark"
] = normalize_detector_rows(
    surp_raw,
    "SurpMark",
    lambda r:
        r["analysis_score"],
)

detector_sets[
    "LogLikelihood"
] = normalize_detector_rows(
    simple_raw,
    "LogLikelihood",
    lambda r:
        r["mean_log_likelihood"],
)

detector_sets[
    "LogRank"
] = normalize_detector_rows(
    simple_raw,
    "LogRank",
    lambda r:
        -float(
            r["mean_log_rank"]
        ),
)

detector_sets[
    "DetectLLM-LRR"
] = normalize_detector_rows(
    simple_raw,
    "DetectLLM-LRR",
    lambda r:
        r["detectllm_lrr"],
)


CALIBRATED = {
    "Fast-DetectGPT",
    "RADAR",
    "SurpMark",
}


# =============================================================================
# DETECTOR METRICS BY DATASET
# =============================================================================

detector_metric_rows = []

for dataset_label, domain in DATASETS:

    for detector, rows in (
        detector_sets.items()
    ):

        human_rows = [
            r
            for r in rows
            if (
                r["condition"]
                ==
                "human"
                and
                (
                    domain is None
                    or
                    r["domain"]
                    == domain
                )
            )
        ]

        # Human descriptive mean.
        hs = absolute_cluster_summary(
            [
                {
                    **r,
                    "value":
                        r[
                            "analysis_score"
                        ],
                }
                for r in human_rows
            ]
        )

        detector_metric_rows.append({
            "dataset":
                dataset_label,

            "detector":
                detector,

            "method":
                "Human",

            "n_base_ids":
                hs[
                    "n_base_ids"
                ],

            "n_instances":
                hs[
                    "n_instances"
                ],

            "mean_analysis_score":
                hs[
                    "mean"
                ],

            "cluster_mean_analysis_score":
                hs[
                    "cluster_mean"
                ],

            "score_ci_low":
                hs[
                    "cluster_ci_low"
                ],

            "score_ci_high":
                hs[
                    "cluster_ci_high"
                ],

            "auroc":
                "",

            "auroc_ci_low":
                "",

            "auroc_ci_high":
                "",

            "tpr_at_5fpr":
                "",

            "tpr5_ci_low":
                "",

            "tpr5_ci_high":
                "",

            "threshold":
                "",
        })


        for method in [
            "Original",
            "Vanilla",
            "Temperature Control",
            "CoPA-3B",
            "HPCD",
        ]:

            condition_aliases = {
                "Original":
                    {"original"},

                "Vanilla":
                    {"vanilla_t10"},

                "Temperature Control":
                    {"vanilla_t11"},

                "CoPA-3B":
                    {
                        "copa3b",
                        "copa_3b",
                    },

                "HPCD":
                    {"hpcd_n0"},
            }

            machine_rows = [
                r
                for r in rows
                if (
                    r["condition"]
                    in
                    condition_aliases[
                        method
                    ]
                    and
                    (
                        domain is None
                        or
                        r["domain"]
                        == domain
                    )
                )
            ]

            ms = absolute_cluster_summary(
                [
                    {
                        **r,
                        "value":
                            r[
                                "analysis_score"
                            ],
                    }
                    for r in machine_rows
                ]
            )

            (
                auc,
                auc_lo,
                auc_hi,
                auc_bases,
            ) = cluster_auc(
                human_rows,
                machine_rows,
            )

            if detector in CALIBRATED:

                threshold = thresholds[
                    detector
                ]

                (
                    tpr,
                    tpr_lo,
                    tpr_hi,
                ) = cluster_tpr(
                    machine_rows,
                    threshold,
                )

            else:
                threshold = ""
                tpr = ""
                tpr_lo = ""
                tpr_hi = ""

            detector_metric_rows.append({
                "dataset":
                    dataset_label,

                "detector":
                    detector,

                "method":
                    method,

                "n_base_ids":
                    ms[
                        "n_base_ids"
                    ],

                "n_instances":
                    ms[
                        "n_instances"
                    ],

                "mean_analysis_score":
                    ms[
                        "mean"
                    ],

                "cluster_mean_analysis_score":
                    ms[
                        "cluster_mean"
                    ],

                "score_ci_low":
                    ms[
                        "cluster_ci_low"
                    ],

                "score_ci_high":
                    ms[
                        "cluster_ci_high"
                    ],

                "auroc":
                    auc,

                "auroc_ci_low":
                    auc_lo,

                "auroc_ci_high":
                    auc_hi,

                "tpr_at_5fpr":
                    tpr,

                "tpr5_ci_low":
                    tpr_lo,

                "tpr5_ci_high":
                    tpr_hi,

                "threshold":
                    threshold,
            })


write_csv(
    DET_METRICS_OUT,
    detector_metric_rows,
)


# =============================================================================
# DETECTOR PAIRED SCORE SHIFTS BY DATASET
# =============================================================================

DET_PAIRS = [
    (
        "Vanilla - Original",
        "Vanilla",
        "Original",
    ),
    (
        "Temperature Control - Original",
        "Temperature Control",
        "Original",
    ),
    (
        "CoPA-3B - Original",
        "CoPA-3B",
        "Original",
    ),
    (
        "HPCD - Original",
        "HPCD",
        "Original",
    ),
    (
        "HPCD - Vanilla",
        "HPCD",
        "Vanilla",
    ),
    (
        "HPCD - Temperature Control",
        "HPCD",
        "Temperature Control",
    ),
    (
        "HPCD - CoPA-3B",
        "HPCD",
        "CoPA-3B",
    ),
]

detector_pair_rows = []

for dataset_label, domain in DATASETS:

    for detector, rows in (
        detector_sets.items()
    ):

        scoped = [
            r
            for r in rows
            if (
                r["condition"]
                !=
                "human"
                and
                (
                    domain is None
                    or
                    r["domain"]
                    == domain
                )
            )
        ]

        for (
            comparison,
            lhs_method,
            rhs_method,
        ) in DET_PAIRS:

            lhs = [
                {
                    **r,
                    "value":
                        r[
                            "analysis_score"
                        ],
                }
                for r in scoped
                if r["method"]
                == lhs_method
            ]

            rhs = [
                {
                    **r,
                    "value":
                        r[
                            "analysis_score"
                        ],
                }
                for r in scoped
                if r["method"]
                == rhs_method
            ]

            result = paired_cluster_summary(
                lhs,
                rhs,
            )

            if result is None:
                continue

            detector_pair_rows.append({
                "dataset":
                    dataset_label,

                "detector":
                    detector,

                "comparison":
                    comparison,

                **result,

                "orientation":
                    "negative = lhs less machine-like",
            })


write_csv(
    DET_PAIR_OUT,
    detector_pair_rows,
)


# =============================================================================
# DETECTOR BY GENERATOR + DATASET×GENERATOR
# =============================================================================

detector_gen_rows = []
detector_pair_gen_rows = []
detector_dg_rows = []

for generator in GENERATORS:

    for detector, rows in (
        detector_sets.items()
    ):

        machine_scope = [
            r
            for r in rows
            if (
                r["condition"]
                !=
                "human"
                and
                r[
                    "source_generator"
                ]
                ==
                generator
            )
        ]

        machine_bases = {
            r["base_id"]
            for r in machine_scope
        }

        human_rows = [
            r
            for r in rows
            if (
                r["condition"]
                ==
                "human"
                and
                r["base_id"]
                in
                machine_bases
            )
        ]

        for method in [
            "Original",
            "Vanilla",
            "Temperature Control",
            "CoPA-3B",
            "HPCD",
        ]:

            machine_rows = [
                r
                for r in machine_scope
                if r["method"]
                == method
            ]

            s = absolute_cluster_summary(
                [
                    {
                        **r,
                        "value":
                            r[
                                "analysis_score"
                            ],
                    }
                    for r in machine_rows
                ]
            )

            (
                auc,
                auc_lo,
                auc_hi,
                auc_bases,
            ) = cluster_auc(
                human_rows,
                machine_rows,
            )

            if detector in CALIBRATED:
                (
                    tpr,
                    tpr_lo,
                    tpr_hi,
                ) = cluster_tpr(
                    machine_rows,
                    thresholds[
                        detector
                    ],
                )
            else:
                tpr = ""
                tpr_lo = ""
                tpr_hi = ""

            detector_gen_rows.append({
                "source_generator":
                    generator,

                "detector":
                    detector,

                "method":
                    method,

                "n_base_ids":
                    s[
                        "n_base_ids"
                    ],

                "n_instances":
                    s[
                        "n_instances"
                    ],

                "mean_analysis_score":
                    s[
                        "mean"
                    ],

                "auroc":
                    auc,

                "auroc_ci_low":
                    auc_lo,

                "auroc_ci_high":
                    auc_hi,

                "tpr_at_5fpr":
                    tpr,

                "tpr5_ci_low":
                    tpr_lo,

                "tpr5_ci_high":
                    tpr_hi,
            })


        for (
            comparison,
            lhs_method,
            rhs_method,
        ) in DET_PAIRS:

            result = paired_cluster_summary(
                [
                    {
                        **r,
                        "value":
                            r[
                                "analysis_score"
                            ],
                    }
                    for r in machine_scope
                    if r["method"]
                    == lhs_method
                ],
                [
                    {
                        **r,
                        "value":
                            r[
                                "analysis_score"
                            ],
                    }
                    for r in machine_scope
                    if r["method"]
                    == rhs_method
                ],
            )

            if result is None:
                continue

            detector_pair_gen_rows.append({
                "source_generator":
                    generator,

                "detector":
                    detector,

                "comparison":
                    comparison,

                **result,
            })


for dataset_label, domain in DATASETS[1:]:

    for generator in GENERATORS:

        for detector, rows in (
            detector_sets.items()
        ):

            machine_scope = [
                r
                for r in rows
                if (
                    r["condition"]
                    !=
                    "human"
                    and
                    r["domain"]
                    == domain
                    and
                    r[
                        "source_generator"
                    ]
                    ==
                    generator
                )
            ]

            if not machine_scope:
                continue

            machine_bases = {
                r["base_id"]
                for r in machine_scope
            }

            human_rows = [
                r
                for r in rows
                if (
                    r["condition"]
                    ==
                    "human"
                    and
                    r["domain"]
                    == domain
                    and
                    r["base_id"]
                    in
                    machine_bases
                )
            ]

            for method in [
                "Original",
                "Vanilla",
                "Temperature Control",
                "CoPA-3B",
                "HPCD",
            ]:

                machine_rows = [
                    r
                    for r in machine_scope
                    if r["method"]
                    == method
                ]

                if not machine_rows:
                    continue

                s = absolute_cluster_summary(
                    [
                        {
                            **r,
                            "value":
                                r[
                                    "analysis_score"
                                ],
                        }
                        for r in machine_rows
                    ]
                )

                (
                    auc,
                    auc_lo,
                    auc_hi,
                    auc_bases,
                ) = cluster_auc(
                    human_rows,
                    machine_rows,
                )

                if detector in CALIBRATED:
                    (
                        tpr,
                        tpr_lo,
                        tpr_hi,
                    ) = cluster_tpr(
                        machine_rows,
                        thresholds[
                            detector
                        ],
                    )
                else:
                    tpr = ""
                    tpr_lo = ""
                    tpr_hi = ""

                detector_dg_rows.append({
                    "dataset":
                        dataset_label,

                    "source_generator":
                        generator,

                    "detector":
                        detector,

                    "method":
                        method,

                    "n_base_ids":
                        s[
                            "n_base_ids"
                        ],

                    "n_instances":
                        s[
                            "n_instances"
                        ],

                    "mean_analysis_score":
                        s[
                            "mean"
                        ],

                    "auroc":
                        auc,

                    "auroc_ci_low":
                        auc_lo,

                    "auroc_ci_high":
                        auc_hi,

                    "tpr_at_5fpr":
                        tpr,

                    "tpr5_ci_low":
                        tpr_lo,

                    "tpr5_ci_high":
                        tpr_hi,
                })


write_csv(
    DET_GEN_OUT,
    detector_gen_rows,
)

write_csv(
    DET_PAIR_GEN_OUT,
    detector_pair_gen_rows,
)

write_csv(
    DET_DG_OUT,
    detector_dg_rows,
)


# =============================================================================
# COPY EXISTING FROZEN ASR TABLES
# =============================================================================

if not ASR_SOURCE.exists():
    raise RuntimeError(
        "Frozen ASR table missing"
    )

if not ASR_PAIR_SOURCE.exists():
    raise RuntimeError(
        "Frozen ASR pairwise table missing"
    )

shutil.copyfile(
    ASR_SOURCE,
    DET_ASR_OUT,
)

shutil.copyfile(
    ASR_PAIR_SOURCE,
    DET_ASR_PAIR_OUT,
)

if (
    sha256_file(
        ASR_SOURCE
    )
    !=
    sha256_file(
        DET_ASR_OUT
    )
):
    raise RuntimeError(
        "ASR copy mismatch"
    )

print("FROZEN_ASR_COPY=PASS")


# =============================================================================
# RAW SIMPLE SCORES BY DATASET
# =============================================================================

SIMPLE_METRICS = [
    (
        "mean_log_likelihood",
        "mean_log_likelihood",
    ),
    (
        "mean_nll",
        "mean_nll",
    ),
    (
        "perplexity",
        "perplexity",
    ),
    (
        "mean_predictive_entropy",
        "mean_predictive_entropy",
    ),
    (
        "mean_rank",
        "mean_rank",
    ),
    (
        "mean_log_rank",
        "mean_log_rank",
    ),
    (
        "detectllm_lrr",
        "detectllm_lrr",
    ),
]

simple_records = []

for r in simple_raw:

    method = COND_DISPLAY[
        r["condition"]
    ]

    for metric, field in (
        SIMPLE_METRICS
    ):
        simple_records.append({
            "base_id":
                r["base_id"],

            "domain":
                r["domain"],

            "source_generator":
                (
                    "human"
                    if r["condition"]
                    ==
                    "human"
                    else
                    r[
                        "source_generator"
                    ]
                ),

            "condition":
                r["condition"],

            "method":
                method,

            "metric":
                metric,

            "value":
                float(
                    r[field]
                ),
        })


simple_rows = []

for dataset_label, domain in DATASETS:

    scoped = group_filter(
        simple_records,
        domain=domain,
    )

    for method in METHOD_ORDER:

        for metric, _ in (
            SIMPLE_METRICS
        ):
            subset = [
                r
                for r in scoped
                if (
                    r["method"]
                    == method
                    and
                    r["metric"]
                    == metric
                )
            ]

            if not subset:
                continue

            s = absolute_cluster_summary(
                subset
            )

            simple_rows.append({
                "dataset":
                    dataset_label,

                "method":
                    method,

                "metric":
                    metric,

                **s,
            })


write_csv(
    SIMPLE_OUT,
    simple_rows,
)


simple_gen_rows = []
simple_dg_rows = []

for generator in GENERATORS:

    scoped = group_filter(
        simple_records,
        generator=generator,
    )

    for method in [
        "Original",
        "Vanilla",
        "Temperature Control",
        "CoPA-3B",
        "HPCD",
    ]:
        for metric, _ in SIMPLE_METRICS:

            subset = [
                r
                for r in scoped
                if (
                    r["method"]
                    == method
                    and
                    r["metric"]
                    == metric
                )
            ]

            if not subset:
                continue

            s = absolute_cluster_summary(
                subset
            )

            simple_gen_rows.append({
                "source_generator":
                    generator,

                "method":
                    method,

                "metric":
                    metric,

                **s,
            })


for dataset_label, domain in DATASETS[1:]:

    for generator in GENERATORS:

        scoped = group_filter(
            simple_records,
            domain=domain,
            generator=generator,
        )

        for method in [
            "Original",
            "Vanilla",
            "Temperature Control",
            "CoPA-3B",
            "HPCD",
        ]:
            for metric, _ in SIMPLE_METRICS:

                subset = [
                    r
                    for r in scoped
                    if (
                        r["method"]
                        == method
                        and
                        r["metric"]
                        == metric
                    )
                ]

                if not subset:
                    continue

                s = absolute_cluster_summary(
                    subset
                )

                simple_dg_rows.append({
                    "dataset":
                        dataset_label,

                    "source_generator":
                        generator,

                    "method":
                        method,

                    "metric":
                        metric,

                    **s,
                })


write_csv(
    SIMPLE_GEN_OUT,
    simple_gen_rows,
)

write_csv(
    SIMPLE_DG_OUT,
    simple_dg_rows,
)


# =============================================================================
# MAIN COMPACT DOMAIN-GENERALIZATION TABLE
# =============================================================================

def quality_mean(
    dataset,
    method,
    metric,
):
    rows = [
        r
        for r in quality_rows
        if (
            r["dataset"]
            == dataset
            and
            r["method"]
            == method
            and
            r["metric"]
            == metric
        )
    ]

    if not rows:
        return ""

    return rows[0][
        "mean"
    ]


def t0_mean(
    dataset,
    method,
):
    rows = [
        r
        for r in t0_rows
        if (
            r["dataset"]
            == dataset
            and
            r["method"]
            == method
        )
    ]

    if not rows:
        return ""

    return rows[0][
        "mean"
    ]


main_domain_rows = []

for dataset_label, _ in DATASETS:

    for method in [
        "Vanilla",
        "CoPA-3B",
        "HPCD",
    ]:

        main_domain_rows.append({
            "dataset":
                dataset_label,

            "method":
                method,

            "T0_down":
                t0_mean(
                    dataset_label,
                    method,
                ),

            "NLI_min_up":
                quality_mean(
                    dataset_label,
                    method,
                    "NLI-min",
                ),

            "MPNet_up":
                quality_mean(
                    dataset_label,
                    method,
                    "MPNet",
                ),

            "Entity_up":
                quality_mean(
                    dataset_label,
                    method,
                    "entity_preservation",
                ),

            "NumDate_up":
                quality_mean(
                    dataset_label,
                    method,
                    "number_date_preservation",
                ),
        })


write_csv(
    MAIN_DOMAIN_TABLE,
    main_domain_rows,
)


# =============================================================================
# DETECTOR TPR5 DOMAIN TABLE
# =============================================================================

detector_domain_rows = []

for dataset_label, _ in DATASETS:

    for method in [
        "Original",
        "Vanilla",
        "CoPA-3B",
        "HPCD",
    ]:

        rec = {
            "dataset":
                dataset_label,

            "method":
                method,
        }

        for detector in [
            "Fast-DetectGPT",
            "RADAR",
            "SurpMark",
        ]:

            rows = [
                r
                for r in detector_metric_rows
                if (
                    r["dataset"]
                    == dataset_label
                    and
                    r["method"]
                    == method
                    and
                    r["detector"]
                    == detector
                )
            ]

            if not rows:
                raise RuntimeError(
                    f"Missing TPR row: "
                    f"{dataset_label}/{method}/{detector}"
                )

            rec[
                detector
                +
                "_TPR5"
            ] = rows[0][
                "tpr_at_5fpr"
            ]

        detector_domain_rows.append(
            rec
        )


write_csv(
    DETECTOR_DOMAIN_TABLE,
    detector_domain_rows,
)


# =============================================================================
# TERMINAL SUMMARY
# =============================================================================

print()
print("=" * 100)
print("T0: HPCD - VANILLA BY DATASET")
print("=" * 100)

for dataset_label, _ in DATASETS:

    r = next(
        x
        for x in t0_pair_rows
        if (
            x["dataset"]
            == dataset_label
            and
            x["comparison"]
            ==
            "HPCD - Vanilla"
        )
    )

    print(
        f"{dataset_label:8s} "
        f"Δ={r['cluster_mean_delta']:+.6f} "
        f"CI=[{r['cluster_ci_low']:+.6f}, "
        f"{r['cluster_ci_high']:+.6f}] "
        f"P(Δ<0)="
        f"{r['bootstrap_prob_delta_lt_0']:.5f}"
    )


print()
print("=" * 100)
print("MAIN DOMAIN GENERALIZATION TABLE")
print("=" * 100)

for r in main_domain_rows:

    print(
        f"{r['dataset']:8s} "
        f"{r['method']:20s} "
        f"T0={float(r['T0_down']):.4f} "
        f"NLI={float(r['NLI_min_up']):.4f} "
        f"MPNet={float(r['MPNet_up']):.4f} "
        f"Entity={float(r['Entity_up']):.4f} "
        f"NumDate="
        f"{float(r['NumDate_up']):.4f}"
        if r["NumDate_up"] != ""
        else
        f"{r['dataset']:8s} "
        f"{r['method']:20s} "
        f"T0={float(r['T0_down']):.4f} "
        f"NLI={float(r['NLI_min_up']):.4f} "
        f"MPNet={float(r['MPNet_up']):.4f} "
        f"Entity={float(r['Entity_up']):.4f} "
        f"NumDate=NA"
    )


print()
print("=" * 100)
print("HPCD TPR@5 BY DATASET")
print("=" * 100)

for dataset_label, _ in DATASETS:

    vals = {}

    for detector in [
        "Fast-DetectGPT",
        "RADAR",
        "SurpMark",
    ]:
        r = next(
            x
            for x in detector_metric_rows
            if (
                x["dataset"]
                == dataset_label
                and
                x["method"]
                == "HPCD"
                and
                x["detector"]
                == detector
            )
        )

        vals[
            detector
        ] = float(
            r[
                "tpr_at_5fpr"
            ]
        )

    print(
        f"{dataset_label:8s} "
        f"Fast={vals['Fast-DetectGPT']:.4f} "
        f"RADAR={vals['RADAR']:.4f} "
        f"SurpMark={vals['SurpMark']:.4f}"
    )


# =============================================================================
# MANIFEST
# =============================================================================

outputs = [
    QUALITY_OUT,
    QUALITY_PAIR_OUT,
    QUALITY_GEN_OUT,
    QUALITY_PAIR_GEN_OUT,
    QUALITY_DG_OUT,

    T0_OUT,
    T0_PAIR_OUT,
    T0_GEN_OUT,
    T0_PAIR_GEN_OUT,
    T0_DG_OUT,

    DET_METRICS_OUT,
    DET_PAIR_OUT,
    DET_GEN_OUT,
    DET_PAIR_GEN_OUT,
    DET_DG_OUT,
    DET_ASR_OUT,
    DET_ASR_PAIR_OUT,

    SIMPLE_OUT,
    SIMPLE_GEN_OUT,
    SIMPLE_DG_OUT,

    MAIN_DOMAIN_TABLE,
    DETECTOR_DOMAIN_TABLE,
]


manifest = {
    "status":
        "PASS",

    "analysis_only":
        True,

    "new_model_inference":
        False,

    "bootstrap_replicates":
        B,

    "bootstrap_seed":
        SEED,

    "primary_inference_unit":
        "RAID base_id",

    "levels":
        [
            "Overall",
            "By dataset",
            "By source generator",
            "Dataset x source generator diagnostics",
        ],

    "dataset_order":
        [
            "Overall",
            "News",
            "Reddit",
            "Reviews",
            "Wiki",
        ],

    "source_generators":
        GENERATORS,

    "BERTScore":
        "not included because not completed",

    "NLI_original":
        (
            "not fabricated; Original is identity/source reference "
            "and was not part of frozen 380-pair NLI run"
        ),

    "input_sha256": {
        str(path):
            sha256_file(path)
        for path in EXPECTED_SHA
    },

    "outputs": {
        str(path):
            sha256_file(path)
        for path in outputs
    },
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
print("=" * 100)
print("FINAL OUTPUT HASHES")
print("=" * 100)

for path in outputs + [
    MANIFEST_OUT,
]:
    print(
        sha256_file(path),
        path,
    )


print()
print("=" * 100)
print("Q7_FINAL_DOMAIN_GENERATOR_AGGREGATION=PASS")
print("=" * 100)
