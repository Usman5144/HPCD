#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import spacy


ROOT = Path(".").resolve()

CANONICAL = ROOT / (
    "results/detectors/fresh95_final_v1/inputs/"
    "fresh95_detector_input_v1.jsonl"
)

EXPECTED_CANONICAL_SHA = (
    "5e5bc0f60923b43675be252f6c3651e7"
    "0338c46ac4ee1a6ba3d211e8f0cdfcb5"
)

HISTORICAL = ROOT / (
    "code/hpcd/"
    "finalize_fresh95_surface_entity_selection_v1.py"
)

EXPECTED_HISTORICAL_SHA = (
    "319fc639a7c705c4eeaf779e4ae50740"
    "fbd919b98bdfc0390c45ffebe9e78b0f"
)

OUTDIR = ROOT / (
    "results/quality/final_semantic_v1/entity"
)

SCORES = OUTDIR / "entity_scores_v1.jsonl"
BY_DATASET = OUTDIR / "entity_by_dataset_v1.csv"
PAIR_DATASET = OUTDIR / "entity_pairwise_by_dataset_v1.csv"
BY_GENERATOR = OUTDIR / "entity_by_generator_v1.csv"
BY_DATASET_GENERATOR = (
    OUTDIR / "entity_by_dataset_generator_v1.csv"
)
MANIFEST = OUTDIR / "entity_manifest_v1.json"

B = 50_000
SEED = 2027

CONDITIONS = [
    "original",
    "vanilla_t10",
    "vanilla_t11",
    "copa3b",
    "hpcd_n0",
]

DISPLAY = {
    "original":
        "Original",

    "vanilla_t10":
        "Vanilla",

    "vanilla_t11":
        "Temperature Control",

    "copa3b":
        "CoPA-3B",

    "hpcd_n0":
        "HPCD",
}

DATASETS = [
    ("Overall", None),
    ("News", "news"),
    ("Reddit", "reddit"),
    ("Reviews", "reviews"),
    ("Wiki", "wiki"),
]

PAIRINGS = [
    (
        "HPCD - Vanilla",
        "hpcd_n0",
        "vanilla_t10",
    ),
    (
        "HPCD - Temperature Control",
        "hpcd_n0",
        "vanilla_t11",
    ),
    (
        "HPCD - CoPA-3B",
        "hpcd_n0",
        "copa3b",
    ),
]


def sha256_file(path: Path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def load_jsonl(path):
    with path.open(
        encoding="utf-8",
    ) as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def write_jsonl(path, rows):
    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_csv(path, rows):
    rows = list(rows)

    if not rows:
        raise RuntimeError(
            f"No output rows: {path}"
        )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def ci(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(values)
    ]

    lo, hi = np.quantile(
        values,
        [0.025, 0.975],
    )

    return float(lo), float(hi)


print("=" * 100)
print("FROZEN SOURCE-ONLY ENTITY PRESERVATION")
print("=" * 100)


# ============================================================================
# 1. Verify frozen provenance.
# ============================================================================

if (
    sha256_file(CANONICAL)
    != EXPECTED_CANONICAL_SHA
):
    raise RuntimeError(
        "Canonical Fresh95 SHA mismatch"
    )

observed_historical = sha256_file(
    HISTORICAL
)

if (
    observed_historical
    != EXPECTED_HISTORICAL_SHA
):
    raise RuntimeError(
        "Historical entity validator SHA mismatch"
    )

print("canonical SHA :", sha256_file(CANONICAL))
print("validator SHA :", observed_historical)


# ============================================================================
# 2. Dynamically import EXACT historical functions.
# ============================================================================

spec = importlib.util.spec_from_file_location(
    "frozen_entity_validator",
    HISTORICAL,
)

module = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    module
)

required = [
    "normalize_surface",
    "source_entities",
    "surface_entity_check",
]

for name in required:
    if not hasattr(
        module,
        name,
    ):
        raise RuntimeError(
            f"Historical function missing: {name}"
        )

normalize_surface = module.normalize_surface
source_entities = module.source_entities
surface_entity_check = module.surface_entity_check

print("HISTORICAL_FUNCTION_IMPORT=PASS")


# ============================================================================
# 3. Load exact spaCy model used historically.
# ============================================================================

nlp = spacy.load(
    "en_core_web_sm"
)

print(
    "spaCy:",
    spacy.__version__,
)

try:
    import en_core_web_sm

    print(
        "en_core_web_sm:",
        en_core_web_sm.__version__,
    )
except Exception:
    print(
        "en_core_web_sm version: unavailable"
    )

print(
    "pipes:",
    nlp.pipe_names,
)


# ============================================================================
# 4. Canonical alignment.
# ============================================================================

rows = load_jsonl(
    CANONICAL
)

if len(rows) != 523:
    raise RuntimeError(
        f"Expected 523 rows, got {len(rows)}"
    )

expected_counts = Counter({
    "human": 48,
    "original": 95,
    "vanilla_t10": 95,
    "vanilla_t11": 95,
    "copa3b": 95,
    "hpcd_n0": 95,
})

if Counter(
    r["condition"]
    for r in rows
) != expected_counts:
    raise RuntimeError(
        "Canonical condition counts mismatch"
    )

original = {
    r["source_sample_id"]: r
    for r in rows
    if r["condition"]
    == "original"
}

if len(original) != 95:
    raise RuntimeError(
        "Expected 95 originals"
    )

maps = {}

for condition in CONDITIONS:

    m = {
        r["source_sample_id"]: r
        for r in rows
        if r["condition"]
        == condition
    }

    if len(m) != 95:
        raise RuntimeError(
            f"{condition}: expected 95"
        )

    if set(m) != set(original):
        raise RuntimeError(
            f"{condition}: source alignment failure"
        )

    maps[
        condition
    ] = m

print("CANONICAL_ALIGNMENT=PASS")


# ============================================================================
# 5. Extract source entities ONCE using historical source-only implementation.
# ============================================================================

source_entities_cache = {}

print()
print("===== SOURCE ENTITY EXTRACTION =====")

for i, sid in enumerate(
    sorted(original),
    1,
):
    src = original[sid]

    ents = source_entities(
        nlp,
        src["text"],
    )

    source_entities_cache[
        sid
    ] = ents

    if i % 20 == 0 or i == 95:
        print(
            f"{i}/95 sources processed"
        )


entity_counts = np.asarray(
    [
        len(
            source_entities_cache[
                sid
            ]
        )
        for sid in sorted(
            source_entities_cache
        )
    ],
    dtype=int,
)

print()
print(
    "source docs with >=1 entity:",
    int(
        np.sum(
            entity_counts > 0
        )
    ),
    "/95",
)

print(
    "source docs with 0 entities:",
    int(
        np.sum(
            entity_counts == 0
        )
    ),
    "/95",
)

print(
    "total unique source entity surfaces:",
    int(
        entity_counts.sum()
    ),
)

print(
    "median entities/source:",
    float(
        np.median(
            entity_counts
        )
    ),
)


# ============================================================================
# 6. Score all five machine conditions.
#
# IMPORTANT ZERO-ENTITY RULE:
#
# entity_preservation_rate:
#   NA for source docs containing zero evaluated entities.
#
# all_entities_preserved:
#   True for zero-entity documents, matching the historical gate's
#   vacuous-pass behavior.
#
# This is recorded explicitly and never hidden.
# ============================================================================

score_rows = []

for condition in CONDITIONS:

    for sid in sorted(original):

        src = original[sid]
        out = maps[
            condition
        ][sid]

        assert (
            src["base_id"]
            ==
            out["base_id"]
        )

        assert (
            src["domain"]
            ==
            out["domain"]
        )

        assert (
            src["source_generator"]
            ==
            out["source_generator"]
        )

        entities = (
            source_entities_cache[
                sid
            ]
        )

        ok, missing = (
            surface_entity_check(
                entities,
                out["text"],
            )
        )

        total = len(
            entities
        )

        missing_n = len(
            missing
        )

        preserved = (
            total
            -
            missing_n
        )

        if (
            preserved < 0
            or
            preserved > total
        ):
            raise RuntimeError(
                "Invalid preserved count"
            )

        if total == 0:
            preservation_rate = None
            denominator_status = (
                "zero_source_entities"
            )
        else:
            preservation_rate = (
                preserved
                /
                total
            )
            denominator_status = (
                "nonzero_source_entities"
            )

        score_rows.append({
            "source_sample_id":
                sid,

            "sample_id":
                out.get(
                    "sample_id"
                ),

            "base_id":
                src["base_id"],

            "domain":
                src["domain"],

            "source_generator":
                src[
                    "source_generator"
                ],

            "condition":
                condition,

            "method":
                DISPLAY[
                    condition
                ],

            "source_text_sha256":
                src["text_sha256"],

            "rewrite_text_sha256":
                out["text_sha256"],

            "source_entity_count":
                total,

            "preserved_entity_count":
                preserved,

            "missing_entity_count":
                missing_n,

            "entity_preservation_rate":
                preservation_rate,

            "all_entities_preserved":
                bool(
                    ok
                ),

            "zero_entity_source":
                bool(
                    total == 0
                ),

            "denominator_status":
                denominator_status,

            "source_entity_surfaces":
                entities,

            "missing_source_entity_surfaces":
                list(
                    missing
                ),

            "status":
                "ok",
        })


if len(score_rows) != 475:
    raise RuntimeError(
        f"Expected 475 rows, got {len(score_rows)}"
    )

write_jsonl(
    SCORES,
    score_rows,
)

print()
print("ENTITY_SCORE_ROWS=475")


# ============================================================================
# 7. Statistical helpers.
# ============================================================================

def cluster_summary(
    subset,
    value_fn,
):
    usable = []

    for row in subset:
        value = value_fn(
            row
        )

        if value is None:
            continue

        value = float(
            value
        )

        if math.isfinite(
            value
        ):
            usable.append(
                (
                    row,
                    value,
                )
            )

    if not usable:
        return {
            "n_base_ids":
                0,

            "n_instances":
                len(subset),

            "n_metric_instances":
                0,

            "mean":
                float("nan"),

            "median":
                float("nan"),

            "std":
                float("nan"),

            "cluster_mean":
                float("nan"),

            "cluster_ci_low":
                float("nan"),

            "cluster_ci_high":
                float("nan"),
        }

    by_base = defaultdict(
        list
    )

    vals = []

    for row, value in usable:
        by_base[
            row["base_id"]
        ].append(
            value
        )
        vals.append(
            value
        )

    base_values = np.asarray(
        [
            np.mean(
                by_base[b]
            )
            for b in sorted(
                by_base
            )
        ],
        dtype=np.float64,
    )

    vals = np.asarray(
        vals,
        dtype=np.float64,
    )

    rng = np.random.default_rng(
        SEED
    )

    draws = rng.integers(
        0,
        len(base_values),
        size=(
            B,
            len(base_values),
        ),
    )

    boot = base_values[
        draws
    ].mean(
        axis=1
    )

    lo, hi = ci(
        boot
    )

    return {
        "n_base_ids":
            len(
                base_values
            ),

        "n_instances":
            len(
                subset
            ),

        "n_metric_instances":
            len(
                usable
            ),

        "mean":
            float(
                np.mean(
                    vals
                )
            ),

        "median":
            float(
                np.median(
                    vals
                )
            ),

        "std":
            float(
                np.std(
                    vals,
                    ddof=1,
                )
            )
            if len(vals) > 1
            else 0.0,

        "cluster_mean":
            float(
                np.mean(
                    base_values
                )
            ),

        "cluster_ci_low":
            lo,

        "cluster_ci_high":
            hi,
    }


def paired_summary(
    lhs_rows,
    rhs_rows,
    value_fn,
):
    lhs = {
        r["source_sample_id"]: r
        for r in lhs_rows
    }

    rhs = {
        r["source_sample_id"]: r
        for r in rhs_rows
    }

    common = sorted(
        set(lhs)
        &
        set(rhs)
    )

    by_base = defaultdict(
        list
    )

    instance_diffs = []

    for sid in common:

        a = lhs[sid]
        b = rhs[sid]

        if (
            a["base_id"]
            !=
            b["base_id"]
        ):
            raise RuntimeError(
                "Paired base mismatch"
            )

        av = value_fn(
            a
        )
        bv = value_fn(
            b
        )

        # For entity_preservation_rate this consistently
        # excludes zero-entity source documents for BOTH methods,
        # because source entity count is shared within each pair.
        if (
            av is None
            or
            bv is None
        ):
            continue

        diff = (
            float(av)
            -
            float(bv)
        )

        instance_diffs.append(
            diff
        )

        by_base[
            a["base_id"]
        ].append(
            diff
        )

    if not instance_diffs:
        raise RuntimeError(
            "No usable paired values"
        )

    base_values = np.asarray(
        [
            np.mean(
                by_base[b]
            )
            for b in sorted(
                by_base
            )
        ],
        dtype=np.float64,
    )

    rng = np.random.default_rng(
        SEED
    )

    draws = rng.integers(
        0,
        len(base_values),
        size=(
            B,
            len(base_values),
        ),
    )

    boot = base_values[
        draws
    ].mean(
        axis=1
    )

    lo, hi = ci(
        boot
    )

    return {
        "n_instances":
            len(
                common
            ),

        "n_metric_pairs":
            len(
                instance_diffs
            ),

        "n_base_ids":
            len(
                base_values
            ),

        "instance_mean_delta":
            float(
                np.mean(
                    instance_diffs
                )
            ),

        "cluster_mean_delta":
            float(
                np.mean(
                    base_values
                )
            ),

        "cluster_ci_low":
            lo,

        "cluster_ci_high":
            hi,

        "bootstrap_prob_delta_gt_0":
            float(
                np.mean(
                    boot > 0
                )
            ),
    }


METRICS = {
    "entity_preservation_rate":
        lambda r:
            r[
                "entity_preservation_rate"
            ],

    "all_entities_preserved_rate":
        lambda r:
            1.0
            if r[
                "all_entities_preserved"
            ]
            else 0.0,
}


# ============================================================================
# 8. Overall + dataset summaries.
# ============================================================================

dataset_rows = []

for dataset_label, domain in DATASETS:

    for condition in CONDITIONS:

        subset = [
            r
            for r in score_rows
            if (
                r["condition"]
                == condition
                and
                (
                    domain is None
                    or
                    r["domain"]
                    == domain
                )
            )
        ]

        for metric, fn in METRICS.items():

            stats = cluster_summary(
                subset,
                fn,
            )

            dataset_rows.append({
                "dataset":
                    dataset_label,

                "method":
                    DISPLAY[
                        condition
                    ],

                "condition":
                    condition,

                "metric":
                    metric,

                **stats,
            })


write_csv(
    BY_DATASET,
    dataset_rows,
)


# ============================================================================
# 9. Paired comparisons overall + dataset.
# ============================================================================

pair_rows = []

for dataset_label, domain in DATASETS:

    scoped = [
        r
        for r in score_rows
        if (
            domain is None
            or
            r["domain"]
            == domain
        )
    ]

    for comparison, lhs, rhs in PAIRINGS:

        lhs_rows = [
            r
            for r in scoped
            if r["condition"]
            == lhs
        ]

        rhs_rows = [
            r
            for r in scoped
            if r["condition"]
            == rhs
        ]

        for metric, fn in METRICS.items():

            stats = paired_summary(
                lhs_rows,
                rhs_rows,
                fn,
            )

            pair_rows.append({
                "dataset":
                    dataset_label,

                "comparison":
                    comparison,

                "lhs":
                    lhs,

                "rhs":
                    rhs,

                "metric":
                    metric,

                **stats,

                "orientation":
                    (
                        "positive = HPCD "
                        "preserves more"
                    ),
            })


write_csv(
    PAIR_DATASET,
    pair_rows,
)


# ============================================================================
# 10. Source-generator summaries.
# ============================================================================

generators = sorted(
    {
        r["source_generator"]
        for r in score_rows
    }
)

generator_rows = []

for generator in generators:

    for condition in CONDITIONS:

        subset = [
            r
            for r in score_rows
            if (
                r["condition"]
                == condition
                and
                r[
                    "source_generator"
                ]
                == generator
            )
        ]

        for metric, fn in METRICS.items():

            stats = cluster_summary(
                subset,
                fn,
            )

            generator_rows.append({
                "source_generator":
                    generator,

                "method":
                    DISPLAY[
                        condition
                    ],

                "condition":
                    condition,

                "metric":
                    metric,

                **stats,
            })


write_csv(
    BY_GENERATOR,
    generator_rows,
)


# ============================================================================
# 11. Dataset × generator diagnostics.
# ============================================================================

dg_rows = []

for dataset_label, domain in DATASETS[1:]:

    for generator in generators:

        for condition in CONDITIONS:

            subset = [
                r
                for r in score_rows
                if (
                    r["condition"]
                    == condition
                    and
                    r["domain"]
                    == domain
                    and
                    r[
                        "source_generator"
                    ]
                    == generator
                )
            ]

            if not subset:
                continue

            for metric, fn in METRICS.items():

                stats = cluster_summary(
                    subset,
                    fn,
                )

                dg_rows.append({
                    "dataset":
                        dataset_label,

                    "source_generator":
                        generator,

                    "method":
                        DISPLAY[
                            condition
                        ],

                    "condition":
                        condition,

                    "metric":
                        metric,

                    **stats,
                })


write_csv(
    BY_DATASET_GENERATOR,
    dg_rows,
)


# ============================================================================
# 12. Terminal report.
# ============================================================================

print()
print("=" * 100)
print("ENTITY PRESERVATION — OVERALL")
print("=" * 100)

for condition in CONDITIONS:

    rate = next(
        r
        for r in dataset_rows
        if (
            r["dataset"]
            == "Overall"
            and
            r["condition"]
            == condition
            and
            r["metric"]
            ==
            "entity_preservation_rate"
        )
    )

    all_ok = next(
        r
        for r in dataset_rows
        if (
            r["dataset"]
            == "Overall"
            and
            r["condition"]
            == condition
            and
            r["metric"]
            ==
            "all_entities_preserved_rate"
        )
    )

    print(
        f"{DISPLAY[condition]:20s} "
        f"entity={rate['mean']:.4f} "
        f"cluster={rate['cluster_mean']:.4f} "
        f"CI=[{rate['cluster_ci_low']:.4f}, "
        f"{rate['cluster_ci_high']:.4f}] "
        f"all-preserved={all_ok['mean']:.4f}"
    )


print()
print("=" * 100)
print("ENTITY PRESERVATION BY DATASET")
print("=" * 100)

for dataset_label, _ in DATASETS[1:]:

    print()
    print(dataset_label)

    for condition in CONDITIONS:

        row = next(
            r
            for r in dataset_rows
            if (
                r["dataset"]
                == dataset_label
                and
                r["condition"]
                == condition
                and
                r["metric"]
                ==
                "entity_preservation_rate"
            )
        )

        print(
            f"  {DISPLAY[condition]:20s} "
            f"mean={row['mean']:.4f} "
            f"CI=[{row['cluster_ci_low']:.4f}, "
            f"{row['cluster_ci_high']:.4f}] "
            f"metric_n={row['n_metric_instances']}"
        )


print()
print("=" * 100)
print("HPCD PAIRWISE ENTITY PRESERVATION")
print("=" * 100)

for dataset_label, _ in DATASETS:

    print()
    print(dataset_label)

    for comparison, _, _ in PAIRINGS:

        row = next(
            r
            for r in pair_rows
            if (
                r["dataset"]
                == dataset_label
                and
                r["comparison"]
                == comparison
                and
                r["metric"]
                ==
                "entity_preservation_rate"
            )
        )

        print(
            f"  {comparison:29s} "
            f"Δ={row['cluster_mean_delta']:+.4f} "
            f"CI=[{row['cluster_ci_low']:+.4f}, "
            f"{row['cluster_ci_high']:+.4f}] "
            f"P(Δ>0)="
            f"{row['bootstrap_prob_delta_gt_0']:.5f}"
        )


# ============================================================================
# 13. Manifest.
# ============================================================================

manifest = {
    "status":
        "PASS",

    "historical_validator_file":
        str(
            HISTORICAL
        ),

    "historical_validator_sha256":
        observed_historical,

    "historical_functions_reused_directly":
        [
            "normalize_surface",
            "source_entities",
            "surface_entity_check",
        ],

    "spacy_model":
        "en_core_web_sm",

    "spacy_version":
        spacy.__version__,

    "protocol":
        (
            "source-only spaCy NER; historical normalization; "
            "normalized source entity surfaces checked directly "
            "against normalized rewrite text; rewrite NER not used"
        ),

    "ignored_spacy_labels":
        [
            "CARDINAL",
            "ORDINAL",
            "PERCENT",
            "MONEY",
            "DATE",
            "TIME",
            "QUANTITY",
        ],

    "zero_entity_rule": {
        "entity_preservation_rate":
            "NA/excluded from rate mean",

        "all_entities_preserved":
            (
                "True, matching historical "
                "vacuous-pass gate behavior"
            ),
    },

    "canonical_input":
        str(
            CANONICAL
        ),

    "canonical_input_sha256":
        sha256_file(
            CANONICAL
        ),

    "conditions":
        CONDITIONS,

    "bootstrap": {
        "unit":
            "base_id",

        "replicates":
            B,

        "seed":
            SEED,
    },

    "source_documents":
        95,

    "source_docs_with_entities":
        int(
            np.sum(
                entity_counts > 0
            )
        ),

    "source_docs_without_entities":
        int(
            np.sum(
                entity_counts == 0
            )
        ),

    "total_source_entity_surfaces":
        int(
            entity_counts.sum()
        ),

    "outputs": {},
}


for p in [
    SCORES,
    BY_DATASET,
    PAIR_DATASET,
    BY_GENERATOR,
    BY_DATASET_GENERATOR,
]:
    manifest[
        "outputs"
    ][
        p.name
    ] = sha256_file(
        p
    )


MANIFEST.write_text(
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
print("FINAL ENTITY ARTIFACTS")
print("=" * 100)

for p in [
    SCORES,
    BY_DATASET,
    PAIR_DATASET,
    BY_GENERATOR,
    BY_DATASET_GENERATOR,
    MANIFEST,
]:
    print(
        sha256_file(p),
        p,
    )


print()
print("=" * 100)
print("FROZEN_ENTITY_PRESERVATION=PASS")
print("=" * 100)
