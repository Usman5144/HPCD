#!/usr/bin/env python3
from __future__ import annotations

import os

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(os.environ.get("HPCD_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))

T1B = ROOT / "results/hjtm/t1b_full_mac_v1"

MANIFEST = T1B / "manifest.jsonl"

OUT = ROOT / "results/hjtm/t1c_joint_cv_v1"

EXPECTED_MANIFEST_SHA = (
    "1e2af1f02eb82323cadf7881e4a4645306b7fe6ae137e3185e9e9b0cb66d6121"
)

SOURCE_TYPES = (
    "human",
    "qwen25_3b",
    "phi35_mini",
    "llama32_3b",
)

GENERATORS = (
    "qwen25_3b",
    "phi35_mini",
    "llama32_3b",
)

DOMAINS = (
    "news",
    "reddit",
    "reviews",
    "wiki",
)

B = 6
E = 3
MAX_ORDER = 4

ALPHA = 1.0

N_FOLDS = 5

SEED = 43

BOOTSTRAPS = 5000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def stable_hash(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def load_jsonl(path: Path):
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if line.strip():
                rows.append(
                    json.loads(line)
                )

    return rows


def boundaries(values, bins):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    qs = [
        i / bins
        for i in range(1, bins)
    ]

    return np.quantile(
        values,
        qs,
        method="linear",
    ).astype(
        np.float64
    )


def discretize(values, bounds):
    return np.searchsorted(
        bounds,
        values,
        side="right",
    ).astype(
        np.int16
    )


def make_folds(
    sample_ids,
    domain_by_id,
):
    fold_by_id = {}

    for domain in DOMAINS:

        ids = [
            sid
            for sid in sample_ids
            if domain_by_id[sid] == domain
        ]

        ids = sorted(
            ids,
            key=lambda sid:
                stable_hash(
                    f"HJTM_T1C_FOLD_{SEED}|{sid}"
                ),
        )

        for i, sid in enumerate(ids):
            fold_by_id[sid] = (
                i
                %
                N_FOLDS
            )

    return fold_by_id


def empty_tables():
    return [
        defaultdict(
            lambda: np.zeros(
                B,
                dtype=np.int64,
            )
        )
        for _ in range(
            MAX_ORDER + 1
        )
    ]


def add_sequence(
    tables,
    b_seq,
    e_seq,
):
    history = []

    for b, e in zip(
        b_seq.tolist(),
        e_seq.tolist(),
    ):

        for order in range(
            MAX_ORDER + 1
        ):

            effective = min(
                order,
                len(history),
            )

            hist = tuple(
                history[
                    -effective:
                ]
            ) if effective else ()

            key = (
                int(e),
                hist,
            )

            tables[
                order
            ][
                key
            ][
                int(b)
            ] += 1

        history.append(
            int(b)
        )


def probability(
    tables,
    order,
    history,
    entropy_state,
    target_state,
):
    effective = min(
        order,
        len(history),
    )

    if effective == 0:

        key = (
            int(entropy_state),
            (),
        )

        counts = tables[
            0
        ].get(
            key
        )

        prior = 1.0 / B

        if counts is None:
            return prior

        total = float(
            counts.sum()
        )

        return float(
            (
                float(
                    counts[
                        target_state
                    ]
                )
                +
                ALPHA * prior
            )
            /
            (
                total
                +
                ALPHA
            )
        )

    hist = tuple(
        history[
            -effective:
        ]
    )

    key = (
        int(entropy_state),
        hist,
    )

    lower = probability(
        tables=tables,
        order=effective - 1,
        history=history,
        entropy_state=entropy_state,
        target_state=target_state,
    )

    counts = tables[
        effective
    ].get(
        key
    )

    if counts is None:
        return lower

    total = float(
        counts.sum()
    )

    return float(
        (
            float(
                counts[
                    target_state
                ]
            )
            +
            ALPHA * lower
        )
        /
        (
            total
            +
            ALPHA
        )
    )


def score_sequence(
    tables,
    b_seq,
    e_seq,
    order,
):
    history = []

    total_logp = 0.0

    token_logps = []

    for b, e in zip(
        b_seq.tolist(),
        e_seq.tolist(),
    ):

        p = probability(
            tables=tables,
            order=order,
            history=history,
            entropy_state=int(e),
            target_state=int(b),
        )

        if (
            not math.isfinite(p)
            or
            p <= 0.0
            or
            p > 1.0
        ):
            raise RuntimeError(
                f"Invalid probability {p}"
            )

        lp = math.log(p)

        total_logp += lp

        token_logps.append(
            lp
        )

        history.append(
            int(b)
        )

    n = len(
        token_logps
    )

    if n == 0:
        raise RuntimeError(
            "Empty sequence"
        )

    return {
        "tokens":
            n,

        "log_joint_probability":
            float(
                total_logp
            ),

        "nll_per_token":
            float(
                -total_logp
                /
                n
            ),
    }


def bootstrap_ci(
    values,
    seed,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if len(values) == 0:
        return (
            float("nan"),
            float("nan"),
        )

    rng = np.random.default_rng(
        seed
    )

    idx = rng.integers(
        0,
        len(values),
        size=(
            BOOTSTRAPS,
            len(values),
        ),
    )

    means = values[
        idx
    ].mean(
        axis=1
    )

    return (
        float(
            np.percentile(
                means,
                2.5,
            )
        ),
        float(
            np.percentile(
                means,
                97.5,
            )
        ),
    )


def main():

    print(
        "=" * 100
    )

    print(
        "HJTM T1C — CROSS-FITTED "
        "CHAIN-RULE JOINT TRAJECTORY MODEL"
    )

    print(
        "=" * 100
    )

    if OUT.exists():
        raise RuntimeError(
            f"Refusing overwrite: {OUT}"
        )

    OUT.mkdir(
        parents=True,
        exist_ok=False,
    )

    manifest_sha = sha256_file(
        MANIFEST
    )

    print(
        "T1B manifest SHA256:",
        manifest_sha,
    )

    if (
        manifest_sha
        !=
        EXPECTED_MANIFEST_SHA
    ):
        raise RuntimeError(
            "T1B manifest checksum mismatch"
        )

    manifest = load_jsonl(
        MANIFEST
    )

    print(
        "documents:",
        len(
            manifest
        ),
    )

    if len(
        manifest
    ) != 1004:
        raise RuntimeError(
            "Expected 1004 documents"
        )

    counts = Counter(
        row[
            "source_type"
        ]
        for row in manifest
    )

    print(
        "source counts:",
        dict(
            counts
        ),
    )

    if any(
        counts[
            s
        ]
        !=
        251
        for s in SOURCE_TYPES
    ):
        raise RuntimeError(
            "Expected 251/source"
        )

    domain_counts = Counter(
        row[
            "domain"
        ]
        for row in manifest
    )

    print(
        "domain counts:",
        dict(
            domain_counts
        ),
    )

    metadata = {
        (
            row[
                "source_type"
            ],
            row[
                "sample_id"
            ],
        ):
            row
        for row in manifest
    }

    sample_ids = sorted(
        {
            row[
                "sample_id"
            ]
            for row in manifest
        }
    )

    if len(
        sample_ids
    ) != 251:
        raise RuntimeError(
            "Expected 251 unique IDs"
        )

    domain_by_id = {}

    for sid in sample_ids:

        domains = {
            metadata[
                (
                    s,
                    sid,
                )
            ][
                "domain"
            ]
            for s
            in SOURCE_TYPES
        }

        if len(domains) != 1:
            raise RuntimeError(
                f"Domain mismatch {sid}"
            )

        domain_by_id[
            sid
        ] = next(
            iter(
                domains
            )
        )

    fold_by_id = make_folds(
        sample_ids,
        domain_by_id,
    )

    fold_counts = Counter(
        fold_by_id.values()
    )

    print(
        "fold counts:",
        dict(
            fold_counts
        ),
    )

    print(
        "\n===== LOAD CONTINUOUS TRAJECTORIES ====="
    )

    trajectories = {}

    total_tokens = 0

    for i, row in enumerate(
        manifest,
        1,
    ):

        path = ROOT / row[
            "trajectory_file"
        ]

        if (
            sha256_file(
                path
            )
            !=
            row[
                "trajectory_sha256"
            ]
        ):
            raise RuntimeError(
                f"Trajectory checksum mismatch: "
                f"{path}"
            )

        with np.load(
            path
        ) as z:

            u = z[
                "probability_mass_position"
            ].astype(
                np.float64
            )

            h = z[
                "context_entropy"
            ].astype(
                np.float64
            )

        if len(u) != len(h):
            raise RuntimeError(
                "Trajectory length mismatch"
            )

        trajectories[
            (
                row[
                    "source_type"
                ],
                row[
                    "sample_id"
                ],
            )
        ] = {
            "u": u,
            "H": h,
        }

        total_tokens += len(
            u
        )

        if (
            i % 200 == 0
            or
            i == len(
                manifest
            )
        ):
            print(
                f"loaded {i}/"
                f"{len(manifest)}",
                flush=True,
            )

    print(
        "total scored tokens:",
        total_tokens,
    )

    if total_tokens != 342589:
        raise RuntimeError(
            "Unexpected total token count"
        )

    print(
        "\n===== CROSS-FITTED HJTM ====="
    )

    result_rows = []

    boundary_rows = []

    for fold in range(
        N_FOLDS
    ):

        print()
        print(
            f"--- fold {fold} ---"
        )

        train_ids = [
            sid
            for sid
            in sample_ids
            if (
                fold_by_id[
                    sid
                ]
                !=
                fold
            )
        ]

        test_ids = [
            sid
            for sid
            in sample_ids
            if (
                fold_by_id[
                    sid
                ]
                ==
                fold
            )
        ]

        print(
            "train humans:",
            len(
                train_ids
            ),
        )

        print(
            "held-out IDs:",
            len(
                test_ids
            ),
        )

        human_u = np.concatenate(
            [
                trajectories[
                    (
                        "human",
                        sid,
                    )
                ][
                    "u"
                ]
                for sid
                in train_ids
            ]
        )

        human_H = np.concatenate(
            [
                trajectories[
                    (
                        "human",
                        sid,
                    )
                ][
                    "H"
                ]
                for sid
                in train_ids
            ]
        )

        mass_bounds = boundaries(
            human_u,
            B,
        )

        entropy_bounds = boundaries(
            human_H,
            E,
        )

        print(
            "mass boundaries:",
            mass_bounds.tolist(),
        )

        print(
            "entropy boundaries:",
            entropy_bounds.tolist(),
        )

        boundary_rows.append({
            "fold":
                fold,

            "train_human_documents":
                len(
                    train_ids
                ),

            "train_human_tokens":
                int(
                    len(
                        human_u
                    )
                ),

            "mass_boundaries":
                mass_bounds.tolist(),

            "entropy_boundaries":
                entropy_bounds.tolist(),
        })

        true_tables = (
            empty_tables()
        )

        null_tables = (
            empty_tables()
        )

        for sid in train_ids:

            traj = trajectories[
                (
                    "human",
                    sid,
                )
            ]

            b_seq = discretize(
                traj[
                    "u"
                ],
                mass_bounds,
            )

            e_seq = discretize(
                traj[
                    "H"
                ],
                entropy_bounds,
            )

            add_sequence(
                true_tables,
                b_seq,
                e_seq,
            )

            seed_int = int(
                stable_hash(
                    f"HJTM_NULL_{SEED}|"
                    f"{fold}|{sid}"
                )[
                    :16
                ],
                16,
            )

            rng = np.random.default_rng(
                seed_int
            )

            perm = rng.permutation(
                len(
                    b_seq
                )
            )

            b_null = b_seq[
                perm
            ]

            e_null = e_seq[
                perm
            ]

            add_sequence(
                null_tables,
                b_null,
                e_null,
            )

        for sid in test_ids:

            for source_type in SOURCE_TYPES:

                traj = trajectories[
                    (
                        source_type,
                        sid,
                    )
                ]

                b_seq = discretize(
                    traj[
                        "u"
                    ],
                    mass_bounds,
                )

                e_seq = discretize(
                    traj[
                        "H"
                    ],
                    entropy_bounds,
                )

                row = {
                    "sample_id":
                        sid,

                    "domain":
                        domain_by_id[
                            sid
                        ],

                    "fold":
                        fold,

                    "source_type":
                        source_type,

                    "tokens":
                        int(
                            len(
                                b_seq
                            )
                        ),
                }

                for order in range(
                    MAX_ORDER + 1
                ):

                    true_score = (
                        score_sequence(
                            true_tables,
                            b_seq,
                            e_seq,
                            order,
                        )
                    )

                    null_score = (
                        score_sequence(
                            null_tables,
                            b_seq,
                            e_seq,
                            order,
                        )
                    )

                    row[
                        f"true_logP_n{order}"
                    ] = (
                        true_score[
                            "log_joint_probability"
                        ]
                    )

                    row[
                        f"true_nll_n{order}"
                    ] = (
                        true_score[
                            "nll_per_token"
                        ]
                    )

                    row[
                        f"null_nll_n{order}"
                    ] = (
                        null_score[
                            "nll_per_token"
                        ]
                    )

                result_rows.append(
                    row
                )

    print(
        "\n===== REGRESSION CHECKS ====="
    )

    if len(
        result_rows
    ) != 1004:
        raise RuntimeError(
            "Expected 1004 cross-fitted rows"
        )

    max_n0_diff = max(
        abs(
            row[
                "true_nll_n0"
            ]
            -
            row[
                "null_nll_n0"
            ]
        )
        for row in result_rows
    )

    print(
        "max true-vs-null n=0 "
        "difference:",
        max_n0_diff,
    )

    if max_n0_diff > 1e-12:
        raise RuntimeError(
            "Shuffling changed n=0 "
            "occupancy model"
        )

    row_map = {
        (
            row[
                "source_type"
            ],
            row[
                "sample_id"
            ],
        ):
            row
        for row in result_rows
    }

    summary = {
        "configuration": {
            "B_mass":
                B,

            "E_entropy":
                E,

            "orders":
                list(
                    range(
                        MAX_ORDER + 1
                    )
                ),

            "alpha":
                ALPHA,

            "folds":
                N_FOLDS,

            "bootstrap":
                BOOTSTRAPS,

            "seed":
                SEED,

            "model":
                (
                    "global human-only "
                    "chain-rule trajectory "
                    "model conditioned on "
                    "current entropy state"
                ),

            "joint_definition":
                (
                    "P(b_1:T | e_1:T) = "
                    "product_t "
                    "P(b_t | b_<t truncated "
                    "to n-history, e_t)"
                ),

            "null":
                (
                    "within-document shuffle "
                    "of paired (b_t,e_t) "
                    "states in human training "
                    "documents"
                ),
        },

        "generator_order_results":
            [],

        "macro_order_results":
            [],

        "cell_results":
            [],
    }

    print(
        "\n===== HUMAN ↔ MACHINE PAIRED RESULTS ====="
    )

    for order in range(
        MAX_ORDER + 1
    ):

        print()
        print(
            f"ORDER n={order}"
        )

        generator_margins = {}

        for gi, generator in enumerate(
            GENERATORS
        ):

            margins_true = []

            margins_null = []

            gains_vs_n0 = []

            margin_adv_null = []

            pair_ids = []

            for sid in sample_ids:

                human_row = row_map[
                    (
                        "human",
                        sid,
                    )
                ]

                machine_row = row_map[
                    (
                        generator,
                        sid,
                    )
                ]

                true_margin = (
                    machine_row[
                        f"true_nll_n{order}"
                    ]
                    -
                    human_row[
                        f"true_nll_n{order}"
                    ]
                )

                null_margin = (
                    machine_row[
                        f"null_nll_n{order}"
                    ]
                    -
                    human_row[
                        f"null_nll_n{order}"
                    ]
                )

                baseline_margin = (
                    machine_row[
                        "true_nll_n0"
                    ]
                    -
                    human_row[
                        "true_nll_n0"
                    ]
                )

                margins_true.append(
                    true_margin
                )

                margins_null.append(
                    null_margin
                )

                gains_vs_n0.append(
                    true_margin
                    -
                    baseline_margin
                )

                margin_adv_null.append(
                    true_margin
                    -
                    null_margin
                )

                pair_ids.append(
                    sid
                )

            margins_true = np.asarray(
                margins_true,
                dtype=np.float64,
            )

            margins_null = np.asarray(
                margins_null,
                dtype=np.float64,
            )

            gains_vs_n0 = np.asarray(
                gains_vs_n0,
                dtype=np.float64,
            )

            margin_adv_null = np.asarray(
                margin_adv_null,
                dtype=np.float64,
            )

            margin_ci = bootstrap_ci(
                margins_true,
                SEED
                +
                100
                +
                10 * order
                +
                gi,
            )

            gain_ci = bootstrap_ci(
                gains_vs_n0,
                SEED
                +
                200
                +
                10 * order
                +
                gi,
            )

            null_adv_ci = bootstrap_ci(
                margin_adv_null,
                SEED
                +
                300
                +
                10 * order
                +
                gi,
            )

            generator_margins[
                generator
            ] = {
                sid:
                    float(
                        margins_true[
                            i
                        ]
                    )
                for i, sid
                in enumerate(
                    pair_ids
                )
            }

            record = {
                "generator":
                    generator,

                "order":
                    order,

                "n_pairs":
                    len(
                        margins_true
                    ),

                "mean_machine_minus_human_nll":
                    float(
                        margins_true.mean()
                    ),

                "margin_ci95_low":
                    margin_ci[
                        0
                    ],

                "margin_ci95_high":
                    margin_ci[
                        1
                    ],

                "positive_margin_fraction":
                    float(
                        np.mean(
                            margins_true
                            >
                            0
                        )
                    ),

                "mean_gain_vs_n0":
                    float(
                        gains_vs_n0.mean()
                    ),

                "gain_vs_n0_ci95_low":
                    gain_ci[
                        0
                    ],

                "gain_vs_n0_ci95_high":
                    gain_ci[
                        1
                    ],

                "mean_margin_advantage_vs_shuffled":
                    float(
                        margin_adv_null.mean()
                    ),

                "margin_adv_shuffled_ci95_low":
                    null_adv_ci[
                        0
                    ],

                "margin_adv_shuffled_ci95_high":
                    null_adv_ci[
                        1
                    ],
            }

            summary[
                "generator_order_results"
            ].append(
                record
            )

            print(
                generator,
                "margin=",
                round(
                    record[
                        "mean_machine_minus_human_nll"
                    ],
                    6,
                ),
                "CI=",
                (
                    round(
                        margin_ci[
                            0
                        ],
                        6,
                    ),
                    round(
                        margin_ci[
                            1
                        ],
                        6,
                    ),
                ),
                "gain_vs_n0=",
                round(
                    record[
                        "mean_gain_vs_n0"
                    ],
                    6,
                ),
                "gain_CI=",
                (
                    round(
                        gain_ci[
                            0
                        ],
                        6,
                    ),
                    round(
                        gain_ci[
                            1
                        ],
                        6,
                    ),
                ),
                "adv_vs_shuffle=",
                round(
                    record[
                        "mean_margin_advantage_vs_shuffled"
                    ],
                    6,
                ),
                flush=True,
            )

        macro_values = []

        macro_gain_values = []

        macro_null_adv_values = []

        for sid in sample_ids:

            per_gen_margin = []

            per_gen_gain = []

            per_gen_null_adv = []

            human_row = row_map[
                (
                    "human",
                    sid,
                )
            ]

            for generator in GENERATORS:

                machine_row = row_map[
                    (
                        generator,
                        sid,
                    )
                ]

                true_margin = (
                    machine_row[
                        f"true_nll_n{order}"
                    ]
                    -
                    human_row[
                        f"true_nll_n{order}"
                    ]
                )

                n0_margin = (
                    machine_row[
                        "true_nll_n0"
                    ]
                    -
                    human_row[
                        "true_nll_n0"
                    ]
                )

                null_margin = (
                    machine_row[
                        f"null_nll_n{order}"
                    ]
                    -
                    human_row[
                        f"null_nll_n{order}"
                    ]
                )

                per_gen_margin.append(
                    true_margin
                )

                per_gen_gain.append(
                    true_margin
                    -
                    n0_margin
                )

                per_gen_null_adv.append(
                    true_margin
                    -
                    null_margin
                )

            macro_values.append(
                np.mean(
                    per_gen_margin
                )
            )

            macro_gain_values.append(
                np.mean(
                    per_gen_gain
                )
            )

            macro_null_adv_values.append(
                np.mean(
                    per_gen_null_adv
                )
            )

        macro_values = np.asarray(
            macro_values
        )

        macro_gain_values = np.asarray(
            macro_gain_values
        )

        macro_null_adv_values = np.asarray(
            macro_null_adv_values
        )

        macro_ci = bootstrap_ci(
            macro_values,
            SEED
            +
            1000
            +
            order,
        )

        macro_gain_ci = bootstrap_ci(
            macro_gain_values,
            SEED
            +
            1100
            +
            order,
        )

        macro_null_ci = bootstrap_ci(
            macro_null_adv_values,
            SEED
            +
            1200
            +
            order,
        )

        macro_record = {
            "order":
                order,

            "n_sample_ids":
                251,

            "macro_mean_machine_minus_human_nll":
                float(
                    macro_values.mean()
                ),

            "macro_margin_ci95_low":
                macro_ci[
                    0
                ],

            "macro_margin_ci95_high":
                macro_ci[
                    1
                ],

            "macro_mean_gain_vs_n0":
                float(
                    macro_gain_values.mean()
                ),

            "macro_gain_vs_n0_ci95_low":
                macro_gain_ci[
                    0
                ],

            "macro_gain_vs_n0_ci95_high":
                macro_gain_ci[
                    1
                ],

            "macro_mean_margin_advantage_vs_shuffled":
                float(
                    macro_null_adv_values.mean()
                ),

            "macro_margin_adv_shuffled_ci95_low":
                macro_null_ci[
                    0
                ],

            "macro_margin_adv_shuffled_ci95_high":
                macro_null_ci[
                    1
                ],
        }

        summary[
            "macro_order_results"
        ].append(
            macro_record
        )

        print(
            "MACRO",
            "margin=",
            round(
                macro_record[
                    "macro_mean_machine_minus_human_nll"
                ],
                6,
            ),
            "CI=",
            (
                round(
                    macro_ci[
                        0
                    ],
                    6,
                ),
                round(
                    macro_ci[
                        1
                    ],
                    6,
                ),
            ),
            "gain_vs_n0=",
            round(
                macro_record[
                    "macro_mean_gain_vs_n0"
                ],
                6,
            ),
            "gain_CI=",
            (
                round(
                    macro_gain_ci[
                        0
                    ],
                    6,
                ),
                round(
                    macro_gain_ci[
                        1
                    ],
                    6,
                ),
            ),
            "adv_vs_shuffle=",
            round(
                macro_record[
                    "macro_mean_margin_advantage_vs_shuffled"
                ],
                6,
            ),
        )

        for generator in GENERATORS:

            for domain in DOMAINS:

                ids = [
                    sid
                    for sid
                    in sample_ids
                    if (
                        domain_by_id[
                            sid
                        ]
                        ==
                        domain
                    )
                ]

                cell = []

                for sid in ids:

                    hrow = row_map[
                        (
                            "human",
                            sid,
                        )
                    ]

                    mrow = row_map[
                        (
                            generator,
                            sid,
                        )
                    ]

                    cell.append(
                        mrow[
                            f"true_nll_n{order}"
                        ]
                        -
                        hrow[
                            f"true_nll_n{order}"
                        ]
                    )

                summary[
                    "cell_results"
                ].append({
                    "generator":
                        generator,

                    "domain":
                        domain,

                    "order":
                        order,

                    "n":
                        len(
                            cell
                        ),

                    "mean_machine_minus_human_nll":
                        float(
                            np.mean(
                                cell
                            )
                        ),

                    "positive_fraction":
                        float(
                            np.mean(
                                np.asarray(
                                    cell
                                )
                                >
                                0
                            )
                        ),
                })

    print(
        "\n===== TRUE HUMAN MODEL VS SHUFFLED NULL ON HELD-OUT HUMANS ====="
    )

    human_null_results = []

    for order in range(
        1,
        MAX_ORDER + 1
    ):

        values = []

        for sid in sample_ids:

            row = row_map[
                (
                    "human",
                    sid,
                )
            ]

            values.append(
                row[
                    f"null_nll_n{order}"
                ]
                -
                row[
                    f"true_nll_n{order}"
                ]
            )

        values = np.asarray(
            values,
            dtype=np.float64,
        )

        ci = bootstrap_ci(
            values,
            SEED
            +
            2000
            +
            order,
        )

        rec = {
            "order":
                order,

            "mean_null_minus_true_human_nll":
                float(
                    values.mean()
                ),

            "ci95_low":
                ci[
                    0
                ],

            "ci95_high":
                ci[
                    1
                ],

            "true_better_fraction":
                float(
                    np.mean(
                        values
                        >
                        0
                    )
                ),
        }

        human_null_results.append(
            rec
        )

        print(
            "n=",
            order,
            "null-minus-true=",
            round(
                rec[
                    "mean_null_minus_true_human_nll"
                ],
                6,
            ),
            "CI=",
            (
                round(
                    ci[
                        0
                    ],
                    6,
                ),
                round(
                    ci[
                        1
                    ],
                    6,
                ),
            ),
            "true-better-frac=",
            round(
                rec[
                    "true_better_fraction"
                ],
                4,
            ),
        )

    summary[
        "human_true_vs_shuffled"
    ] = human_null_results

    print(
        "\n===== PREDECLARED CORE JOINT-STRUCTURE DIAGNOSTICS ====="
    )

    diagnostics = []

    for order in range(
        1,
        MAX_ORDER + 1
    ):

        macro = next(
            x
            for x in summary[
                "macro_order_results"
            ]
            if x[
                "order"
            ]
            ==
            order
        )

        human_null = next(
            x
            for x in human_null_results
            if x[
                "order"
            ]
            ==
            order
        )

        generator_records = [
            x
            for x in summary[
                "generator_order_results"
            ]
            if x[
                "order"
            ]
            ==
            order
        ]

        cells = [
            x
            for x in summary[
                "cell_results"
            ]
            if x[
                "order"
            ]
            ==
            order
        ]

        positive_cells = sum(
            x[
                "mean_machine_minus_human_nll"
            ]
            >
            0
            for x in cells
        )

        generators_gain_ci_positive = sum(
            x[
                "gain_vs_n0_ci95_low"
            ]
            >
            0
            for x in generator_records
        )

        diag = {
            "order":
                order,

            "macro_margin_positive_ci":
                (
                    macro[
                        "macro_margin_ci95_low"
                    ]
                    >
                    0
                ),

            "macro_gain_over_n0_positive_ci":
                (
                    macro[
                        "macro_gain_vs_n0_ci95_low"
                    ]
                    >
                    0
                ),

            "human_true_beats_shuffled_ci":
                (
                    human_null[
                        "ci95_low"
                    ]
                    >
                    0
                ),

            "macro_margin_beats_shuffled_ci":
                (
                    macro[
                        "macro_margin_adv_shuffled_ci95_low"
                    ]
                    >
                    0
                ),

            "positive_generator_domain_cells":
                int(
                    positive_cells
                ),

            "generators_with_positive_gain_ci":
                int(
                    generators_gain_ci_positive
                ),
        }

        diag[
            "strong_joint_structure_pass"
        ] = bool(
            diag[
                "macro_margin_positive_ci"
            ]
            and
            diag[
                "macro_gain_over_n0_positive_ci"
            ]
            and
            diag[
                "human_true_beats_shuffled_ci"
            ]
            and
            diag[
                "macro_margin_beats_shuffled_ci"
            ]
            and
            diag[
                "positive_generator_domain_cells"
            ]
            >=
            9
            and
            diag[
                "generators_with_positive_gain_ci"
            ]
            >=
            2
        )

        diagnostics.append(
            diag
        )

        print(
            json.dumps(
                diag,
                indent=2,
            )
        )

    summary[
        "joint_structure_diagnostics"
    ] = diagnostics

    strong_orders = [
        x[
            "order"
        ]
        for x in diagnostics
        if x[
            "strong_joint_structure_pass"
        ]
    ]

    summary[
        "core_joint_structure_verdict"
    ] = (
        "PASS"
        if strong_orders
        else
        "NO_STRONG_PASS"
    )

    summary[
        "passing_orders"
    ] = strong_orders

    print(
        "\nCORE JOINT-STRUCTURE VERDICT:",
        summary[
            "core_joint_structure_verdict"
        ],
    )

    print(
        "passing orders:",
        strong_orders,
    )

    score_path = (
        OUT
        /
        "crossfit_document_scores.jsonl"
    )

    with score_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in result_rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                +
                "\n"
            )

    boundaries_path = (
        OUT
        /
        "fold_boundaries.json"
    )

    boundaries_path.write_text(
        json.dumps(
            boundary_rows,
            indent=2,
        )
        +
        "\n",
        encoding="utf-8",
    )

    summary_path = (
        OUT
        /
        "joint_cv_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        +
        "\n",
        encoding="utf-8",
    )

    cell_path = (
        OUT
        /
        "cell_summary.csv"
    )

    with cell_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "generator",
                "domain",
                "order",
                "n",
                "mean_machine_minus_human_nll",
                "positive_fraction",
            ],
        )

        writer.writeheader()

        writer.writerows(
            summary[
                "cell_results"
            ]
        )

    print(
        "\n===== OUTPUT HASHES ====="
    )

    print(
        "scores SHA256:",
        sha256_file(
            score_path
        ),
    )

    print(
        "boundaries SHA256:",
        sha256_file(
            boundaries_path
        ),
    )

    print(
        "summary SHA256:",
        sha256_file(
            summary_path
        ),
    )

    print(
        "cells SHA256:",
        sha256_file(
            cell_path
        ),
    )

    print(
        "script SHA256:",
        sha256_file(
            Path(
                __file__
            ).resolve()
        ),
    )

    print()
    print(
        "=" * 100
    )

    print(
        "HJTM T1C CHAIN-RULE "
        "JOINT MODEL: COMPLETE"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()
