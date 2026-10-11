import os
#!/usr/bin/env python3

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(os.environ.get("HPCD_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))

IN_DIR = ROOT / "results/hjtm/t1c_joint_cv_v1"

SCORES = (
    IN_DIR /
    "crossfit_document_scores.jsonl"
)

SUMMARY = (
    IN_DIR /
    "joint_cv_summary.json"
)

OUT_DIR = (
    ROOT /
    "results/hjtm/t1d_typicality_v1"
)

EXPECTED_SCORES_SHA = (
    "b6dab4080da6e8f70db4ca4d4b5a7ffc"
    "41cb349f1e24a70d882ccc2cdd3700a6"
)

EXPECTED_SUMMARY_SHA = (
    "dd27ba483e40b1aebb954a82e2d18c9f"
    "01d91f7ef4f87a60ad3bc26a46328856"
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

ORDERS = range(5)

BOOTSTRAPS = 5000
SEED = 43


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def load_jsonl(path):
    rows = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            if line.strip():
                rows.append(
                    json.loads(line)
                )

    return rows


def bootstrap_ci(
    values,
    seed,
):
    values = np.asarray(
        values,
        dtype=np.float64,
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
        "HJTM T1D — CROSS-FITTED "
        "HUMAN TRAJECTORY TYPICALITY"
    )

    print(
        "=" * 100
    )

    if OUT_DIR.exists():
        raise RuntimeError(
            f"Refusing overwrite: {OUT_DIR}"
        )

    OUT_DIR.mkdir(
        parents=True
    )

    score_sha = sha256_file(
        SCORES
    )

    summary_sha = sha256_file(
        SUMMARY
    )

    print(
        "scores SHA256:",
        score_sha,
    )

    print(
        "T1C summary SHA256:",
        summary_sha,
    )

    if (
        score_sha
        !=
        EXPECTED_SCORES_SHA
    ):
        raise RuntimeError(
            "T1C scores hash mismatch"
        )

    if (
        summary_sha
        !=
        EXPECTED_SUMMARY_SHA
    ):
        raise RuntimeError(
            "T1C summary hash mismatch"
        )

    rows = load_jsonl(
        SCORES
    )

    print(
        "rows:",
        len(rows)
    )

    if len(rows) != 1004:
        raise RuntimeError(
            "Expected 1004 rows"
        )

    source_counts = Counter(
        r[
            "source_type"
        ]
        for r in rows
    )

    print(
        "source counts:",
        dict(source_counts),
    )

    human_rows = [
        r
        for r in rows
        if r[
            "source_type"
        ]
        ==
        "human"
    ]

    if len(human_rows) != 251:
        raise RuntimeError(
            "Expected 251 humans"
        )

    row_map = {
        (
            r[
                "source_type"
            ],
            r[
                "sample_id"
            ],
        ):
            r
        for r in rows
    }

    sample_ids = sorted(
        {
            r[
                "sample_id"
            ]
            for r in human_rows
        }
    )

    domain_by_id = {
        r[
            "sample_id"
        ]:
            r[
                "domain"
            ]
        for r in human_rows
    }

    fold_by_id = {
        r[
            "sample_id"
        ]:
            int(
                r[
                    "fold"
                ]
            )
        for r in human_rows
    }

    calibrated_rows = []

    print(
        "\n===== OUT-OF-FOLD HUMAN CALIBRATION ====="
    )

    for order in ORDERS:

        for sid in sample_ids:

            domain = domain_by_id[
                sid
            ]

            fold = fold_by_id[
                sid
            ]

            true_cal = [
                r[
                    f"true_nll_n{order}"
                ]
                for r in human_rows
                if (
                    r[
                        "domain"
                    ]
                    ==
                    domain
                    and
                    int(
                        r[
                            "fold"
                        ]
                    )
                    !=
                    fold
                )
            ]

            null_cal = [
                r[
                    f"null_nll_n{order}"
                ]
                for r in human_rows
                if (
                    r[
                        "domain"
                    ]
                    ==
                    domain
                    and
                    int(
                        r[
                            "fold"
                        ]
                    )
                    !=
                    fold
                )
            ]

            if len(true_cal) < 30:
                raise RuntimeError(
                    f"Too few calibration humans "
                    f"{domain}/{fold}: "
                    f"{len(true_cal)}"
                )

            true_mu = float(
                np.mean(
                    true_cal
                )
            )

            true_sd = float(
                np.std(
                    true_cal,
                    ddof=1,
                )
            )

            null_mu = float(
                np.mean(
                    null_cal
                )
            )

            null_sd = float(
                np.std(
                    null_cal,
                    ddof=1,
                )
            )

            if (
                true_sd <= 0
                or
                null_sd <= 0
            ):
                raise RuntimeError(
                    "Invalid human "
                    "calibration SD"
                )

            for source_type in (
                "human",
                *GENERATORS,
            ):

                r = row_map[
                    (
                        source_type,
                        sid,
                    )
                ]

                true_nll = float(
                    r[
                        f"true_nll_n{order}"
                    ]
                )

                null_nll = float(
                    r[
                        f"null_nll_n{order}"
                    ]
                )

                true_z = (
                    true_nll
                    -
                    true_mu
                ) / true_sd

                null_z = (
                    null_nll
                    -
                    null_mu
                ) / null_sd

                calibrated_rows.append({
                    "sample_id":
                        sid,

                    "domain":
                        domain,

                    "fold":
                        fold,

                    "source_type":
                        source_type,

                    "order":
                        order,

                    "true_nll":
                        true_nll,

                    "human_mu_true":
                        true_mu,

                    "human_sd_true":
                        true_sd,

                    "true_signed_z":
                        float(
                            true_z
                        ),

                    "true_typicality":
                        float(
                            abs(
                                true_z
                            )
                        ),

                    "null_nll":
                        null_nll,

                    "human_mu_null":
                        null_mu,

                    "human_sd_null":
                        null_sd,

                    "null_signed_z":
                        float(
                            null_z
                        ),

                    "null_typicality":
                        float(
                            abs(
                                null_z
                            )
                        ),
                })

    calibrated_map = {
        (
            r[
                "source_type"
            ],
            r[
                "sample_id"
            ],
            r[
                "order"
            ],
        ):
            r
        for r in calibrated_rows
    }

    summary = {
        "definition":
            (
                "T_H(x)=abs("
                "(NLL_H(x)-mu_H,d)"
                "/sigma_H,d)"
            ),

        "calibration":
            (
                "same-domain out-of-fold "
                "human NLLs excluding the "
                "evaluation document's fold"
            ),

        "orders":
            [],
    }

    print(
        "\n===== MACHINE − HUMAN TYPICALITY ====="
    )

    for order in ORDERS:

        print()
        print(
            f"ORDER n={order}"
        )

        generator_results = []

        for gi, generator in enumerate(
            GENERATORS
        ):

            margins = []

            null_margins = []

            signed_machine = []

            signed_human = []

            for sid in sample_ids:

                h = calibrated_map[
                    (
                        "human",
                        sid,
                        order,
                    )
                ]

                m = calibrated_map[
                    (
                        generator,
                        sid,
                        order,
                    )
                ]

                margins.append(
                    m[
                        "true_typicality"
                    ]
                    -
                    h[
                        "true_typicality"
                    ]
                )

                null_margins.append(
                    m[
                        "null_typicality"
                    ]
                    -
                    h[
                        "null_typicality"
                    ]
                )

                signed_machine.append(
                    m[
                        "true_signed_z"
                    ]
                )

                signed_human.append(
                    h[
                        "true_signed_z"
                    ]
                )

            margins = np.asarray(
                margins,
                dtype=np.float64,
            )

            null_margins = np.asarray(
                null_margins,
                dtype=np.float64,
            )

            ci = bootstrap_ci(
                margins,
                SEED
                +
                100 * order
                +
                gi,
            )

            adv = (
                margins
                -
                null_margins
            )

            adv_ci = bootstrap_ci(
                adv,
                SEED
                +
                500
                +
                100 * order
                +
                gi,
            )

            rec = {
                "generator":
                    generator,

                "mean_machine_minus_human_typicality":
                    float(
                        margins.mean()
                    ),

                "ci95_low":
                    ci[
                        0
                    ],

                "ci95_high":
                    ci[
                        1
                    ],

                "positive_fraction":
                    float(
                        np.mean(
                            margins
                            >
                            0
                        )
                    ),

                "mean_machine_signed_z":
                    float(
                        np.mean(
                            signed_machine
                        )
                    ),

                "mean_human_signed_z":
                    float(
                        np.mean(
                            signed_human
                        )
                    ),

                "mean_true_advantage_vs_shuffled":
                    float(
                        adv.mean()
                    ),

                "adv_ci95_low":
                    adv_ci[
                        0
                    ],

                "adv_ci95_high":
                    adv_ci[
                        1
                    ],
            }

            generator_results.append(
                rec
            )

            print(
                generator,
                "typ-margin=",
                round(
                    rec[
                        "mean_machine_minus_human_typicality"
                    ],
                    6,
                ),
                "CI=",
                (
                    round(
                        ci[0],
                        6,
                    ),
                    round(
                        ci[1],
                        6,
                    ),
                ),
                "machine-z=",
                round(
                    rec[
                        "mean_machine_signed_z"
                    ],
                    4,
                ),
                "human-z=",
                round(
                    rec[
                        "mean_human_signed_z"
                    ],
                    4,
                ),
                "adv-vs-shuffle=",
                round(
                    rec[
                        "mean_true_advantage_vs_shuffled"
                    ],
                    6,
                ),
            )

        macro = []

        macro_null = []

        for sid in sample_ids:

            h = calibrated_map[
                (
                    "human",
                    sid,
                    order,
                )
            ]

            per_gen = []

            per_gen_null = []

            for generator in GENERATORS:

                m = calibrated_map[
                    (
                        generator,
                        sid,
                        order,
                    )
                ]

                per_gen.append(
                    m[
                        "true_typicality"
                    ]
                    -
                    h[
                        "true_typicality"
                    ]
                )

                per_gen_null.append(
                    m[
                        "null_typicality"
                    ]
                    -
                    h[
                        "null_typicality"
                    ]
                )

            macro.append(
                np.mean(
                    per_gen
                )
            )

            macro_null.append(
                np.mean(
                    per_gen_null
                )
            )

        macro = np.asarray(
            macro
        )

        macro_null = np.asarray(
            macro_null
        )

        macro_ci = bootstrap_ci(
            macro,
            SEED
            +
            2000
            +
            order,
        )

        macro_adv = (
            macro
            -
            macro_null
        )

        macro_adv_ci = bootstrap_ci(
            macro_adv,
            SEED
            +
            3000
            +
            order,
        )

        cells = []

        positive_cells = 0

        for generator in GENERATORS:

            for domain in DOMAINS:

                values = []

                for sid in sample_ids:

                    if (
                        domain_by_id[
                            sid
                        ]
                        !=
                        domain
                    ):
                        continue

                    h = calibrated_map[
                        (
                            "human",
                            sid,
                            order,
                        )
                    ]

                    m = calibrated_map[
                        (
                            generator,
                            sid,
                            order,
                        )
                    ]

                    values.append(
                        m[
                            "true_typicality"
                        ]
                        -
                        h[
                            "true_typicality"
                        ]
                    )

                mean_value = float(
                    np.mean(
                        values
                    )
                )

                if mean_value > 0:
                    positive_cells += 1

                cells.append({
                    "generator":
                        generator,

                    "domain":
                        domain,

                    "mean_margin":
                        mean_value,

                    "n":
                        len(
                            values
                        ),
                })

        generator_sig = sum(
            x[
                "ci95_low"
            ]
            >
            0
            for x in generator_results
        )

        order_record = {
            "order":
                order,

            "generator_results":
                generator_results,

            "macro_mean_machine_minus_human_typicality":
                float(
                    macro.mean()
                ),

            "macro_ci95_low":
                macro_ci[
                    0
                ],

            "macro_ci95_high":
                macro_ci[
                    1
                ],

            "macro_true_advantage_vs_shuffled":
                float(
                    macro_adv.mean()
                ),

            "macro_adv_ci95_low":
                macro_adv_ci[
                    0
                ],

            "macro_adv_ci95_high":
                macro_adv_ci[
                    1
                ],

            "positive_generator_domain_cells":
                positive_cells,

            "generators_significant_positive":
                generator_sig,

            "cells":
                cells,
        }

        summary[
            "orders"
        ].append(
            order_record
        )

        print(
            "MACRO typ-margin=",
            round(
                order_record[
                    "macro_mean_machine_minus_human_typicality"
                ],
                6,
            ),
            "CI=",
            (
                round(
                    macro_ci[0],
                    6,
                ),
                round(
                    macro_ci[1],
                    6,
                ),
            ),
            "adv-vs-shuffle=",
            round(
                order_record[
                    "macro_true_advantage_vs_shuffled"
                ],
                6,
            ),
            "adv-CI=",
            (
                round(
                    macro_adv_ci[0],
                    6,
                ),
                round(
                    macro_adv_ci[1],
                    6,
                ),
            ),
            "positive cells=",
            positive_cells,
            "/12",
            "sig generators=",
            generator_sig,
            "/3",
        )

    print(
        "\n===== T1 FINAL DIAGNOSTIC ====="
    )

    t1c_summary = json.loads(
        SUMMARY.read_text(
            encoding="utf-8"
        )
    )

    human_shuffle = {
        x[
            "order"
        ]:
            x
        for x in t1c_summary[
            "human_true_vs_shuffled"
        ]
    }

    diagnostics = []

    for order_record in summary[
        "orders"
    ]:

        order = order_record[
            "order"
        ]

        if order == 0:
            continue

        temporal_human = (
            human_shuffle[
                order
            ][
                "ci95_low"
            ]
            >
            0
        )

        typicality_sep = (
            order_record[
                "macro_ci95_low"
            ]
            >
            0
        )

        stable_cells = (
            order_record[
                "positive_generator_domain_cells"
            ]
            >=
            9
        )

        stable_generators = (
            order_record[
                "generators_significant_positive"
            ]
            >=
            2
        )

        diag = {
            "order":
                order,

            "heldout_human_temporal_structure":
                temporal_human,

            "machine_human_typicality_separation":
                typicality_sep,

            "positive_cells_at_least_9_of_12":
                stable_cells,

            "significant_generators_at_least_2_of_3":
                stable_generators,

            "T1_candidate_pass":
                bool(
                    temporal_human
                    and
                    typicality_sep
                    and
                    stable_cells
                    and
                    stable_generators
                ),
        }

        diagnostics.append(
            diag
        )

        print(
            json.dumps(
                diag,
                indent=2,
            )
        )

    passing = [
        x[
            "order"
        ]
        for x in diagnostics
        if x[
            "T1_candidate_pass"
        ]
    ]

    summary[
        "T1_diagnostics"
    ] = diagnostics

    summary[
        "passing_orders"
    ] = passing

    summary[
        "T1_typicality_verdict"
    ] = (
        "PASS"
        if passing
        else
        "FAIL"
    )

    print()
    print(
        "T1 TYPICALITY VERDICT:",
        summary[
            "T1_typicality_verdict"
        ],
    )

    print(
        "passing orders:",
        passing,
    )

    scores_out = (
        OUT_DIR /
        "trajectory_typicality_scores.jsonl"
    )

    with scores_out.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in calibrated_rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                +
                "\n"
            )

    summary_out = (
        OUT_DIR /
        "trajectory_typicality_summary.json"
    )

    summary_out.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        +
        "\n",
        encoding="utf-8",
    )

    print(
        "\n===== OUTPUT HASHES ====="
    )

    print(
        "typicality scores SHA256:",
        sha256_file(
            scores_out
        ),
    )

    print(
        "typicality summary SHA256:",
        sha256_file(
            summary_out
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
        "HJTM T1D TYPICALITY: COMPLETE"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()
