#!/usr/bin/env python3
from __future__ import annotations

import os

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(os.environ.get("HPCD_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))

T1B = ROOT / "results/hjtm/t1b_independent_qwen15b_v1"
MANIFEST = T1B / "manifest.jsonl"

OUT = ROOT / "results/hjtm/independent_qwen15b_hjtm_v1"

EXPECTED_MANIFEST_SHA = "94f21f24b5ba80e278260c0cef54639c66c358017178ab452f5b7de7f797a5e6"

B = 6
E = 3

ORDERS = (0, 1, 2)

ALPHA = 1.0

SEED = 43

DOMAINS = (
    "news",
    "reddit",
    "reviews",
    "wiki",
)


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


def quantile_boundaries(
    values,
    bins,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return np.quantile(
        values,
        [
            i / bins
            for i in range(
                1,
                bins,
            )
        ],
        method="linear",
    ).astype(
        np.float64
    )


def discretize(
    values,
    bounds,
):
    return np.searchsorted(
        bounds,
        values,
        side="right",
    ).astype(
        np.int16
    )


def make_tables():
    return [
        defaultdict(
            lambda: np.zeros(
                B,
                dtype=np.int64,
            )
        )
        for _ in range(
            max(ORDERS) + 1
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

        for order in ORDERS:

            effective = min(
                order,
                len(history),
            )

            if effective:
                hist = tuple(
                    history[
                        -effective:
                    ]
                )
            else:
                hist = ()

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


def conditional_probability(
    tables,
    order,
    history,
    entropy_state,
    target_state,
):
    effective = min(
        int(order),
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

    key = (
        int(entropy_state),
        tuple(
            history[
                -effective:
            ]
        ),
    )

    lower = conditional_probability(
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

    costs = []

    for b, e in zip(
        b_seq.tolist(),
        e_seq.tolist(),
    ):

        p = conditional_probability(
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
                f"Invalid conditional "
                f"probability {p}"
            )

        costs.append(
            -math.log(p)
        )

        history.append(
            int(b)
        )

    if not costs:
        raise RuntimeError(
            "Cannot score empty trajectory"
        )

    return float(
        np.mean(
            costs
        )
    )


def table_to_json(
    table,
):
    output = {}

    for (
        entropy_state,
        history,
    ), counts in sorted(
        table.items(),
        key=lambda x: (
            x[0][0],
            x[0][1],
        ),
    ):

        history_string = ",".join(
            str(x)
            for x in history
        )

        key = (
            f"{entropy_state}"
            f"|"
            f"{history_string}"
        )

        output[
            key
        ] = [
            int(x)
            for x in counts.tolist()
        ]

    return output


def summarize_nll(
    values,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if len(values) < 2:
        raise RuntimeError(
            "Need >=2 human NLLs"
        )

    return {
        "n":
            int(
                len(values)
            ),

        "mean":
            float(
                np.mean(
                    values
                )
            ),

        "sd":
            float(
                np.std(
                    values,
                    ddof=1,
                )
            ),

        "median":
            float(
                np.median(
                    values
                )
            ),

        "q05":
            float(
                np.quantile(
                    values,
                    0.05,
                )
            ),

        "q25":
            float(
                np.quantile(
                    values,
                    0.25,
                )
            ),

        "q75":
            float(
                np.quantile(
                    values,
                    0.75,
                )
            ),

        "q95":
            float(
                np.quantile(
                    values,
                    0.95,
                )
            ),
    }


def main():

    print(
        "=" * 100
    )

    print(
        "HJTM FINAL DECODER MODEL V1"
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
            "T1B manifest checksum "
            "mismatch"
        )

    manifest = load_jsonl(
        MANIFEST
    )

    human_rows = [
        r
        for r in manifest
        if (
            r[
                "source_type"
            ]
            ==
            "human"
        )
    ]

    print(
        "human documents:",
        len(
            human_rows
        ),
    )

    if len(
        human_rows
    ) != 251:
        raise RuntimeError(
            "Expected exactly "
            "251 human trajectories"
        )

    domain_counts = Counter(
        r[
            "domain"
        ]
        for r in human_rows
    )

    print(
        "human domains:",
        dict(
            domain_counts
        ),
    )

    trajectories = {}

    all_u = []
    all_H = []

    total_human_tokens = 0

    print(
        "\n===== LOAD HUMAN TRAJECTORIES ====="
    )

    for i, row in enumerate(
        human_rows,
        1,
    ):

        path = (
            ROOT
            /
            row[
                "trajectory_file"
            ]
        )

        observed_sha = sha256_file(
            path
        )

        if (
            observed_sha
            !=
            row[
                "trajectory_sha256"
            ]
        ):
            raise RuntimeError(
                f"Trajectory checksum "
                f"mismatch: {path}"
            )

        with np.load(
            path
        ) as z:

            u = z[
                "probability_mass_position"
            ].astype(
                np.float64
            )

            H = z[
                "context_entropy"
            ].astype(
                np.float64
            )

        if len(u) != len(H):
            raise RuntimeError(
                "Trajectory length mismatch"
            )

        sid = row[
            "sample_id"
        ]

        trajectories[
            sid
        ] = {
            "domain":
                row[
                    "domain"
                ],

            "u":
                u,

            "H":
                H,
        }

        all_u.append(
            u
        )

        all_H.append(
            H
        )

        total_human_tokens += len(
            u
        )

        if (
            i % 50 == 0
            or
            i == len(
                human_rows
            )
        ):
            print(
                f"loaded {i}/"
                f"{len(human_rows)}"
            )

    all_u = np.concatenate(
        all_u
    )

    all_H = np.concatenate(
        all_H
    )

    print(
        "human scored tokens:",
        total_human_tokens,
    )

    print(
        "\n===== FINAL HUMAN-ONLY BOUNDARIES ====="
    )

    mass_bounds = quantile_boundaries(
        all_u,
        B,
    )

    entropy_bounds = quantile_boundaries(
        all_H,
        E,
    )

    print(
        "B=6 mass boundaries:",
        mass_bounds.tolist(),
    )

    print(
        "E=3 entropy boundaries:",
        entropy_bounds.tolist(),
    )

    state_sequences = {}

    for sid, traj in trajectories.items():

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

        if not np.all(
            (
                b_seq >= 0
            )
            &
            (
                b_seq < B
            )
        ):
            raise RuntimeError(
                "Invalid mass state"
            )

        if not np.all(
            (
                e_seq >= 0
            )
            &
            (
                e_seq < E
            )
        ):
            raise RuntimeError(
                "Invalid entropy state"
            )

        state_sequences[
            sid
        ] = {
            "domain":
                traj[
                    "domain"
                ],

            "b":
                b_seq,

            "e":
                e_seq,
        }

    print(
        "\n===== FIT TRUE HUMAN MODEL ====="
    )

    true_tables = make_tables()

    for sid, seq in state_sequences.items():
        add_sequence(
            true_tables,
            seq[
                "b"
            ],
            seq[
                "e"
            ],
        )

    for order in ORDERS:
        print(
            f"n={order} contexts:",
            len(
                true_tables[
                    order
                ]
            ),
        )

    print(
        "\n===== FIT SHUFFLED-HUMAN NULL MODEL ====="
    )

    null_tables = make_tables()

    for sid, seq in state_sequences.items():

        seed_int = int(
            stable_hash(
                f"HJTM_FINAL_NULL_"
                f"{SEED}|{sid}"
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
                seq[
                    "b"
                ]
            )
        )

        add_sequence(
            null_tables,
            seq[
                "b"
            ][
                perm
            ],
            seq[
                "e"
            ][
                perm
            ],
        )

    for order in ORDERS:
        print(
            f"null n={order} contexts:",
            len(
                null_tables[
                    order
                ]
            ),
        )

    print(
        "\n===== HUMAN NLL CALIBRATION ====="
    )

    human_scores = []

    for sid, seq in state_sequences.items():

        row = {
            "sample_id":
                sid,

            "domain":
                seq[
                    "domain"
                ],
        }

        for order in ORDERS:

            row[
                f"true_nll_n{order}"
            ] = score_sequence(
                true_tables,
                seq[
                    "b"
                ],
                seq[
                    "e"
                ],
                order,
            )

            row[
                f"null_nll_n{order}"
            ] = score_sequence(
                null_tables,
                seq[
                    "b"
                ],
                seq[
                    "e"
                ],
                order,
            )

        human_scores.append(
            row
        )

    calibration = {
        "global": {},
        "by_domain": {},
    }

    for order in ORDERS:

        global_values = [
            r[
                f"true_nll_n{order}"
            ]
            for r in human_scores
        ]

        calibration[
            "global"
        ][
            str(order)
        ] = summarize_nll(
            global_values
        )

        print(
            f"n={order} global:",
            calibration[
                "global"
            ][
                str(order)
            ],
        )

    for domain in DOMAINS:

        calibration[
            "by_domain"
        ][
            domain
        ] = {}

        subset = [
            r
            for r in human_scores
            if (
                r[
                    "domain"
                ]
                ==
                domain
            )
        ]

        for order in ORDERS:

            values = [
                r[
                    f"true_nll_n{order}"
                ]
                for r in subset
            ]

            calibration[
                "by_domain"
            ][
                domain
            ][
                str(order)
            ] = summarize_nll(
                values
            )

    print(
        "\n===== BUILD SERIALIZED MODEL ====="
    )

    model_object = {
        "model_name":
            "HJTM_INDEPENDENT_QWEN15B_V1",

        "purpose":
            (
                "human-only probability-state "
                "process model for HPCD "
                "decoder development"
            ),

        "training_source":
            "251 genuine human RAID signature_train documents",

        "machine_reference_used":
            False,

        "detector_information_used":
            False,

        "state_definition": {
            "primary_coordinate":
                (
                    "cumulative probability-"
                    "mass position"
                ),

            "B":
                B,

            "mass_boundaries":
                mass_bounds.tolist(),

            "entropy_coordinate":
                "context entropy",

            "E":
                E,

            "entropy_boundaries":
                entropy_bounds.tolist(),
        },

        "orders":
            list(
                ORDERS
            ),

        "smoothing": {
            "type":
                "hierarchical_dirichlet_backoff",

            "alpha":
                ALPHA,

            "base_prior":
                "uniform_over_B_states",
        },

        "human_documents":
            len(
                human_rows
            ),

        "human_tokens":
            int(
                total_human_tokens
            ),

        "domain_counts":
            dict(
                domain_counts
            ),

        "true_tables": {
            str(order):
                table_to_json(
                    true_tables[
                        order
                    ]
                )
            for order in ORDERS
        },

        "shuffled_null_tables": {
            str(order):
                table_to_json(
                    null_tables[
                        order
                    ]
                )
            for order in ORDERS
        },

        "human_nll_calibration":
            calibration,

        "supported_decoder_orders":
            [
                0,
                1,
                2,
            ],

        "primary_temporal_orders":
            [
                1,
                2,
            ],

        "required_decoder_comparison":
            [
                "vanilla",
                "occupancy_n0",
                "joint_n1",
                "joint_n2",
            ],
    }

    model_path = (
        OUT
        /
        "hjtm_final_v1.json"
    )

    model_path.write_text(
        json.dumps(
            model_object,
            indent=2,
            ensure_ascii=False,
        )
        +
        "\n",
        encoding="utf-8",
    )

    boundaries_path = (
        OUT
        /
        "state_boundaries.json"
    )

    boundaries_path.write_text(
        json.dumps(
            {
                "B":
                    B,

                "mass_boundaries":
                    mass_bounds.tolist(),

                "E":
                    E,

                "entropy_boundaries":
                    entropy_bounds.tolist(),
            },
            indent=2,
        )
        +
        "\n",
        encoding="utf-8",
    )

    scores_path = (
        OUT
        /
        "human_fullmodel_nll.jsonl"
    )

    with scores_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in human_scores:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                +
                "\n"
            )

    summary = {
        "status":
            "PASS",

        "model":
            str(
                model_path.relative_to(
                    ROOT
                )
            ),

        "model_sha256":
            sha256_file(
                model_path
            ),

        "boundaries":
            str(
                boundaries_path.relative_to(
                    ROOT
                )
            ),

        "boundaries_sha256":
            sha256_file(
                boundaries_path
            ),

        "human_scores":
            str(
                scores_path.relative_to(
                    ROOT
                )
            ),

        "human_scores_sha256":
            sha256_file(
                scores_path
            ),

        "T1B_manifest_sha256":
            manifest_sha,

        "human_documents":
            len(
                human_rows
            ),

        "human_tokens":
            int(
                total_human_tokens
            ),

        "orders":
            list(
                ORDERS
            ),

        "B":
            B,

        "E":
            E,

        "alpha":
            ALPHA,
    }

    summary_path = (
        OUT
        /
        "summary.json"
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

    print(
        "\n===== FINAL HASHES ====="
    )

    print(
        "model SHA256:",
        summary[
            "model_sha256"
        ],
    )

    print(
        "boundaries SHA256:",
        summary[
            "boundaries_sha256"
        ],
    )

    print(
        "human scores SHA256:",
        summary[
            "human_scores_sha256"
        ],
    )

    print(
        "summary SHA256:",
        sha256_file(
            summary_path
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
        "HJTM FINAL DECODER MODEL V1: PASS"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()
