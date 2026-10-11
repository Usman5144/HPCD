#!/usr/bin/env python3
from __future__ import annotations

import os

import difflib
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from hpcd_controller_v1 import HJTMModel


ROOT = Path(os.environ.get("HPCD_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))

HPCD_INPUT = (
    ROOT
    / "results/hpcd/"
      "t2b7_tv_matched_generation_v1/"
      "generations.jsonl"
)

VANILLA_INPUT = (
    ROOT
    / "results/hpcd/"
      "t2b3_strong_prompt_support_v1/"
      "temperature_generations.jsonl"
)

OUT = (
    ROOT
    / "results/hpcd/"
      "t2b7_tv_matched_common_rescore_v1.json"
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

EXPECTED_VANILLA_SHA = (
    "68fa2c08196c9051b44797e29b8af47367656d7f12944b0cb9fd32dbb4dc4906"
)

ORDERS = (
    0,
    1,
    2,
)

CONDITIONS = (
    "vanilla",
    "hpcd_n0",
    "hpcd_n1",
    "hpcd_n2",
)

EXPECTED_IDS = {
    "news":
        "raid_train_news_cb2094f392792b11",

    "reddit":
        "raid_train_reddit_60d624b49800dd8a",

    "reviews":
        "raid_train_reviews_75579e48b7dbfe52",

    "wiki":
        "raid_train_wiki_7af67fee2b7efa7b",
}


def sha256_file(path: Path) -> str:

    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(
                chunk
            )

    return h.hexdigest()


def load_jsonl(path: Path):

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            if line.strip():

                rows.append(
                    json.loads(
                        line
                    )
                )

    return rows


def score_tokens(
    token_ids,
    *,
    model,
    hjtm,
    device,
    domain,
    order,
    eos_token_id,
):

    clean = []

    for token_id in token_ids:

        token_id = int(
            token_id
        )

        if (
            eos_token_id is not None
            and
            token_id
            ==
            int(
                eos_token_id
            )
        ):
            break

        clean.append(
            token_id
        )

    if len(
        clean
    ) < 2:

        raise RuntimeError(
            "Need at least two "
            "nonterminal output tokens"
        )

    ids = torch.tensor(
        [
            clean
        ],
        dtype=torch.long,
        device=device,
    )

    with torch.inference_mode():

        logits = model(
            input_ids=ids,
            use_cache=False,
        ).logits[
            0,
            :-1,
            :
        ].float()

    targets = ids[
        0,
        1:
    ]

    states = []
    entropy_states = []

    CHUNK = 16

    for start in range(
        0,
        len(
            targets
        ),
        CHUNK,
    ):

        end = min(
            start
            +
            CHUNK,
            len(
                targets
            ),
        )

        z = logits[
            start:end
        ]

        y = targets[
            start:end
        ]

        selected_logits = (
            z.gather(
                1,
                y.unsqueeze(
                    1
                ),
            )
            .squeeze(
                1
            )
        )

        logp = torch.log_softmax(
            z,
            dim=-1,
        )

        p = torch.exp(
            logp
        )

        mass = (
            p
            *
            (
                z
                >=
                selected_logits.unsqueeze(
                    1
                )
            )
        ).sum(
            dim=-1
        )

        entropy = -(
            p
            *
            logp
        ).sum(
            dim=-1
        )

        mass_np = (
            mass
            .detach()
            .cpu()
            .numpy()
        )

        entropy_np = (
            entropy
            .detach()
            .cpu()
            .numpy()
        )

        state_chunk = np.searchsorted(
            hjtm.mass_boundaries,
            mass_np,
            side="right",
        )

        entropy_chunk = np.searchsorted(
            hjtm.entropy_boundaries,
            entropy_np,
            side="right",
        )

        states.extend(
            int(x)
            for x in state_chunk
        )

        entropy_states.extend(
            int(x)
            for x in entropy_chunk
        )

    history = []

    costs = []

    for state, entropy_state in zip(
        states,
        entropy_states,
    ):

        ph = (
            hjtm
            .conditional_vector(
                history=history,
                entropy_state=entropy_state,
                order=order,
            )
        )

        prob = float(
            ph[
                state
            ]
        )

        if (
            prob <= 0
            or
            not math.isfinite(
                prob
            )
        ):
            raise RuntimeError(
                "Invalid HJTM probability"
            )

        costs.append(
            -math.log(
                prob
            )
        )

        history.append(
            state
        )

    nll = float(
        np.mean(
            costs
        )
    )

    mu, sd = (
        hjtm
        .calibration_for(
            order=order,
            domain=domain,
        )
    )

    typicality = abs(
        (
            nll
            -
            float(
                mu
            )
        )
        /
        float(
            sd
        )
    )

    return {
        "scored_tokens":
            int(
                len(
                    costs
                )
            ),

        "nll":
            nll,

        "typicality":
            float(
                typicality
            ),

        "states":
            [
                int(x)
                for x in states
            ],
    }


def main():

    print(
        "=" * 100
    )

    print(
        "HPCD T2B.7 — STRONG-PROMPT "
        "COMMON-METRIC RESCORING"
    )

    print(
        "=" * 100
    )

    if OUT.exists():

        raise RuntimeError(
            f"Refusing overwrite: {OUT}"
        )

    if not HPCD_INPUT.exists():

        raise RuntimeError(
            "Controlled generation file "
            "does not exist yet. "
            "Run T2B.7 generator first."
        )

    if not VANILLA_INPUT.exists():

        raise RuntimeError(
            "T2B.3 Vanilla file missing"
        )

    vanilla_sha = sha256_file(
        VANILLA_INPUT
    )

    hpcd_sha = sha256_file(
        HPCD_INPUT
    )

    print(
        "Vanilla input SHA256:",
        vanilla_sha
    )

    print(
        "HPCD input SHA256:",
        hpcd_sha
    )

    if (
        vanilla_sha
        !=
        EXPECTED_VANILLA_SHA
    ):

        raise RuntimeError(
            "Frozen T2B.3 Vanilla "
            "input hash mismatch"
        )

    controlled_rows = load_jsonl(
        HPCD_INPUT
    )

    if len(
        controlled_rows
    ) != 12:

        raise RuntimeError(
            f"Expected 12 HPCD rows; "
            f"found {len(controlled_rows)}"
        )

    condition_counts = {}

    for condition in (
        "hpcd_n0",
        "hpcd_n1",
        "hpcd_n2",
    ):

        count = sum(
            row[
                "label"
            ]
            ==
            condition
            for row in controlled_rows
        )

        condition_counts[
            condition
        ] = count

        if count != 4:

            raise RuntimeError(
                f"{condition}: "
                f"expected 4 rows, got {count}"
            )

    vanilla_candidates = [
        dict(
            row
        )
        for row in load_jsonl(
            VANILLA_INPUT
        )
        if abs(
            float(
                row[
                    "temperature"
                ]
            )
            -
            1.0
        )
        <
        1e-9
    ]

    if len(
        vanilla_candidates
    ) != 4:

        raise RuntimeError(
            "Expected four strong-prompt "
            "Vanilla rows at T=1.0"
        )

    vanilla_rows = []

    for row in (
        vanilla_candidates
    ):

        rec = dict(
            row
        )

        rec[
            "label"
        ] = "vanilla"

        rec[
            "mean_control_tv"
        ] = 0.0

        rec[
            "max_control_tv"
        ] = 0.0

        vanilla_rows.append(
            rec
        )

    rows = (
        vanilla_rows
        +
        controlled_rows
    )

    if len(
        rows
    ) != 16:

        raise RuntimeError(
            f"Expected 16 combined rows; "
            f"found {len(rows)}"
        )

    observed_ids = {
        row[
            "domain"
        ]:
            row[
                "sample_id"
            ]
        for row in rows
        if (
            row[
                "label"
            ]
            ==
            "vanilla"
        )
    }

    if (
        observed_ids
        !=
        EXPECTED_IDS
    ):

        raise RuntimeError(
            "Frozen document IDs changed: "
            f"{observed_ids}"
        )

    print(
        "Frozen documents verified: PASS"
    )

    if not torch.backends.mps.is_available():

        raise RuntimeError(
            "MPS unavailable"
        )

    device = torch.device(
        "mps"
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
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

    model.to(
        device
    )

    model.eval()

    hjtm = HJTMModel(
        HJTM_PATH
    )

    print()
    print(
        "===== RESCORE 16 OUTPUTS "
        "UNDER n=0 / n=1 / n=2 ====="
    )

    rescored = []

    for i, row in enumerate(
        rows,
        1,
    ):

        source_ids = tokenizer(
            row[
                "source_text"
            ],
            add_special_tokens=False,
        )[
            "input_ids"
        ]

        eos = (
            tokenizer.eos_token_id
        )

        output_ids = []

        for token_id in row[
            "token_ids"
        ]:

            token_id = int(
                token_id
            )

            if (
                eos is not None
                and
                token_id
                ==
                int(
                    eos
                )
            ):
                break

            output_ids.append(
                token_id
            )

        similarity = (
            difflib
            .SequenceMatcher(
                a=source_ids,
                b=output_ids,
                autojunk=False,
            )
            .ratio()
        )

        rec = {
            "sample_id":
                row[
                    "sample_id"
                ],

            "domain":
                row[
                    "domain"
                ],

            "source_generator":
                row[
                    "source_generator"
                ],

            "generation_condition":
                row[
                    "label"
                ],

            "source_text":
                row[
                    "source_text"
                ],

            "text":
                row[
                    "text"
                ],

            "generated_tokens":
                int(
                    len(
                        row[
                            "token_ids"
                        ]
                    )
                ),

            "exact_source_copy":
                (
                    row[
                        "text"
                    ].strip()
                    ==
                    row[
                        "source_text"
                    ].strip()
                ),

            "token_similarity_ratio":
                float(
                    similarity
                ),

            "token_edit_proxy":
                float(
                    1.0
                    -
                    similarity
                ),

            "mean_support_size":
                float(
                    row.get(
                        "mean_support_size",
                        float(
                            "nan"
                        ),
                    )
                ),

            "controllable_fraction":
                float(
                    row.get(
                        "controllable_fraction",
                        float(
                            "nan"
                        ),
                    )
                ),

            "mean_control_tv":
                float(
                    row.get(
                        "mean_control_tv",
                        0.0,
                    )
                ),

            "scores":
                {},
        }

        for order in ORDERS:

            rec[
                "scores"
            ][
                f"n{order}"
            ] = score_tokens(
                row[
                    "token_ids"
                ],
                model=model,
                hjtm=hjtm,
                device=device,
                domain=row[
                    "domain"
                ],
                order=order,
                eos_token_id=(
                    tokenizer.eos_token_id
                ),
            )

        rescored.append(
            rec
        )

        print(
            f"{i:02d}/16",
            rec[
                "generation_condition"
            ],
            rec[
                "domain"
            ],
            "edit=",
            round(
                rec[
                    "token_edit_proxy"
                ],
                4,
            ),
            "typ[n0,n1,n2]=",
            [
                round(
                    rec[
                        "scores"
                    ][
                        f"n{x}"
                    ][
                        "typicality"
                    ],
                    6,
                )
                for x in ORDERS
            ],
        )

    by_key = {
        (
            row[
                "sample_id"
            ],
            row[
                "generation_condition"
            ],
        ):
            row
        for row in rescored
    }

    sample_ids = sorted(
        EXPECTED_IDS.values()
    )

    print()
    print(
        "=" * 100
    )

    print(
        "COMMON-METRIC MATCHED COMPARISONS"
    )

    print(
        "=" * 100
    )

    comparisons = {}

    for condition in (
        "hpcd_n0",
        "hpcd_n1",
        "hpcd_n2",
    ):

        comparisons[
            condition
        ] = {}

        for score_order in ORDERS:

            deltas = []

            improved = 0

            per_document = []

            for sid in sample_ids:

                vanilla = by_key[
                    (
                        sid,
                        "vanilla",
                    )
                ]

                controlled = by_key[
                    (
                        sid,
                        condition,
                    )
                ]

                vanilla_typ = float(
                    vanilla[
                        "scores"
                    ][
                        f"n{score_order}"
                    ][
                        "typicality"
                    ]
                )

                controlled_typ = float(
                    controlled[
                        "scores"
                    ][
                        f"n{score_order}"
                    ][
                        "typicality"
                    ]
                )

                delta = (
                    controlled_typ
                    -
                    vanilla_typ
                )

                deltas.append(
                    delta
                )

                if delta < 0:

                    improved += 1

                per_document.append({
                    "sample_id":
                        sid,

                    "domain":
                        vanilla[
                            "domain"
                        ],

                    "vanilla_typicality":
                        vanilla_typ,

                    "controlled_typicality":
                        controlled_typ,

                    "delta_HPCD_minus_Vanilla":
                        float(
                            delta
                        ),
                })

            record = {
                "mean_delta_typicality":
                    float(
                        np.mean(
                            deltas
                        )
                    ),

                "median_delta_typicality":
                    float(
                        np.median(
                            deltas
                        )
                    ),

                "improved_documents":
                    int(
                        improved
                    ),

                "total_documents":
                    len(
                        deltas
                    ),

                "per_document":
                    per_document,
            }

            comparisons[
                condition
            ][
                f"score_n{score_order}"
            ] = record

            print(
                condition,
                f"scored_with_n{score_order}",
                "mean_delta=",
                round(
                    record[
                        "mean_delta_typicality"
                    ],
                    6,
                ),
                "median_delta=",
                round(
                    record[
                        "median_delta_typicality"
                    ],
                    6,
                ),
                "improved=",
                f"{improved}/"
                f"{len(deltas)}",
            )

    print()
    print(
        "=" * 100
    )

    print(
        "GENERATION / CONTROL SUMMARY"
    )

    print(
        "=" * 100
    )

    condition_summary = {}

    for condition in CONDITIONS:

        subset = [
            row
            for row in rescored
            if (
                row[
                    "generation_condition"
                ]
                ==
                condition
            )
        ]

        summary = {
            "documents":
                len(
                    subset
                ),

            "exact_source_copies":
                int(
                    sum(
                        row[
                            "exact_source_copy"
                        ]
                        for row in subset
                    )
                ),

            "mean_token_edit_proxy":
                float(
                    np.mean(
                        [
                            row[
                                "token_edit_proxy"
                            ]
                            for row in subset
                        ]
                    )
                ),

            "mean_support_size":
                float(
                    np.nanmean(
                        [
                            row[
                                "mean_support_size"
                            ]
                            for row in subset
                        ]
                    )
                ),

            "mean_controllable_fraction":
                float(
                    np.nanmean(
                        [
                            row[
                                "controllable_fraction"
                            ]
                            for row in subset
                        ]
                    )
                ),

            "mean_control_tv":
                float(
                    np.mean(
                        [
                            row[
                                "mean_control_tv"
                            ]
                            for row in subset
                        ]
                    )
                ),
        }

        condition_summary[
            condition
        ] = summary

        print(
            condition,
            "copies=",
            f"{summary['exact_source_copies']}/4",
            "edit=",
            round(
                summary[
                    "mean_token_edit_proxy"
                ],
                5,
            ),
            "support=",
            round(
                summary[
                    "mean_support_size"
                ],
                4,
            ),
            "controllable=",
            round(
                summary[
                    "mean_controllable_fraction"
                ],
                4,
            ),
            "TV=",
            round(
                summary[
                    "mean_control_tv"
                ],
                6,
            ),
        )

    output = {
        "status":
            "COMPLETE",

        "stage":
            (
                "T2B.7 strong-prompt "
                "common-metric matched comparison"
            ),

        "interpretation_rule":
            (
                "Negative delta typicality "
                "means HPCD is closer to the "
                "human typicality target than "
                "matched Vanilla under the same "
                "scoring order."
            ),

        "frozen_generation_setting":
            {
                "temperature":
                    1.0,

                "top_p":
                    0.95,

                "beta":
                    0.25,

                "seed":
                    2027,
            },

        "vanilla_input_sha256":
            vanilla_sha,

        "hpcd_input_sha256":
            hpcd_sha,

        "rows":
            rescored,

        "matched_common_metric_comparisons":
            comparisons,

        "condition_summary":
            condition_summary,
    }

    OUT.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        )
        +
        "\n",
        encoding="utf-8",
    )

    print()
    print(
        "output SHA256:",
        sha256_file(
            OUT
        )
    )

    print(
        "script SHA256:",
        sha256_file(
            Path(
                __file__
            ).resolve()
        )
    )

    print()
    print(
        "=" * 100
    )

    print(
        "HPCD T2B.7 COMMON-METRIC "
        "RESCORING: COMPLETE"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()
