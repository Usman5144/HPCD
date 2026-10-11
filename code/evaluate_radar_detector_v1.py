#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer


INPUT = Path(
    "detector_eval/inputs/detector_eval_corpus_v1.jsonl"
)

OUTDIR = Path(
    "detector_eval/radar_v1"
)

SCORES = (
    OUTDIR
    /
    "radar_scores_v1.jsonl"
)

RESULTS = (
    OUTDIR
    /
    "radar_results_v1.json"
)

MODEL_NAME = (
    "TrustSafeAI/RADAR-Vicuna-7B"
)

EXPECTED_INPUT_SHA = (
    "c2381868db7f8ed820fb30f48b879908734c2e88e3db88d4f3a168472af5fe11"
)

EXPECTED_ROWS = 1057

BATCH_SIZE = 16
MAX_LENGTH = 512


def sha256_file(path):
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
        "r",
        encoding="utf-8",
    ) as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def quantile_threshold(
    human_scores,
    target_fpr,
):

    human_scores = np.asarray(
        human_scores,
        dtype=np.float64,
    )

    q = (
        1.0
        -
        target_fpr
    )

    try:
        threshold = np.quantile(
            human_scores,
            q,
            method="higher",
        )

    except TypeError:
        threshold = np.quantile(
            human_scores,
            q,
            interpolation="higher",
        )

    return float(
        threshold
    )


def tpr_at_threshold(
    machine_scores,
    threshold,
):

    machine_scores = np.asarray(
        machine_scores,
        dtype=np.float64,
    )

    return float(
        np.mean(
            machine_scores
            >=
            threshold
        )
    )


def fpr_at_threshold(
    human_scores,
    threshold,
):

    human_scores = np.asarray(
        human_scores,
        dtype=np.float64,
    )

    return float(
        np.mean(
            human_scores
            >=
            threshold
        )
    )


def auc(
    human_scores,
    machine_scores,
):

    y = np.concatenate([
        np.zeros(
            len(
                human_scores
            ),
            dtype=int,
        ),
        np.ones(
            len(
                machine_scores
            ),
            dtype=int,
        ),
    ])

    scores = np.concatenate([
        np.asarray(
            human_scores,
            dtype=np.float64,
        ),
        np.asarray(
            machine_scores,
            dtype=np.float64,
        ),
    ])

    return float(
        roc_auc_score(
            y,
            scores,
        )
    )


def summarize(
    human_eval,
    machine_original,
    machine_attack,
    thresholds,
):

    result = {
        "auroc_original":
            auc(
                human_eval,
                machine_original,
            ),

        "auroc_attack":
            auc(
                human_eval,
                machine_attack,
            ),

        "mean_machine_score_original":
            float(
                np.mean(
                    machine_original
                )
            ),

        "mean_machine_score_attack":
            float(
                np.mean(
                    machine_attack
                )
            ),

        "median_machine_score_original":
            float(
                np.median(
                    machine_original
                )
            ),

        "median_machine_score_attack":
            float(
                np.median(
                    machine_attack
                )
            ),
    }

    for label, threshold in thresholds.items():

        result[
            f"tpr_original_{label}"
        ] = tpr_at_threshold(
            machine_original,
            threshold,
        )

        result[
            f"tpr_attack_{label}"
        ] = tpr_at_threshold(
            machine_attack,
            threshold,
        )

        result[
            f"tpr_drop_{label}"
        ] = (
            result[
                f"tpr_original_{label}"
            ]
            -
            result[
                f"tpr_attack_{label}"
            ]
        )

    return result


def main():

    print(
        "===== RADAR PRECHECK ====="
    )

    observed_sha = sha256_file(
        INPUT
    )

    print(
        "input SHA256:",
        observed_sha,
    )

    if observed_sha != EXPECTED_INPUT_SHA:
        raise RuntimeError(
            "Detector input checksum mismatch"
        )

    if (
        SCORES.exists()
        or
        RESULTS.exists()
    ):
        raise RuntimeError(
            "Refusing overwrite of RADAR outputs"
        )

    rows = load_jsonl(
        INPUT
    )

    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} rows, "
            f"found {len(rows)}"
        )

    roles = Counter(
        row[
            "role"
        ]
        for row in rows
    )

    print(
        "role counts:",
        dict(
            roles
        ),
    )


    print()
    print(
        "===== LOAD RADAR ====="
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_NAME
        )
    )

    if torch.cuda.is_available():
        device = torch.device(
            "cuda"
        )

    elif torch.backends.mps.is_available():
        device = torch.device(
            "mps"
        )

    else:
        device = torch.device(
            "cpu"
        )

    model.to(
        device
    )

    model.eval()

    print(
        "device:",
        device,
    )

    print(
        "model labels:",
        model.config.id2label,
    )

    num_labels = int(
        model.config.num_labels
    )

    if num_labels != 2:
        raise RuntimeError(
            f"Expected binary detector, "
            f"found {num_labels} labels"
        )


    print()
    print(
        "===== IDENTIFY AI LABEL ====="
    )

    ai_label = 0
    orientation_infer = False

    print(
        "RADAR official AI label index:",
        ai_label,
    )

    print()
    print(
        "===== SCORE 1057 TEXTS ====="
    )

    raw_probs = []

    texts = [
        row[
            "text"
        ]
        for row in rows
    ]

    with torch.inference_mode():

        for start in range(
            0,
            len(texts),
            BATCH_SIZE,
        ):

            batch_texts = texts[
                start:
                start
                +
                BATCH_SIZE
            ]

            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )

            enc = {
                key:
                    value.to(
                        device
                    )
                for key, value
                in enc.items()
            }

            logits = model(
                **enc
            ).logits

            probs = torch.softmax(
                logits,
                dim=-1,
            )

            raw_probs.append(
                probs.detach()
                .cpu()
                .numpy()
            )

            done = min(
                start
                +
                BATCH_SIZE,
                len(texts),
            )

            if (
                done % 128 == 0
                or
                done == len(texts)
            ):
                print(
                    f"scored {done}/"
                    f"{len(texts)}"
                )


    raw_probs = np.concatenate(
        raw_probs,
        axis=0,
    )

    if raw_probs.shape != (
        EXPECTED_ROWS,
        2,
    ):
        raise RuntimeError(
            f"Unexpected probability shape: "
            f"{raw_probs.shape}"
        )


    if orientation_infer:

        human_idx = [
            i
            for i, row in enumerate(
                rows
            )
            if (
                row[
                    "role"
                ]
                ==
                "human_calibration"
            )
        ]

        machine_idx = [
            i
            for i, row in enumerate(
                rows
            )
            if (
                row[
                    "role"
                ]
                ==
                "machine_original"
            )
        ]

        orientation = []

        for label_idx in (
            0,
            1,
        ):

            human_mean = float(
                np.mean(
                    raw_probs[
                        human_idx,
                        label_idx
                    ]
                )
            )

            machine_mean = float(
                np.mean(
                    raw_probs[
                        machine_idx,
                        label_idx
                    ]
                )
            )

            orientation.append(
                (
                    machine_mean
                    -
                    human_mean,
                    label_idx,
                    human_mean,
                    machine_mean,
                )
            )

        orientation.sort(
            reverse=True
        )

        (
            _,
            ai_label,
            human_mean,
            machine_mean,
        ) = orientation[
            0
        ]

        print(
            "inferred AI label index:",
            ai_label,
        )

        print(
            "calibration human mean probability:",
            human_mean,
        )

        print(
            "machine-original mean probability:",
            machine_mean,
        )


    scores = raw_probs[
        :,
        ai_label
    ].astype(
        np.float64
    )


    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with SCORES.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row, score in zip(
            rows,
            scores,
        ):

            out = dict(
                row
            )

            out[
                "detector"
            ] = "RADAR"

            out[
                "detector_model"
            ] = MODEL_NAME

            out[
                "radar_ai_label_index"
            ] = int(
                ai_label
            )

            out[
                "detector_score"
            ] = float(
                score
            )

            f.write(
                json.dumps(
                    out,
                    ensure_ascii=False,
                )
                +
                "\n"
            )


    print()
    print(
        "===== CALIBRATE THRESHOLDS ====="
    )

    calibration_scores = [
        float(
            score
        )
        for row, score in zip(
            rows,
            scores,
        )
        if (
            row[
                "role"
            ]
            ==
            "human_calibration"
        )
    ]

    thresholds = {
        "fpr1":
            quantile_threshold(
                calibration_scores,
                0.01,
            ),

        "fpr5":
            quantile_threshold(
                calibration_scores,
                0.05,
            ),
    }

    for label, threshold in thresholds.items():

        print(
            label,
            "threshold:",
            threshold,
        )

        print(
            label,
            "calibration empirical FPR:",
            fpr_at_threshold(
                calibration_scores,
                threshold,
            ),
        )


    print()
    print(
        "===== EVALUATION ====="
    )

    human_eval_scores = [
        float(
            score
        )
        for row, score in zip(
            rows,
            scores,
        )
        if (
            row[
                "role"
            ]
            ==
            "human_eval"
        )
    ]

    per_generator = {}

    auroc_original = []
    auroc_attack = []

    for generator in (
        "qwen25_3b",
        "phi35_mini",
        "llama32_3b",
    ):

        original = [
            float(
                score
            )
            for row, score in zip(
                rows,
                scores,
            )
            if (
                row[
                    "role"
                ]
                ==
                "machine_original"
                and
                row[
                    "generator_key"
                ]
                ==
                generator
            )
        ]

        attack = [
            float(
                score
            )
            for row, score in zip(
                rows,
                scores,
            )
            if (
                row[
                    "role"
                ]
                ==
                "machine_attack"
                and
                row[
                    "generator_key"
                ]
                ==
                generator
            )
        ]

        if (
            len(
                original
            )
            !=
            111
            or
            len(
                attack
            )
            !=
            111
        ):
            raise RuntimeError(
                f"{generator}: unexpected "
                f"score counts"
            )

        result = summarize(
            human_eval_scores,
            original,
            attack,
            thresholds,
        )

        per_generator[
            generator
        ] = result

        auroc_original.append(
            result[
                "auroc_original"
            ]
        )

        auroc_attack.append(
            result[
                "auroc_attack"
            ]
        )

        print()
        print(
            generator
        )

        print(
            json.dumps(
                result,
                indent=2,
            )
        )


    paired_original = {}
    paired_attack = {}

    action_map = {}

    for row, score in zip(
        rows,
        scores,
    ):

        if (
            row[
                "role"
            ]
            ==
            "machine_original"
        ):
            paired_original[
                row[
                    "pair_id"
                ]
            ] = float(
                score
            )

            action_map[
                row[
                    "pair_id"
                ]
            ] = row[
                "action"
            ]

        elif (
            row[
                "role"
            ]
            ==
            "machine_attack"
        ):
            paired_attack[
                row[
                    "pair_id"
                ]
            ] = float(
                score
            )


    if set(
        paired_original
    ) != set(
        paired_attack
    ):
        raise RuntimeError(
            "Original/attack score pairs differ"
        )


    all_deltas = np.asarray([
        paired_attack[
            key
        ]
        -
        paired_original[
            key
        ]
        for key in paired_original
    ])

    rewritten_keys = [
        key
        for key in paired_original
        if (
            action_map[
                key
            ]
            ==
            "rewrite"
        )
    ]

    rewritten_deltas = np.asarray([
        paired_attack[
            key
        ]
        -
        paired_original[
            key
        ]
        for key in rewritten_keys
    ])


    paired = {
        "all_333":
            {
                "mean_score_delta_attack_minus_original":
                    float(
                        np.mean(
                            all_deltas
                        )
                    ),

                "median_score_delta_attack_minus_original":
                    float(
                        np.median(
                            all_deltas
                        )
                    ),

                "fraction_score_decreased":
                    float(
                        np.mean(
                            all_deltas
                            <
                            0
                        )
                    ),
            },

        "rewritten_225":
            {
                "mean_score_delta_attack_minus_original":
                    float(
                        np.mean(
                            rewritten_deltas
                        )
                    ),

                "median_score_delta_attack_minus_original":
                    float(
                        np.median(
                            rewritten_deltas
                        )
                    ),

                "fraction_score_decreased":
                    float(
                        np.mean(
                            rewritten_deltas
                            <
                            0
                        )
                    ),
            },
    }


    macro = {
        "auroc_original":
            float(
                np.mean(
                    auroc_original
                )
            ),

        "auroc_attack":
            float(
                np.mean(
                    auroc_attack
                )
            ),

        "auroc_drop":
            float(
                np.mean(
                    auroc_original
                )
                -
                np.mean(
                    auroc_attack
                )
            ),
    }


    results = {
        "detector":
            "RADAR",

        "model":
            MODEL_NAME,

        "ai_label_index":
            int(
                ai_label
            ),

        "max_length":
            MAX_LENGTH,

        "thresholds":
            thresholds,

        "human_calibration_count":
            280,

        "human_eval_count":
            111,

        "per_generator":
            per_generator,

        "macro":
            macro,

        "paired_machine_score_change":
            paired,

        "input_sha256":
            EXPECTED_INPUT_SHA,

        "scores_file":
            str(
                SCORES
            ),
    }


    RESULTS.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
        )
        +
        "\n",
        encoding="utf-8",
    )


    scores_sha = sha256_file(
        SCORES
    )

    results_sha = sha256_file(
        RESULTS
    )


    print()
    print(
        "===== RADAR FINAL ====="
    )

    print(
        "macro AUROC original:",
        macro[
            "auroc_original"
        ],
    )

    print(
        "macro AUROC attack:",
        macro[
            "auroc_attack"
        ],
    )

    print(
        "macro AUROC drop:",
        macro[
            "auroc_drop"
        ],
    )

    print(
        "rewritten-225 mean detector "
        "score delta:",
        paired[
            "rewritten_225"
        ][
            "mean_score_delta_attack_minus_original"
        ],
    )

    print(
        "rewritten-225 fraction score "
        "decreased:",
        paired[
            "rewritten_225"
        ][
            "fraction_score_decreased"
        ],
    )

    print()
    print(
        "scores SHA256:",
        scores_sha,
    )

    print(
        "results SHA256:",
        results_sha,
    )

    print()
    print(
        "PASS — RADAR DETECTOR "
        "EVALUATION COMPLETE"
    )


if __name__ == "__main__":
    main()
