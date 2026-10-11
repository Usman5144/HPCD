#!/usr/bin/env python3

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


INPUT = Path(
    "detector_eval/inputs/detector_eval_corpus_v1.jsonl"
)

OUTDIR = Path(
    "detector_eval/fastdetectgpt_neo27_mac_v1"
)

SCORES = OUTDIR / "fastdetectgpt_neo27_scores_v1.jsonl"
RESULTS = OUTDIR / "fastdetectgpt_neo27_results_v1.json"

EXPECTED_INPUT_SHA = (
    "c2381868db7f8ed820fb30f48b879908734c2e88e3db88d4f3a168472af5fe11"
)

MODEL_NAME = "EleutherAI/gpt-neo-2.7B"

GENERATORS = (
    "qwen25_3b",
    "phi35_mini",
    "llama32_3b",
)


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
    with path.open("r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def discrepancy_analytic(
    logits,
    labels,
):
    labels = labels.unsqueeze(-1)

    lprobs = torch.log_softmax(
        logits,
        dim=-1,
    )

    probs = torch.softmax(
        logits,
        dim=-1,
    )

    log_likelihood = (
        lprobs
        .gather(
            dim=-1,
            index=labels,
        )
        .squeeze(-1)
    )

    mean_ref = (
        probs
        *
        lprobs
    ).sum(
        dim=-1
    )

    var_ref = (
        probs
        *
        torch.square(
            lprobs
        )
    ).sum(
        dim=-1
    ) - torch.square(
        mean_ref
    )

    statistic = (
        log_likelihood.sum(dim=-1)
        -
        mean_ref.sum(dim=-1)
    ) / torch.sqrt(
        var_ref.sum(dim=-1)
    )

    return float(
        statistic.mean().item()
    )


def auc(human, machine):
    h = np.asarray(
        human,
        dtype=np.float64,
    )

    m = np.asarray(
        machine,
        dtype=np.float64,
    )

    diff = (
        m[:, None]
        -
        h[None, :]
    )

    return float(
        np.mean(diff > 0)
        +
        0.5
        *
        np.mean(diff == 0)
    )


def threshold(scores, fpr):
    try:
        return float(
            np.quantile(
                scores,
                1.0 - fpr,
                method="higher",
            )
        )

    except TypeError:
        return float(
            np.quantile(
                scores,
                1.0 - fpr,
                interpolation="higher",
            )
        )


def rate(scores, threshold_value):
    return float(
        np.mean(
            np.asarray(
                scores,
                dtype=np.float64,
            )
            >=
            threshold_value
        )
    )


print("===== FAST-DETECTGPT NEO-2.7B PRECHECK =====")

input_sha = sha256_file(INPUT)

print(
    "input SHA256:",
    input_sha,
)

if input_sha != EXPECTED_INPUT_SHA:
    raise RuntimeError(
        "Frozen detector corpus SHA mismatch"
    )

rows = load_jsonl(INPUT)

if len(rows) != 1057:
    raise RuntimeError(
        f"Expected 1057 rows, found {len(rows)}"
    )

print(
    "roles:",
    dict(
        Counter(
            row["role"]
            for row in rows
        )
    ),
)

if SCORES.exists() or RESULTS.exists():
    raise RuntimeError(
        "Refusing to overwrite existing outputs"
    )

if not torch.backends.mps.is_available():
    raise RuntimeError(
        "MPS is not available on this Mac"
    )

device = torch.device("mps")

print(
    "device:",
    device,
)


print()
print("===== LOAD GPT-NEO-2.7B =====")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = (
        tokenizer.eos_token_id
    )

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    low_cpu_mem_usage=True,
)

model.to(device)
model.eval()

print(
    "model:",
    MODEL_NAME,
)

print(
    "sampling model = scoring model"
)

print(
    "dtype:",
    next(
        model.parameters()
    ).dtype,
)


print()
print("===== TOKEN LENGTH CHECK =====")

token_lengths = []

for row in rows:

    ids = tokenizer(
        row["text"],
        add_special_tokens=True,
        truncation=False,
    )["input_ids"]

    token_lengths.append(
        len(ids)
    )


max_tokens = max(
    token_lengths
)

truncated_documents = sum(
    length > 2048
    for length in token_lengths
)

print(
    "max raw tokens:",
    max_tokens,
)

print(
    "median raw tokens:",
    float(
        np.median(
            token_lengths
        )
    ),
)

print(
    "documents truncated to 2048:",
    truncated_documents,
)


print()
print("===== SCORE 1057 TEXTS =====")

scores = []

with torch.inference_mode():

    for i, row in enumerate(
        rows,
        1,
    ):

        encoded = tokenizer(
            row["text"],
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            return_token_type_ids=False,
        )

        encoded = {
            key:
                value.to(device)
            for key, value
            in encoded.items()
        }

        if (
            encoded[
                "input_ids"
            ].shape[1]
            <
            2
        ):
            raise RuntimeError(
                f"Too few tokens: "
                f"{row['row_id']}"
            )

        labels = (
            encoded[
                "input_ids"
            ][
                :,
                1:
            ]
        )

        logits = (
            model(
                **encoded
            )
            .logits[
                :,
                :-1,
                :
            ]
        )

        score = discrepancy_analytic(
            logits,
            labels,
        )

        if not np.isfinite(
            score
        ):
            raise RuntimeError(
                f"Non-finite score: "
                f"{row['row_id']}"
            )

        scores.append(
            score
        )

        del (
            encoded,
            labels,
            logits,
        )

        if (
            i % 25 == 0
            or
            i == len(rows)
        ):
            print(
                f"scored {i}/{len(rows)}"
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

        output = dict(
            row
        )

        output.update({
            "detector":
                "Fast-DetectGPT",

            "configuration":
                "gpt-neo-2.7B_gpt-neo-2.7B",

            "sampling_model":
                MODEL_NAME,

            "scoring_model":
                MODEL_NAME,

            "criterion":
                "sampling_discrepancy_analytic",

            "detector_score":
                float(score),
        })

        f.write(
            json.dumps(
                output,
                ensure_ascii=False,
            )
            +
            "\n"
        )


calibration = np.asarray([
    score
    for row, score in zip(
        rows,
        scores,
    )
    if (
        row["role"]
        ==
        "human_calibration"
    )
])

human_eval = np.asarray([
    score
    for row, score in zip(
        rows,
        scores,
    )
    if (
        row["role"]
        ==
        "human_eval"
    )
])


thresholds = {
    "fpr1":
        threshold(
            calibration,
            0.01,
        ),

    "fpr5":
        threshold(
            calibration,
            0.05,
        ),
}


print()
print("===== THRESHOLDS =====")

for name, value in thresholds.items():

    print(
        name,
        "threshold:",
        value,
        "empirical FPR:",
        rate(
            calibration,
            value,
        ),
    )


print()
print("===== PER-GENERATOR RESULTS =====")

per_generator = {}

macro_original = []
macro_attack = []

for generator in GENERATORS:

    original = np.asarray([
        score
        for row, score in zip(
            rows,
            scores,
        )
        if (
            row["role"]
            ==
            "machine_original"
            and
            row["generator_key"]
            ==
            generator
        )
    ])

    attack = np.asarray([
        score
        for row, score in zip(
            rows,
            scores,
        )
        if (
            row["role"]
            ==
            "machine_attack"
            and
            row["generator_key"]
            ==
            generator
        )
    ])

    result = {
        "auroc_original":
            auc(
                human_eval,
                original,
            ),

        "auroc_attack":
            auc(
                human_eval,
                attack,
            ),

        "mean_score_original":
            float(
                np.mean(
                    original
                )
            ),

        "mean_score_attack":
            float(
                np.mean(
                    attack
                )
            ),

        "tpr_original_fpr1":
            rate(
                original,
                thresholds[
                    "fpr1"
                ],
            ),

        "tpr_attack_fpr1":
            rate(
                attack,
                thresholds[
                    "fpr1"
                ],
            ),

        "tpr_original_fpr5":
            rate(
                original,
                thresholds[
                    "fpr5"
                ],
            ),

        "tpr_attack_fpr5":
            rate(
                attack,
                thresholds[
                    "fpr5"
                ],
            ),
    }

    result[
        "auroc_drop"
    ] = (
        result[
            "auroc_original"
        ]
        -
        result[
            "auroc_attack"
        ]
    )

    result[
        "tpr_drop_fpr1"
    ] = (
        result[
            "tpr_original_fpr1"
        ]
        -
        result[
            "tpr_attack_fpr1"
        ]
    )

    result[
        "tpr_drop_fpr5"
    ] = (
        result[
            "tpr_original_fpr5"
        ]
        -
        result[
            "tpr_attack_fpr5"
        ]
    )

    per_generator[
        generator
    ] = result

    macro_original.append(
        result[
            "auroc_original"
        ]
    )

    macro_attack.append(
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


original_map = {}
attack_map = {}
action_map = {}

for row, score in zip(
    rows,
    scores,
):

    if (
        row["role"]
        ==
        "machine_original"
    ):

        original_map[
            row[
                "pair_id"
            ]
        ] = score

        action_map[
            row[
                "pair_id"
            ]
        ] = row[
            "action"
        ]

    elif (
        row["role"]
        ==
        "machine_attack"
    ):

        attack_map[
            row[
                "pair_id"
            ]
        ] = score


rewritten_keys = [
    key
    for key in original_map
    if (
        action_map[
            key
        ]
        ==
        "rewrite"
    )
]

if len(
    rewritten_keys
) != 225:
    raise RuntimeError(
        "Expected 225 rewritten pairs"
    )


deltas = np.asarray([
    attack_map[
        key
    ]
    -
    original_map[
        key
    ]
    for key in rewritten_keys
])


macro = {
    "auroc_original":
        float(
            np.mean(
                macro_original
            )
        ),

    "auroc_attack":
        float(
            np.mean(
                macro_attack
            )
        ),
}

macro[
    "auroc_drop"
] = (
    macro[
        "auroc_original"
    ]
    -
    macro[
        "auroc_attack"
    ]
)


paired = {
    "rewritten_225_mean_delta":
        float(
            np.mean(
                deltas
            )
        ),

    "rewritten_225_median_delta":
        float(
            np.median(
                deltas
            )
        ),

    "rewritten_225_fraction_score_decreased":
        float(
            np.mean(
                deltas < 0
            )
        ),
}


results = {
    "detector":
        "Fast-DetectGPT",

    "configuration":
        "GPT-Neo-2.7B / GPT-Neo-2.7B",

    "configuration_type":
        "official supported same-model Fast-DetectGPT configuration",

    "sampling_model":
        MODEL_NAME,

    "scoring_model":
        MODEL_NAME,

    "criterion":
        "sampling_discrepancy_analytic",

    "max_observed_raw_tokens":
        max_tokens,

    "max_scored_tokens":
        2048,

    "truncated_documents":
        truncated_documents,

    "thresholds":
        thresholds,

    "per_generator":
        per_generator,

    "macro":
        macro,

    "paired":
        paired,

    "input_sha256":
        EXPECTED_INPUT_SHA,
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


print()
print("===== FAST-DETECTGPT FINAL =====")

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
    "rewritten-225 mean score delta:",
    paired[
        "rewritten_225_mean_delta"
    ],
)

print(
    "rewritten-225 fraction score decreased:",
    paired[
        "rewritten_225_fraction_score_decreased"
    ],
)

print(
    "scores SHA256:",
    sha256_file(
        SCORES
    ),
)

print(
    "results SHA256:",
    sha256_file(
        RESULTS
    ),
)

print()
print(
    "PASS — FAST-DETECTGPT "
    "GPT-NEO-2.7B MAC EVALUATION COMPLETE"
)
