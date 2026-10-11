#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import spacy
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


# =============================================================================
# FROZEN SCIENTIFIC CONTRACT
# =============================================================================

ROOT = Path(".").resolve()

MODEL_ID = (
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
)

REVISION = (
    "6f5cf0a2b59cabb106aca4c287eed12e357e90eb"
)

MODEL_PATH = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--MoritzLaurer--DeBERTa-v3-base-mnli-fever-anli"
    / "snapshots"
    / REVISION
)

CANONICAL = ROOT / (
    "results/detectors/fresh95_final_v1/inputs/"
    "fresh95_detector_input_v1.jsonl"
)

EXPECTED_CANONICAL_SHA = (
    "5e5bc0f60923b43675be252f6c3651e7"
    "0338c46ac4ee1a6ba3d211e8f0cdfcb5"
)

OUTDIR = ROOT / (
    "results/quality/final_semantic_v1/nli"
)

SCORES = OUTDIR / "nli_scores_v1.jsonl"
SUMMARY = OUTDIR / "nli_summary_v1.csv"
BY_DATASET = OUTDIR / "nli_by_dataset_v1.csv"
BY_GENERATOR = OUTDIR / "nli_by_generator_v1.csv"
BY_DATASET_GENERATOR = (
    OUTDIR
    / "nli_by_dataset_generator_v1.csv"
)
PAIRWISE_DATASET = (
    OUTDIR
    / "nli_pairwise_by_dataset_v1.csv"
)
MANIFEST = OUTDIR / "nli_manifest_v1.json"


CONDITIONS = [
    "vanilla_t10",
    "vanilla_t11",
    "copa3b",
    "hpcd_n0",
]

DISPLAY = {
    "vanilla_t10":
        "Vanilla",

    "vanilla_t11":
        "Temperature Control",

    "copa3b":
        "CoPA-3B",

    "hpcd_n0":
        "HPCD",
}

DATASET_ORDER = [
    ("Overall", None),
    ("News", "news"),
    ("Reddit", "reddit"),
    ("Reviews", "reviews"),
    ("Wiki", "wiki"),
]

METRICS = [
    "forward_entailment",
    "reverse_entailment",
    "nli_min",
    "nli_mean",
]

# Frozen BEFORE full outcomes.
MAX_MODEL_LENGTH = 512

# Each pair contains at most:
# 220 + 220 + model special tokens,
# safely below 512.
CHUNK_TOKEN_LIMIT = 220

BATCH_SIZE = 16

BOOTSTRAP_B = 50_000
BOOTSTRAP_SEED = 2027

ENTAILMENT_ID = 0

CHUNK_PROTOCOL = (
    "sentence-aware contiguous chunks <=220 NLI tokens; "
    "source->rewrite: for each rewrite hypothesis chunk, "
    "take maximum entailment over all source premise chunks, "
    "then target-token-weighted mean; reverse symmetrically; "
    "NLI-min=min(document forward, document reverse)"
)


# =============================================================================
# UTILITIES
# =============================================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def load_jsonl(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def append_jsonl(path: Path, row):
    with path.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )
        f.flush()


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


def quantile_ci(x):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    x = x[
        np.isfinite(x)
    ]

    lo, hi = np.quantile(
        x,
        [0.025, 0.975],
    )

    return float(lo), float(hi)


# =============================================================================
# VERIFY FROZEN INPUTS / MODEL
# =============================================================================

print("=" * 100)
print("FINAL FROZEN BIDIRECTIONAL NLI")
print("=" * 100)

if (
    sha256_file(CANONICAL)
    !=
    EXPECTED_CANONICAL_SHA
):
    raise RuntimeError(
        "Canonical Fresh95 SHA mismatch"
    )

if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Frozen NLI snapshot missing: {MODEL_PATH}"
    )

config = AutoConfig.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

id2label = {
    int(k): str(v)
    for k, v in config.id2label.items()
}

EXPECTED_LABELS = {
    0: "entailment",
    1: "neutral",
    2: "contradiction",
}

if {
    k: v.lower()
    for k, v in id2label.items()
} != EXPECTED_LABELS:
    raise RuntimeError(
        f"Unexpected label mapping: {id2label}"
    )

if int(
    config.max_position_embeddings
) != MAX_MODEL_LENGTH:
    raise RuntimeError(
        "NLI architecture limit changed"
    )

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

print("model ID :", MODEL_ID)
print("revision :", REVISION)
print("labels   :", id2label)
print("max len  :", MAX_MODEL_LENGTH)
print("chunk cap:", CHUNK_TOKEN_LIMIT)
print("protocol :", CHUNK_PROTOCOL)
print("MODEL_CONTRACT=PASS")


# =============================================================================
# BUILD FROZEN 380-PAIR UNIVERSE
# =============================================================================

rows = load_jsonl(
    CANONICAL
)

assert len(rows) == 523

original = {
    r["source_sample_id"]: r
    for r in rows
    if r["condition"] == "original"
}

assert len(original) == 95

condition_maps = {}

for condition in CONDITIONS:

    m = {
        r["source_sample_id"]: r
        for r in rows
        if r["condition"] == condition
    }

    assert len(m) == 95
    assert set(m) == set(original)

    condition_maps[
        condition
    ] = m


pairs = []

for condition in CONDITIONS:

    for sid in sorted(original):

        src = original[sid]
        rew = condition_maps[
            condition
        ][sid]

        assert (
            src["base_id"]
            ==
            rew["base_id"]
        )

        assert (
            src["domain"]
            ==
            rew["domain"]
        )

        assert (
            src["source_generator"]
            ==
            rew["source_generator"]
        )

        pairs.append({
            "source_sample_id":
                sid,

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

            "source_text":
                src["text"],

            "rewrite_text":
                rew["text"],

            "source_hash":
                src["text_sha256"],

            "rewrite_hash":
                rew["text_sha256"],
        })


if len(pairs) != 380:
    raise RuntimeError(
        f"Expected 380 pairs, got {len(pairs)}"
    )

print()
print("pair count:", len(pairs))
print(
    "domains:",
    Counter(
        p["domain"]
        for p in pairs
    ),
)
print(
    "methods:",
    Counter(
        p["method"]
        for p in pairs
    ),
)
print("PAIR_UNIVERSE=PASS")


# =============================================================================
# DETERMINISTIC SENTENCE-AWARE CHUNKER
# =============================================================================

segmenter = spacy.blank("en")
segmenter.add_pipe(
    "sentencizer"
)


def token_count(text: str) -> int:
    return len(
        tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
    )


def split_long_sentence(
    sentence: str,
):
    ids = tokenizer(
        sentence,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    chunks = []

    for start in range(
        0,
        len(ids),
        CHUNK_TOKEN_LIMIT,
    ):
        piece_ids = ids[
            start:
            start + CHUNK_TOKEN_LIMIT
        ]

        text = tokenizer.decode(
            piece_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

        if not text:
            raise RuntimeError(
                "Empty decoded token chunk"
            )

        chunks.append(text)

    return chunks


def chunk_document(text: str):
    doc = segmenter(text)

    sentences = [
        sent.text.strip()
        for sent in doc.sents
        if sent.text.strip()
    ]

    if not sentences:
        sentences = [
            text.strip()
        ]

    chunks = []
    current = []
    current_tokens = 0

    for sentence in sentences:

        n = token_count(
            sentence
        )

        if n > CHUNK_TOKEN_LIMIT:

            if current:
                chunks.append(
                    " ".join(
                        current
                    ).strip()
                )
                current = []
                current_tokens = 0

            chunks.extend(
                split_long_sentence(
                    sentence
                )
            )

            continue

        if (
            current
            and
            current_tokens + n
            > CHUNK_TOKEN_LIMIT
        ):
            chunks.append(
                " ".join(
                    current
                ).strip()
            )

            current = []
            current_tokens = 0

        current.append(
            sentence
        )

        current_tokens += n

    if current:
        chunks.append(
            " ".join(
                current
            ).strip()
        )

    if not chunks:
        raise RuntimeError(
            "No chunks generated"
        )

    lengths = [
        token_count(c)
        for c in chunks
    ]

    if max(lengths) > CHUNK_TOKEN_LIMIT:
        raise RuntimeError(
            "Chunk exceeds frozen token cap"
        )

    return chunks, lengths


# =============================================================================
# LOAD MODEL
# =============================================================================

device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)

print()
print("device:", device)

load_start = time.perf_counter()

model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )
)

model.eval()
model.to(device)

model_load_seconds = (
    time.perf_counter()
    -
    load_start
)

print(
    "model load sec:",
    round(
        model_load_seconds,
        3,
    )
)


# =============================================================================
# BATCHED ENTAILMENT
# =============================================================================

def entailment_batch(
    premises,
    hypotheses,
):
    if (
        len(premises)
        !=
        len(hypotheses)
    ):
        raise RuntimeError(
            "Batch length mismatch"
        )

    encoded = tokenizer(
        premises,
        hypotheses,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    actual_lengths = (
        encoded[
            "attention_mask"
        ]
        .sum(dim=1)
        .cpu()
        .numpy()
    )

    if int(
        actual_lengths.max()
    ) > MAX_MODEL_LENGTH:
        raise RuntimeError(
            "Chunk pair exceeded model limit: "
            f"{int(actual_lengths.max())}"
        )

    encoded = {
        k: v.to(device)
        for k, v in encoded.items()
    }

    with torch.inference_mode():
        logits = model(
            **encoded
        ).logits.float()

        probs = torch.softmax(
            logits,
            dim=-1,
        )[
            :,
            ENTAILMENT_ID
        ]

    return (
        probs.cpu()
        .numpy()
        .astype(
            np.float64
        ),
        actual_lengths,
    )


def score_direction(
    premise_chunks,
    hypothesis_chunks,
    hypothesis_token_lengths,
):
    """
    For every hypothesis chunk:
      score against all premise chunks,
      retain maximum entailment support.

    Then compute hypothesis-token-weighted mean.
    """

    tasks_p = []
    tasks_h = []
    target_index = []

    for h_idx, h in enumerate(
        hypothesis_chunks
    ):
        for p in premise_chunks:
            tasks_p.append(p)
            tasks_h.append(h)
            target_index.append(
                h_idx
            )

    scores = np.empty(
        len(tasks_p),
        dtype=np.float64,
    )

    max_pair_tokens = 0

    for start in range(
        0,
        len(tasks_p),
        BATCH_SIZE,
    ):
        end = min(
            len(tasks_p),
            start + BATCH_SIZE,
        )

        batch_scores, lens = (
            entailment_batch(
                tasks_p[
                    start:end
                ],
                tasks_h[
                    start:end
                ],
            )
        )

        scores[
            start:end
        ] = batch_scores

        max_pair_tokens = max(
            max_pair_tokens,
            int(
                np.max(lens)
            ),
        )

    best = np.full(
        len(
            hypothesis_chunks
        ),
        -np.inf,
        dtype=np.float64,
    )

    for score, h_idx in zip(
        scores,
        target_index,
    ):
        if score > best[
            h_idx
        ]:
            best[
                h_idx
            ] = score

    if not np.all(
        np.isfinite(best)
    ):
        raise RuntimeError(
            "Non-finite chunk support"
        )

    weights = np.asarray(
        hypothesis_token_lengths,
        dtype=np.float64,
    )

    if np.sum(weights) <= 0:
        raise RuntimeError(
            "Invalid chunk weights"
        )

    document_score = float(
        np.average(
            best,
            weights=weights,
        )
    )

    return {
        "score":
            document_score,

        "best_support_scores":
            [
                float(x)
                for x in best
            ],

        "max_pair_tokens":
            max_pair_tokens,

        "n_cross_pairs":
            len(tasks_p),
    }


# =============================================================================
# RESUME-SAFE FULL SCORING
# =============================================================================

expected_keys = {
    (
        p["condition"],
        p["source_sample_id"],
    )
    for p in pairs
}

completed = {}

if SCORES.exists():

    print()
    print(
        "Existing partial NLI score file found; "
        "validating for resume."
    )

    for r in load_jsonl(
        SCORES
    ):
        key = (
            r["condition"],
            r["source_sample_id"],
        )

        if key in completed:
            raise RuntimeError(
                f"Duplicate existing key: {key}"
            )

        if key not in expected_keys:
            raise RuntimeError(
                f"Unexpected existing key: {key}"
            )

        completed[
            key
        ] = r

    print(
        "resume rows:",
        len(completed),
        "/ 380",
    )


score_start = time.perf_counter()

for idx, p in enumerate(
    pairs,
    1,
):
    key = (
        p["condition"],
        p["source_sample_id"],
    )

    if key in completed:
        continue

    source_chunks, source_lengths = (
        chunk_document(
            p["source_text"]
        )
    )

    rewrite_chunks, rewrite_lengths = (
        chunk_document(
            p["rewrite_text"]
        )
    )

    # Forward:
    # source premise -> rewrite hypothesis.
    forward = score_direction(
        source_chunks,
        rewrite_chunks,
        rewrite_lengths,
    )

    # Reverse:
    # rewrite premise -> source hypothesis.
    reverse = score_direction(
        rewrite_chunks,
        source_chunks,
        source_lengths,
    )

    fwd = float(
        forward["score"]
    )

    rev = float(
        reverse["score"]
    )

    if not (
        math.isfinite(fwd)
        and
        math.isfinite(rev)
    ):
        raise RuntimeError(
            f"Non-finite NLI score: {key}"
        )

    record = {
        "sample_id":
            p["source_sample_id"],

        "source_sample_id":
            p["source_sample_id"],

        "base_id":
            p["base_id"],

        "domain":
            p["domain"],

        "source_generator":
            p["source_generator"],

        "condition":
            p["condition"],

        "method":
            p["method"],

        "source_hash":
            p["source_hash"],

        "rewrite_hash":
            p["rewrite_hash"],

        "forward_entailment":
            fwd,

        "reverse_entailment":
            rev,

        "nli_min":
            min(
                fwd,
                rev,
            ),

        "nli_mean":
            (
                fwd + rev
            )
            / 2.0,

        "source_chunk_count":
            len(
                source_chunks
            ),

        "rewrite_chunk_count":
            len(
                rewrite_chunks
            ),

        "source_chunk_token_lengths":
            source_lengths,

        "rewrite_chunk_token_lengths":
            rewrite_lengths,

        "forward_cross_chunk_pairs":
            forward[
                "n_cross_pairs"
            ],

        "reverse_cross_chunk_pairs":
            reverse[
                "n_cross_pairs"
            ],

        "max_actual_model_pair_tokens":
            max(
                forward[
                    "max_pair_tokens"
                ],
                reverse[
                    "max_pair_tokens"
                ],
            ),

        "chunk_token_limit":
            CHUNK_TOKEN_LIMIT,

        "model_max_length":
            MAX_MODEL_LENGTH,

        "chunked":
            (
                len(source_chunks) > 1
                or
                len(rewrite_chunks) > 1
            ),

        "chunk_protocol":
            CHUNK_PROTOCOL,

        "status":
            "ok",
    }

    append_jsonl(
        SCORES,
        record,
    )

    completed[
        key
    ] = record

    if (
        len(completed) % 10 == 0
        or
        len(completed) == 380
    ):
        elapsed = (
            time.perf_counter()
            -
            score_start
        )

        done_this_run = max(
            1,
            len(completed)
        )

        print(
            f"[{len(completed):3d}/380] "
            f"{p['method']:20s} "
            f"{p['domain']:8s} "
            f"src_chunks={len(source_chunks)} "
            f"rew_chunks={len(rewrite_chunks)} "
            f"F={fwd:.4f} "
            f"R={rev:.4f} "
            f"MIN={min(fwd, rev):.4f}"
        )


# =============================================================================
# FINAL SCORE-FILE VALIDATION
# =============================================================================

score_rows = load_jsonl(
    SCORES
)

if len(score_rows) != 380:
    raise RuntimeError(
        f"Expected 380 final rows, got {len(score_rows)}"
    )

keys = [
    (
        r["condition"],
        r["source_sample_id"],
    )
    for r in score_rows
]

if len(set(keys)) != 380:
    raise RuntimeError(
        "Duplicate final NLI keys"
    )

if set(keys) != expected_keys:
    raise RuntimeError(
        "Final NLI key universe mismatch"
    )

if not all(
    r["status"] == "ok"
    for r in score_rows
):
    raise RuntimeError(
        "Non-ok final rows"
    )

if max(
    r["max_actual_model_pair_tokens"]
    for r in score_rows
) > MAX_MODEL_LENGTH:
    raise RuntimeError(
        "A scored chunk pair exceeded 512"
    )

print()
print("FULL_NLI_SCORE_FILE=PASS")


# =============================================================================
# STATISTICAL HELPERS
# =============================================================================

rng = np.random.default_rng(
    BOOTSTRAP_SEED
)


def clustered_absolute(
    selected,
    metric,
):
    by_base = defaultdict(
        list
    )

    for r in selected:
        by_base[
            r["base_id"]
        ].append(
            float(
                r[metric]
            )
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

    n_base = len(
        base_values
    )

    draws = rng.integers(
        0,
        n_base,
        size=(
            BOOTSTRAP_B,
            n_base,
        ),
    )

    boot = base_values[
        draws
    ].mean(
        axis=1
    )

    lo, hi = quantile_ci(
        boot
    )

    raw = np.asarray(
        [
            float(
                r[metric]
            )
            for r in selected
        ],
        dtype=np.float64,
    )

    return {
        "n_base_ids":
            n_base,

        "n_instances":
            len(selected),

        "mean":
            float(
                np.mean(raw)
            ),

        "median":
            float(
                np.median(raw)
            ),

        "std":
            float(
                np.std(
                    raw,
                    ddof=1,
                )
            )
            if len(raw) > 1
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


def paired_cluster(
    lhs_rows,
    rhs_rows,
    metric,
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

    if not common:
        raise RuntimeError(
            "No paired samples"
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

        diff = (
            float(
                a[metric]
            )
            -
            float(
                b[metric]
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

    n_base = len(
        base_values
    )

    draws = rng.integers(
        0,
        n_base,
        size=(
            BOOTSTRAP_B,
            n_base,
        ),
    )

    boot = base_values[
        draws
    ].mean(
        axis=1
    )

    lo, hi = quantile_ci(
        boot
    )

    return {
        "n_instances":
            len(
                instance_diffs
            ),

        "n_base_ids":
            n_base,

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


# =============================================================================
# OVERALL SUMMARY
# =============================================================================

summary_rows = []

for condition in CONDITIONS:

    subset = [
        r
        for r in score_rows
        if r["condition"]
        == condition
    ]

    for metric in METRICS:

        stats = clustered_absolute(
            subset,
            metric,
        )

        summary_rows.append({
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
    SUMMARY,
    summary_rows,
)


# =============================================================================
# BY-DATASET — REQUIRED
# =============================================================================

dataset_rows = []

for dataset_label, domain in DATASET_ORDER:

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

        for metric in METRICS:

            stats = clustered_absolute(
                subset,
                metric,
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


# =============================================================================
# BY SOURCE GENERATOR — REQUIRED LEVEL 3
# =============================================================================

generator_values = sorted(
    {
        r["source_generator"]
        for r in score_rows
    }
)

generator_rows = []

for generator in generator_values:

    for condition in CONDITIONS:

        subset = [
            r
            for r in score_rows
            if (
                r["condition"]
                == condition
                and
                r["source_generator"]
                == generator
            )
        ]

        for metric in METRICS:

            stats = clustered_absolute(
                subset,
                metric,
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


# =============================================================================
# DATASET × GENERATOR DIAGNOSTIC
# =============================================================================

dataset_generator_rows = []

for dataset_label, domain in DATASET_ORDER[1:]:

    for generator in generator_values:

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
                    r["source_generator"]
                    == generator
                )
            ]

            if not subset:
                continue

            for metric in METRICS:

                stats = clustered_absolute(
                    subset,
                    metric,
                )

                dataset_generator_rows.append({
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
    dataset_generator_rows,
)


# =============================================================================
# REQUIRED PAIRED NLI COMPARISONS BY DATASET
# =============================================================================

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

pairwise_rows = []

for dataset_label, domain in DATASET_ORDER:

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

        for metric in METRICS:

            stats = paired_cluster(
                lhs_rows,
                rhs_rows,
                metric,
            )

            pairwise_rows.append({
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
                    "positive = HPCD higher semantic preservation",
            })


write_csv(
    PAIRWISE_DATASET,
    pairwise_rows,
)


# =============================================================================
# TERMINAL REPORT
# =============================================================================

print()
print("=" * 100)
print("OVERALL BIDIRECTIONAL NLI")
print("=" * 100)

for condition in CONDITIONS:

    row_f = next(
        r
        for r in summary_rows
        if (
            r["condition"]
            == condition
            and
            r["metric"]
            == "forward_entailment"
        )
    )

    row_r = next(
        r
        for r in summary_rows
        if (
            r["condition"]
            == condition
            and
            r["metric"]
            == "reverse_entailment"
        )
    )

    row_m = next(
        r
        for r in summary_rows
        if (
            r["condition"]
            == condition
            and
            r["metric"]
            == "nli_min"
        )
    )

    print(
        f"{DISPLAY[condition]:20s} "
        f"F={row_f['mean']:.4f} "
        f"R={row_r['mean']:.4f} "
        f"MIN={row_m['mean']:.4f} "
        f"clusterMIN={row_m['cluster_mean']:.4f} "
        f"CI=[{row_m['cluster_ci_low']:.4f}, "
        f"{row_m['cluster_ci_high']:.4f}]"
    )


print()
print("=" * 100)
print("NLI-MIN BY DATASET")
print("=" * 100)

for dataset_label, _ in DATASET_ORDER:

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
                == "nli_min"
            )
        )

        print(
            f"  {DISPLAY[condition]:20s} "
            f"mean={row['mean']:.4f} "
            f"cluster={row['cluster_mean']:.4f} "
            f"CI=[{row['cluster_ci_low']:.4f}, "
            f"{row['cluster_ci_high']:.4f}] "
            f"n={row['n_instances']} "
            f"bases={row['n_base_ids']}"
        )


print()
print("=" * 100)
print("HPCD PAIRWISE NLI-MIN")
print("=" * 100)

for dataset_label, _ in DATASET_ORDER:

    print()
    print(dataset_label)

    for comparison, _, _ in PAIRINGS:

        row = next(
            r
            for r in pairwise_rows
            if (
                r["dataset"]
                == dataset_label
                and
                r["comparison"]
                == comparison
                and
                r["metric"]
                == "nli_min"
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


# =============================================================================
# MANIFEST / HASHES
# =============================================================================

scoring_seconds = (
    time.perf_counter()
    -
    score_start
)

manifest = {
    "status":
        "PASS",

    "model_id":
        MODEL_ID,

    "revision":
        REVISION,

    "local_snapshot":
        str(
            MODEL_PATH
        ),

    "label_mapping":
        id2label,

    "entailment_class_id":
        ENTAILMENT_ID,

    "model_max_length":
        MAX_MODEL_LENGTH,

    "chunk_token_limit":
        CHUNK_TOKEN_LIMIT,

    "chunk_protocol":
        CHUNK_PROTOCOL,

    "chunk_protocol_frozen_before_full_outcomes":
        True,

    "all_documents_use_same_chunk_protocol":
        True,

    "simple_512_truncation_used":
        False,

    "batch_size":
        BATCH_SIZE,

    "device":
        str(
            device
        ),

    "torch_version":
        torch.__version__,

    "canonical_input":
        str(
            CANONICAL
        ),

    "canonical_input_sha256":
        sha256_file(
            CANONICAL
        ),

    "pairs":
        380,

    "conditions":
        CONDITIONS,

    "bootstrap": {
        "unit":
            "base_id",

        "replicates":
            BOOTSTRAP_B,

        "seed":
            BOOTSTRAP_SEED,
    },

    "model_load_seconds":
        model_load_seconds,

    "scoring_seconds_this_invocation":
        scoring_seconds,

    "outputs": {},
}


for p in [
    SCORES,
    SUMMARY,
    BY_DATASET,
    BY_GENERATOR,
    BY_DATASET_GENERATOR,
    PAIRWISE_DATASET,
]:
    manifest[
        "outputs"
    ][
        p.name
    ] = sha256_file(p)


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
print("FINAL NLI ARTIFACTS")
print("=" * 100)

for p in [
    SCORES,
    SUMMARY,
    BY_DATASET,
    BY_GENERATOR,
    BY_DATASET_GENERATOR,
    PAIRWISE_DATASET,
    MANIFEST,
]:
    print(
        sha256_file(p),
        p,
    )

print()
print("=" * 100)
print("FROZEN_BIDIRECTIONAL_NLI=PASS")
print("=" * 100)
