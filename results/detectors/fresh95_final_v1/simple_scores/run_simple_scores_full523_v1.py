#!/usr/bin/env python3
from __future__ import annotations

import os

import csv
import gc
import hashlib
import json
import math
import platform
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(os.environ.get("HPCD_PROJECT_ROOT", str(Path(__file__).resolve().parents[4])))

INPUT = ROOT / (
    "results/detectors/fresh95_final_v1/inputs/"
    "fresh95_detector_input_v1.jsonl"
)

OUTDIR = ROOT / (
    "results/detectors/fresh95_final_v1/simple_scores"
)

JSONL_OUT = OUTDIR / "document_scores_v1.jsonl"
CSV_OUT = OUTDIR / "document_scores_v1.csv"
SUMMARY_OUT = OUTDIR / "summary_v1.json"

EXPECTED_INPUT_SHA = (
    "5e5bc0f60923b43675be252f6c3651e7"
    "0338c46ac4ee1a6ba3d211e8f0cdfcb5"
)

MODEL_ID = "EleutherAI/gpt-neo-2.7B"
MAX_LENGTH = 2048
LOGIT_CHUNK = 128

EXPECTED_COUNTS = Counter({
    "human": 48,
    "original": 95,
    "vanilla_t10": 95,
    "vanilla_t11": 95,
    "copa3b": 95,
    "hpcd_n0": 95,
})


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)
    return h.hexdigest()


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def row_key(r):
    if r["condition"] == "human":
        return f"human::{r['base_id']}"

    return (
        f"{r['condition']}::"
        f"{r['source_sample_id']}"
    )


def score_document(
    text,
    tokenizer,
    model,
    device,
):
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        return_token_type_ids=False,
    )

    ids = encoded["input_ids"]

    original_token_count = int(ids.shape[1])

    if original_token_count < 2:
        raise RuntimeError(
            "Text has fewer than two model tokens"
        )

    encoded = {
        k: v.to(device)
        for k, v in encoded.items()
    }

    with torch.inference_mode():
        output = model(**encoded)

    logits = output.logits[:, :-1, :]
    labels = encoded["input_ids"][:, 1:]

    n = int(labels.numel())

    if n < 1:
        raise RuntimeError(
            "No scored tokens"
        )

    sum_logprob = 0.0
    sum_entropy = 0.0
    sum_rank = 0.0
    sum_logrank = 0.0

    # Work in chunks so entropy/rank do not require
    # several full 2048 x vocabulary FP32 tensors at once.
    for start in range(
        0,
        logits.shape[1],
        LOGIT_CHUNK,
    ):
        end = min(
            start + LOGIT_CHUNK,
            logits.shape[1],
        )

        chunk_logits = (
            logits[:, start:end, :]
            .float()
        )

        chunk_labels = (
            labels[:, start:end]
        )

        log_probs = torch.log_softmax(
            chunk_logits,
            dim=-1,
        )

        target_log_probs = (
            log_probs.gather(
                -1,
                chunk_labels.unsqueeze(-1),
            )
            .squeeze(-1)
        )

        probs = torch.exp(log_probs)

        entropy = -(
            probs * log_probs
        ).sum(dim=-1)

        target_logits = (
            chunk_logits.gather(
                -1,
                chunk_labels.unsqueeze(-1),
            )
        )

        # Rank convention:
        # rank 1 = highest-logit token.
        # Exact comparison; ties are assigned the best tied rank.
        ranks = (
            chunk_logits
            .gt(target_logits)
            .sum(dim=-1)
            .to(torch.float32)
            + 1.0
        )

        log_ranks = torch.log(ranks)

        sum_logprob += float(
            target_log_probs.sum().item()
        )

        sum_entropy += float(
            entropy.sum().item()
        )

        sum_rank += float(
            ranks.sum().item()
        )

        sum_logrank += float(
            log_ranks.sum().item()
        )

        del (
            chunk_logits,
            chunk_labels,
            log_probs,
            target_log_probs,
            probs,
            entropy,
            target_logits,
            ranks,
            log_ranks,
        )

    mean_log_likelihood = (
        sum_logprob / n
    )

    mean_nll = (
        -mean_log_likelihood
    )

    try:
        perplexity = math.exp(mean_nll)
    except OverflowError:
        perplexity = float("inf")

    mean_entropy = (
        sum_entropy / n
    )

    mean_rank = (
        sum_rank / n
    )

    mean_log_rank = (
        sum_logrank / n
    )

    if sum_logrank <= 0.0:
        raise RuntimeError(
            "DetectLLM-LRR denominator <= 0"
        )

    # DetectLLM Eq. (1):
    #
    # LRR =
    #   - sum_i log p_theta(x_i | x_<i)
    #     --------------------------------
    #       sum_i log r_theta(x_i | x_<i)
    #
    detectllm_lrr = (
        -sum_logprob
        / sum_logrank
    )

    return {
        "original_model_token_count":
            original_token_count,

        "token_count_scored":
            n,

        "truncated":
            original_token_count >= MAX_LENGTH,

        "mean_log_likelihood":
            mean_log_likelihood,

        "mean_nll":
            mean_nll,

        "perplexity":
            perplexity,

        "mean_predictive_entropy":
            mean_entropy,

        "mean_rank":
            mean_rank,

        "mean_log_rank":
            mean_log_rank,

        "detectllm_lrr":
            detectllm_lrr,
    }


print("=" * 96)
print("D4 — FINAL Fresh95 SINGLE-LM SCORE SUITE")
print("=" * 96)


# ============================================================
# 1. Frozen input
# ============================================================

observed_sha = sha256_file(INPUT)

if observed_sha != EXPECTED_INPUT_SHA:
    raise RuntimeError(
        "Frozen canonical input SHA mismatch"
    )

rows = load_jsonl(INPUT)

if len(rows) != 523:
    raise RuntimeError(
        f"Expected 523 rows, got {len(rows)}"
    )

counts = Counter(
    r["condition"]
    for r in rows
)

if counts != EXPECTED_COUNTS:
    raise RuntimeError(
        f"Condition mismatch: {counts}"
    )

keys = [row_key(r) for r in rows]

if len(set(keys)) != 523:
    raise RuntimeError(
        "Canonical row keys are not unique"
    )

canonical = {
    row_key(r): r
    for r in rows
}

print("\n===== INPUT =====")
print("rows       :", len(rows))
print("SHA256     :", observed_sha)
print("conditions :", dict(counts))
print("INPUT_STATUS=PASS")


# ============================================================
# 2. Resume validation
# ============================================================

existing = {}

if JSONL_OUT.exists():

    for r in load_jsonl(JSONL_OUT):

        k = r["row_key"]

        if k in existing:
            raise RuntimeError(
                f"Duplicate existing key: {k}"
            )

        if k not in canonical:
            raise RuntimeError(
                f"Unknown existing key: {k}"
            )

        if (
            r["text_sha256"]
            !=
            canonical[k]["text_sha256"]
        ):
            raise RuntimeError(
                f"Text SHA mismatch: {k}"
            )

        existing[k] = r


remaining = [
    r
    for r in rows
    if row_key(r) not in existing
]

print()
print("already completed:", len(existing))
print("remaining        :", len(remaining))


# ============================================================
# 3. Model
# ============================================================

if remaining:

    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS unavailable"
        )

    device = torch.device("mps")

    print("\n===== MODEL =====")
    print("model      :", MODEL_ID)
    print("device     :", device)
    print("dtype      : float16")
    print("max_length :", MAX_LENGTH)
    print("logit chunk:", LOGIT_CHUNK)

    tload = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        local_files_only=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = (
            tokenizer.eos_token_id
        )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        local_files_only=True,
    )

    model = model.to(device)
    model.eval()

    print(
        "model load seconds:",
        round(
            time.perf_counter() - tload,
            3,
        ),
    )

    run_start = time.perf_counter()

    JSONL_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with JSONL_OUT.open(
        "a",
        encoding="utf-8",
    ) as fout:

        for j, src in enumerate(
            remaining,
            1,
        ):

            t0 = time.perf_counter()

            scores = score_document(
                src["text"],
                tokenizer,
                model,
                device,
            )

            elapsed = (
                time.perf_counter()
                - t0
            )

            rec = {
                "row_key":
                    row_key(src),

                "base_id":
                    src["base_id"],

                "sample_id":
                    src["sample_id"],

                "source_sample_id":
                    src["source_sample_id"],

                "domain":
                    src["domain"],

                "source_generator":
                    src["source_generator"],

                "condition":
                    src["condition"],

                "is_human":
                    src["is_human"],

                "text_sha256":
                    src["text_sha256"],

                "reference_model":
                    MODEL_ID,

                "max_length":
                    MAX_LENGTH,

                "status":
                    "ok",

                "runtime_seconds":
                    elapsed,

                **scores,
            }

            fout.write(
                json.dumps(
                    rec,
                    ensure_ascii=False,
                )
                + "\n"
            )

            fout.flush()

            if (
                j == 1
                or j % 25 == 0
                or j == len(remaining)
            ):

                total = (
                    len(existing)
                    + j
                )

                print(
                    f"scored {total}/523 | "
                    f"logp={rec['mean_log_likelihood']:+.4f} | "
                    f"logrank={rec['mean_log_rank']:.4f} | "
                    f"LRR={rec['detectllm_lrr']:.4f}",
                    flush=True,
                )

    print(
        "scoring wall seconds:",
        round(
            time.perf_counter()
            - run_start,
            3,
        ),
    )

    del model
    del tokenizer

    gc.collect()

    try:
        torch.mps.empty_cache()
    except Exception:
        pass


# ============================================================
# 4. Final integrity
# ============================================================

scored = load_jsonl(
    JSONL_OUT
)

if len(scored) != 523:
    raise RuntimeError(
        f"Output incomplete: {len(scored)}/523"
    )

score_keys = [
    r["row_key"]
    for r in scored
]

if (
    len(set(score_keys)) != 523
    or set(score_keys) != set(keys)
):
    raise RuntimeError(
        "Output key universe mismatch"
    )

for r in scored:
    src = canonical[r["row_key"]]

    if (
        r["text_sha256"]
        != src["text_sha256"]
    ):
        raise RuntimeError(
            "Final text alignment failure"
        )

    if r["status"] != "ok":
        raise RuntimeError(
            "Non-OK scoring row"
        )

numeric_fields = [
    "mean_log_likelihood",
    "mean_nll",
    "perplexity",
    "mean_predictive_entropy",
    "mean_rank",
    "mean_log_rank",
    "detectllm_lrr",
]

for r in scored:
    for field in numeric_fields:
        x = float(r[field])
        if not math.isfinite(x):
            raise RuntimeError(
                f"Non-finite {field}: "
                f"{r['row_key']}"
            )

print("\nFINAL_ROWS=523")
print("FAILURES=0")
print(
    "truncated_documents=",
    sum(
        bool(r["truncated"])
        for r in scored
    ),
)


# ============================================================
# 5. CSV
# ============================================================

fieldnames = [
    "row_key",
    "base_id",
    "sample_id",
    "source_sample_id",
    "domain",
    "source_generator",
    "condition",
    "is_human",
    "text_sha256",
    "reference_model",
    "max_length",
    "original_model_token_count",
    "token_count_scored",
    "truncated",
    "mean_log_likelihood",
    "mean_nll",
    "perplexity",
    "mean_predictive_entropy",
    "mean_rank",
    "mean_log_rank",
    "detectllm_lrr",
    "runtime_seconds",
    "status",
]

with CSV_OUT.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for r in scored:
        writer.writerow({
            k: r.get(k)
            for k in fieldnames
        })


# ============================================================
# 6. Descriptive means only
# No inferential claims at this stage.
# ============================================================

by_condition = defaultdict(list)

for r in scored:
    by_condition[
        r["condition"]
    ].append(r)

condition_order = [
    "human",
    "original",
    "vanilla_t10",
    "vanilla_t11",
    "copa3b",
    "hpcd_n0",
]

descriptive = {}

for condition in condition_order:

    rr = by_condition[condition]

    descriptive[condition] = {
        "n":
            len(rr),

        "mean_log_likelihood":
            float(np.mean([
                x["mean_log_likelihood"]
                for x in rr
            ])),

        "mean_nll":
            float(np.mean([
                x["mean_nll"]
                for x in rr
            ])),

        "mean_perplexity":
            float(np.mean([
                x["perplexity"]
                for x in rr
            ])),

        "mean_predictive_entropy":
            float(np.mean([
                x["mean_predictive_entropy"]
                for x in rr
            ])),

        "mean_rank":
            float(np.mean([
                x["mean_rank"]
                for x in rr
            ])),

        "mean_log_rank":
            float(np.mean([
                x["mean_log_rank"]
                for x in rr
            ])),

        "mean_detectllm_lrr":
            float(np.mean([
                x["detectllm_lrr"]
                for x in rr
            ])),
    }


summary = {
    "status":
        "COMPLETE",

    "stage":
        "Fresh95 lightweight single-LM scoring suite",

    "canonical_input":
        str(INPUT.relative_to(ROOT)),

    "canonical_input_sha256":
        observed_sha,

    "rows":
        523,

    "condition_counts":
        dict(counts),

    "reference_model":
        MODEL_ID,

    "device":
        "mps",

    "model_dtype":
        "float16",

    "statistics_dtype":
        "float32",

    "max_length":
        MAX_LENGTH,

    "rank_definition":
        (
            "1 + count(logit > target_logit); "
            "rank 1 is highest-logit token"
        ),

    "metrics": {
        "mean_log_likelihood":
            "higher native value treated as more machine-like",

        "mean_nll":
            "lower native value corresponds to higher log-likelihood",

        "perplexity":
            "lower native value corresponds to higher log-likelihood",

        "mean_predictive_entropy":
            (
                "raw predictive entropy; detector orientation "
                "fixed from DetectLLM convention, not outcomes"
            ),

        "mean_rank":
            (
                "raw mean target-token rank; orientation "
                "fixed before outcome inspection"
            ),

        "mean_log_rank":
            (
                "raw mean log target-token rank; orientation "
                "fixed before outcome inspection"
            ),

        "detectllm_lrr":
            (
                "-sum(log p(target))/sum(log rank), "
                "DetectLLM Eq. (1); higher treated as "
                "more machine-like"
            ),
    },

    "descriptive_means":
        descriptive,

    "document_scores_jsonl":
        str(JSONL_OUT.relative_to(ROOT)),

    "document_scores_jsonl_sha256":
        sha256_file(JSONL_OUT),

    "document_scores_csv":
        str(CSV_OUT.relative_to(ROOT)),

    "document_scores_csv_sha256":
        sha256_file(CSV_OUT),

    "torch_version":
        torch.__version__,

    "transformers_version":
        transformers.__version__,

    "platform":
        platform.platform(),

    "generation_performed":
        False,

    "detector_tuning_performed":
        False,

    "outcome_based_orientation_change":
        False,
}

SUMMARY_OUT.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)


# ============================================================
# 7. Print compact result
# ============================================================

print("\n" + "=" * 96)
print("DESCRIPTIVE CONDITION MEANS")
print("=" * 96)

print(
    f"{'condition':14s} "
    f"{'logP':>9s} "
    f"{'NLL':>9s} "
    f"{'PPL':>9s} "
    f"{'entropy':>9s} "
    f"{'rank':>10s} "
    f"{'logrank':>9s} "
    f"{'LRR':>9s}"
)

for c in condition_order:

    x = descriptive[c]

    print(
        f"{c:14s} "
        f"{x['mean_log_likelihood']:9.4f} "
        f"{x['mean_nll']:9.4f} "
        f"{x['mean_perplexity']:9.3f} "
        f"{x['mean_predictive_entropy']:9.4f} "
        f"{x['mean_rank']:10.2f} "
        f"{x['mean_log_rank']:9.4f} "
        f"{x['mean_detectllm_lrr']:9.4f}"
    )


print("\n===== OUTPUT =====")
print(
    "JSONL SHA :",
    sha256_file(JSONL_OUT),
)
print(
    "CSV SHA   :",
    sha256_file(CSV_OUT),
)
print(
    "summary   :",
    SUMMARY_OUT,
)
print(
    "summary SHA:",
    sha256_file(SUMMARY_OUT),
)

print()
print("=" * 96)
print("D4_SIMPLE_SCORE_SUITE_FULL523=COMPLETE")
print("=" * 96)
