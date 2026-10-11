#!/usr/bin/env python3

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import pickle
import random
import re
import sys
import time

from argparse import Namespace
from collections import Counter
from pathlib import Path

import numpy as np
import torch


# =====================================================================
# Frozen paths/settings
# =====================================================================

ROOT = Path(os.environ.get("HPCD_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))

REPO = Path(
    "/path/to/copa_official"
)

INPUT = (
    ROOT
    / "data/processed/"
      "raid_fresh95_machine_sources_v1.jsonl"
)

OUT = (
    ROOT
    / "results/copa/"
      "fresh95_qwen25_3b_v1"
)

CHECKPOINTS = (
    OUT
    / "checkpoints"
)

RAW = (
    OUT
    / "raw_generations.jsonl"
)

FINAL = (
    OUT
    / "final_outputs.jsonl"
)

FAILURES = (
    OUT
    / "failures.jsonl"
)

SUMMARY = (
    OUT
    / "summary.json"
)

MANIFEST = (
    OUT
    / "manifest.json"
)

RUNTIME = (
    OUT
    / "runtime.json"
)

MODEL = Path(
    "/path/to/models/Qwen2.5-3B-Instruct"
)

EXPECTED_INPUT_SHA = (
    "eaccc088b2120879173507d5dd40e48a97e5a05a6ca1de06b5379241db9e6b77"
)

SEED = 0

LAMBDA = 0.5
ALPHA = 1e-5

TEMPERATURE = 1.0
MAX_TRIES = 5

SIM_THRESHOLD = 0.76


# =====================================================================
# Utilities
# =====================================================================

def sha256(path: Path) -> str:

    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def text_sha(text: str) -> str:

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def read_jsonl(path: Path):

    if not path.exists():
        return []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def first(row, names):

    for name in names:

        if (
            name in row
            and row[name] is not None
            and row[name] != ""
        ):
            return row[name]

    raise KeyError(
        f"Missing all of {names}; "
        f"available={sorted(row.keys())}"
    )


def normalize(row):

    return {
        "instance_id":
            str(
                first(
                    row,
                    [
                        "instance_id",
                        "machine_sample_id",
                        "generation_id",
                        "sample_id",
                        "id",
                    ],
                )
            ),

        "base_id":
            str(
                first(
                    row,
                    [
                        "base_id",
                        "source_id",
                        "sample_id",
                        "id",
                    ],
                )
            ),

        "domain":
            str(
                first(
                    row,
                    ["domain"],
                )
            ),

        "source_generator":
            str(
                first(
                    row,
                    [
                        "source_generator",
                        "generator_key",
                        "generator",
                        "model_key",
                        "model",
                    ],
                )
            ),

        "source_text":
            str(
                first(
                    row,
                    [
                        "source_text",
                        "generated_text",
                        "machine_text",
                        "text",
                        "output_text",
                        "generation",
                    ],
                )
            ),
    }


class Tee(io.StringIO):

    def write(self, value):

        sys.__stdout__.write(
            value
        )

        sys.__stdout__.flush()

        return super().write(
            value
        )


TRACE_PATTERN = re.compile(
    r"Current n:(\d+)\s+"
    r"Current length:(\d+)\s+"
    r"Score:([+-]?(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[eE][+-]?\d+)?)"
)


def parse_trace(text):

    trace = []

    for match in TRACE_PATTERN.finditer(
        text
    ):

        trace.append({
            "attempt":
                int(
                    match.group(1)
                ),

            "output_words":
                int(
                    match.group(2)
                ),

            "native_sim":
                float(
                    match.group(3)
                ),
        })

    return trace


def capture_rng():

    return {
        "python":
            random.getstate(),

        "numpy":
            np.random.get_state(),

        "torch_cpu":
            torch.get_rng_state(),

        "torch_cuda":
            (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
    }


def restore_rng(state):

    random.setstate(
        state["python"]
    )

    np.random.set_state(
        state["numpy"]
    )

    torch.set_rng_state(
        state["torch_cpu"]
    )

    if (
        torch.cuda.is_available()
        and state["torch_cuda"] is not None
    ):

        torch.cuda.set_rng_state_all(
            state["torch_cuda"]
        )


def atomic_pickle(path, obj):

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    with tmp.open(
        "wb"
    ) as f:

        pickle.dump(
            obj,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

        f.flush()

        os.fsync(
            f.fileno()
        )

    os.replace(
        tmp,
        path,
    )


def atomic_text(path, text):

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    with tmp.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(text)

        f.flush()

        os.fsync(
            f.fileno()
        )

    os.replace(
        tmp,
        path,
    )


def load_checkpoint(index):

    path = (
        CHECKPOINTS
        / f"{index:03d}.pkl"
    )

    if not path.exists():
        return None

    with path.open(
        "rb"
    ) as f:

        return pickle.load(
            f
        )


def completed_prefix(docs):

    completed = []

    gap_seen = False

    for index, doc in enumerate(
        docs
    ):

        ckpt = load_checkpoint(
            index
        )

        if ckpt is None:

            gap_seen = True
            continue

        if gap_seen:

            raise RuntimeError(
                "Non-contiguous checkpoint set detected."
            )

        record = ckpt[
            "record"
        ]

        assert (
            record["instance_id"]
            ==
            doc["instance_id"]
        )

        assert (
            record["source_text"]
            ==
            doc["source_text"]
        )

        completed.append(
            ckpt
        )

    return completed


def rebuild_outputs(checkpoints):

    rows = [
        x["record"]
        for x in checkpoints
    ]

    raw_text = "".join(
        json.dumps(
            row,
            ensure_ascii=False,
        )
        + "\n"
        for row in rows
    )

    atomic_text(
        RAW,
        raw_text,
    )

    failures = [
        row
        for row in rows
        if row["copa_abstain"]
    ]

    failure_text = "".join(
        json.dumps(
            row,
            ensure_ascii=False,
        )
        + "\n"
        for row in failures
    )

    atomic_text(
        FAILURES,
        failure_text,
    )

    return rows


# =====================================================================
# Frozen Fresh95 validation
# =====================================================================

assert sha256(
    INPUT
) == EXPECTED_INPUT_SHA

raw_input = read_jsonl(
    INPUT
)

docs = [
    normalize(row)
    for row in raw_input
]

assert len(docs) == 95

assert len({
    x["instance_id"]
    for x in docs
}) == 95

assert len({
    x["base_id"]
    for x in docs
}) == 48

assert Counter(
    x["domain"]
    for x in docs
) == Counter({
    "news": 24,
    "reddit": 24,
    "reviews": 24,
    "wiki": 23,
})

assert Counter(
    x["source_generator"]
    for x in docs
) == Counter({
    "Qwen2.5-3B-Instruct": 48,
    "DeepSeek-R1-Distill-Qwen-7B": 47,
})

print(
    "Fresh95 frozen universe: PASS"
)

print(
    "rows=95 base_ids=48"
)


# =====================================================================
# Author implementation
# =====================================================================

os.chdir(
    REPO
)

from data_builder import DataBuilder


args = Namespace(
    openai_model=None,
    gemini_model=None,

    base_model_name="qwen2.5-3b",

    device="cuda",

    cache_dir="/path/to/copa_cache",

    dataset="xsum",
    data_name="fresh95",
    source_model="fresh95",

    output_file=None,

    n_samples=95,

    paraphrase=True,

    lamda=LAMBDA,
    alpha=ALPHA,

    batch_size=1,

    do_top_k=False,
    top_k=40,

    do_top_p=False,
    top_p=0.96,

    do_temperature=True,
    temperature=TEMPERATURE,

    seed=SEED,

    max_tries=MAX_TRIES,
)


# Official CLI seed behavior.
random.seed(
    SEED
)

np.random.seed(
    SEED
)

torch.manual_seed(
    SEED
)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        SEED
    )


# =====================================================================
# Load CoPA once
# =====================================================================

print()
print("=" * 88)
print("LOADING CoPA-3B FOR FULL FRESH95")
print("=" * 88)

load_start = time.time()

builder = DataBuilder(
    args
)

load_seconds = (
    time.time()
    - load_start
)

model_device = str(
    next(
        builder.base_model.parameters()
    ).device
)

assert model_device.startswith(
    "cuda"
)

print(
    "MODEL_DEVICE:",
    model_device
)

print(
    "LOAD_SECONDS:",
    round(
        load_seconds,
        3,
    )
)


# =====================================================================
# Resume from atomic per-source checkpoints
# =====================================================================

existing = completed_prefix(
    docs
)

if existing:

    last_state = existing[-1][
        "rng_state_after"
    ]

    restore_rng(
        last_state
    )

    print(
        f"RESUME: {len(existing)}/95 "
        "completed sources restored."
    )

else:

    print(
        "STARTING FRESH FULL RUN: 0/95"
    )


rows = rebuild_outputs(
    existing
)


# =====================================================================
# Run remaining sources
# =====================================================================

for index in range(
    len(existing),
    95,
):

    doc = docs[index]

    print()
    print("=" * 88)

    print(
        f"FULL CoPA-3B "
        f"{index + 1:03d}/095"
    )

    print("=" * 88)

    print(
        "instance_id:",
        doc["instance_id"]
    )

    print(
        "base_id:",
        doc["base_id"]
    )

    print(
        "generator:",
        doc["source_generator"]
    )

    print(
        "domain:",
        doc["domain"]
    )

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    capture = Tee()

    start = time.time()

    error = None
    raw_output = ""

    try:

        with contextlib.redirect_stdout(
            capture
        ):

            result = (
                builder
                ._paraphrase_from_model(
                    doc["source_text"],
                    index,
                    MAX_TRIES,
                )
            )

        if isinstance(
            result,
            str,
        ):

            raw_output = (
                result.strip()
            )

    except Exception as exc:

        error = repr(
            exc
        )

    torch.cuda.synchronize()

    elapsed = (
        time.time()
        - start
    )

    trace = parse_trace(
        capture.getvalue()
    )

    success = (
        error is None
        and bool(
            raw_output
        )
    )

    # Frozen ITT failure handling:
    # technical failure -> retain source unchanged.
    final_output = (
        raw_output
        if success
        else doc["source_text"]
    )

    native_sim_max = (
        max(
            x["native_sim"]
            for x in trace
        )
        if trace
        else None
    )

    threshold_pass = (
        native_sim_max > SIM_THRESHOLD
        if native_sim_max is not None
        else None
    )

    input_tokens = len(
        builder.base_tokenizer(
            doc["source_text"],
            add_special_tokens=False,
        )["input_ids"]
    )

    output_tokens = len(
        builder.base_tokenizer(
            final_output,
            add_special_tokens=False,
        )["input_ids"]
    )

    peak_gib = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
    )

    record = {
        "sample_id":
            doc["instance_id"],

        "instance_id":
            doc["instance_id"],

        "base_id":
            doc["base_id"],

        "domain":
            doc["domain"],

        "source_generator":
            doc["source_generator"],

        "source_text":
            doc["source_text"],

        "copa_raw_text":
            raw_output,

        "copa_text":
            final_output,

        "copa_success":
            bool(
                success
            ),

        "copa_abstain":
            bool(
                not success
            ),

        "official_model":
            "Qwen2.5-3B-Instruct",

        "scientific_label":
            "CoPA-3B",

        "official_config_id":
            "copa_qwen25_3b_fresh95_v1",

        "seed":
            SEED,

        "runtime_seconds":
            float(
                elapsed
            ),

        "peak_cuda_allocated_gib":
            float(
                peak_gib
            ),

        "input_words":
            len(
                doc["source_text"].split()
            ),

        "output_words":
            len(
                final_output.split()
            ),

        "input_tokens":
            int(
                input_tokens
            ),

        "output_tokens":
            int(
                output_tokens
            ),

        "error":
            error,

        "native_metadata": {
            "attempts_used":
                len(
                    trace
                ),

            "attempt_trace":
                trace,

            "native_sim_max":
                native_sim_max,

            "native_similarity_threshold":
                SIM_THRESHOLD,

            "native_threshold_pass":
                threshold_pass,

            "max_tries":
                MAX_TRIES,

            "selection_rule":
                (
                    "native CoPA: stop once SIM > 0.76; "
                    "otherwise highest-SIM candidate "
                    "after max_tries"
                ),
        },
    }

    next_state = capture_rng()

    checkpoint = {
        "index":
            index,

        "record":
            record,

        "rng_state_after":
            next_state,
    }

    checkpoint_path = (
        CHECKPOINTS
        / f"{index:03d}.pkl"
    )

    atomic_pickle(
        checkpoint_path,
        checkpoint,
    )

    # Rebuild from durable checkpoints.
    existing = completed_prefix(
        docs
    )

    rows = rebuild_outputs(
        existing
    )

    done = len(
        rows
    )

    success_n = sum(
        row["copa_success"]
        for row in rows
    )

    abstain_n = (
        done
        - success_n
    )

    total_runtime = sum(
        float(
            row["runtime_seconds"]
        )
        for row in rows
    )

    avg_runtime = (
        total_runtime
        / done
    )

    eta_hours = (
        avg_runtime
        * (
            95
            - done
        )
        / 3600
    )

    print()
    print(
        "SOURCE RESULT:",
        {
            "success":
                success,

            "attempts":
                len(
                    trace
                ),

            "native_sim_max":
                native_sim_max,

            "threshold_pass":
                threshold_pass,

            "runtime_seconds":
                round(
                    elapsed,
                    2,
                ),

            "peak_cuda_gib":
                round(
                    peak_gib,
                    3,
                ),

            "words":
                (
                    record["input_words"],
                    record["output_words"],
                ),
        },
    )

    if error is not None:

        print(
            "ERROR:",
            error,
        )

    print(
        f"PROGRESS {done}/95 | "
        f"success={success_n} | "
        f"abstain={abstain_n} | "
        f"avg={avg_runtime:.2f}s/source | "
        f"ETA={eta_hours:.2f}h",
        flush=True,
    )


# =====================================================================
# Final freeze
# =====================================================================

existing = completed_prefix(
    docs
)

assert len(
    existing
) == 95

rows = rebuild_outputs(
    existing
)

assert len(
    rows
) == 95

assert len({
    row["instance_id"]
    for row in rows
}) == 95


# Exact frozen-source verification.
source_map = {
    doc["instance_id"]:
        doc
    for doc in docs
}

for row in rows:

    source = source_map[
        row["instance_id"]
    ]

    assert (
        row["source_text"]
        ==
        source["source_text"]
    )

    assert (
        row["base_id"]
        ==
        source["base_id"]
    )

    assert (
        row["domain"]
        ==
        source["domain"]
    )

    assert (
        row["source_generator"]
        ==
        source["source_generator"]
    )


# Clean final ITT output file.
final_rows = []

for row in rows:

    final_rows.append({
        "sample_id":
            row["sample_id"],

        "instance_id":
            row["instance_id"],

        "base_id":
            row["base_id"],

        "domain":
            row["domain"],

        "source_generator":
            row["source_generator"],

        "source_text":
            row["source_text"],

        "copa_text":
            row["copa_text"],

        "copa_success":
            row["copa_success"],

        "copa_abstain":
            row["copa_abstain"],

        "official_model":
            row["official_model"],

        "scientific_label":
            row["scientific_label"],

        "official_config_id":
            row["official_config_id"],

        "seed":
            row["seed"],

        "runtime_seconds":
            row["runtime_seconds"],

        "input_words":
            row["input_words"],

        "output_words":
            row["output_words"],

        "input_tokens":
            row["input_tokens"],

        "output_tokens":
            row["output_tokens"],

        "native_metadata":
            row["native_metadata"],
    })


atomic_text(
    FINAL,
    "".join(
        json.dumps(
            row,
            ensure_ascii=False,
        )
        + "\n"
        for row in final_rows
    ),
)


runtime_values = [
    float(
        row["runtime_seconds"]
    )
    for row in rows
]

success_n = sum(
    row["copa_success"]
    for row in rows
)

abstain_n = (
    95
    - success_n
)

threshold_pass_n = sum(
    row[
        "native_metadata"
    ][
        "native_threshold_pass"
    ] is True
    for row in rows
)

attempt_counts = Counter(
    row[
        "native_metadata"
    ][
        "attempts_used"
    ]
    for row in rows
)


summary = {
    "status":
        "COMPLETE",

    "scientific_label":
        "CoPA-3B",

    "expected_rows":
        95,

    "rows":
        95,

    "base_ids":
        48,

    "success":
        int(
            success_n
        ),

    "abstain":
        int(
            abstain_n
        ),

    "coverage":
        float(
            success_n / 95
        ),

    "native_threshold_pass":
        int(
            threshold_pass_n
        ),

    "native_threshold_pass_rate":
        float(
            threshold_pass_n / 95
        ),

    "attempt_count_distribution":
        {
            str(k):
                int(v)
            for k, v in sorted(
                attempt_counts.items()
            )
        },

    "mean_runtime_seconds":
        float(
            np.mean(
                runtime_values
            )
        ),

    "median_runtime_seconds":
        float(
            np.median(
                runtime_values
            )
        ),

    "total_runtime_seconds":
        float(
            np.sum(
                runtime_values
            )
        ),

    "mean_input_words":
        float(
            np.mean([
                row["input_words"]
                for row in rows
            ])
        ),

    "mean_output_words":
        float(
            np.mean([
                row["output_words"]
                for row in rows
            ])
        ),

    "input_sha256":
        sha256(
            INPUT
        ),

    "raw_sha256":
        sha256(
            RAW
        ),

    "final_sha256":
        sha256(
            FINAL
        ),

    "failures_sha256":
        sha256(
            FAILURES
        ),
}


atomic_text(
    SUMMARY,
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
)


manifest = {
    "scientific_label":
        "CoPA-3B",

    "input_sha256":
        sha256(
            INPUT
        ),

    "rows": [
        {
            "instance_id":
                row["instance_id"],

            "base_id":
                row["base_id"],

            "domain":
                row["domain"],

            "source_generator":
                row["source_generator"],

            "source_sha256":
                text_sha(
                    row["source_text"]
                ),

            "copa_sha256":
                text_sha(
                    row["copa_text"]
                ),

            "copa_success":
                row["copa_success"],

            "copa_abstain":
                row["copa_abstain"],
        }
        for row in rows
    ],
}


atomic_text(
    MANIFEST,
    json.dumps(
        manifest,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
)


runtime_obj = {
    "per_source_seconds":
        runtime_values,

    "mean_seconds":
        summary[
            "mean_runtime_seconds"
        ],

    "median_seconds":
        summary[
            "median_runtime_seconds"
        ],

    "total_seconds":
        summary[
            "total_runtime_seconds"
        ],
}


atomic_text(
    RUNTIME,
    json.dumps(
        runtime_obj,
        indent=2,
    )
    + "\n",
)


print()
print("=" * 88)
print("CoPA-3B FRESH95 COMPLETE")
print("=" * 88)

print(
    json.dumps(
        summary,
        indent=2,
    )
)

print()
print(
    "FINAL OUTPUT:",
    FINAL
)

print(
    "FINAL SHA256:",
    sha256(
        FINAL
    )
)

print(
    "SUMMARY SHA256:",
    sha256(
        SUMMARY
    )
)

print(
    "MANIFEST SHA256:",
    sha256(
        MANIFEST
    )
)

print()
print(
    "CoPA-3B GENERATION FREEZE READY"
)
