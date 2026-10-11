#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/hpcd"))

import run_t2b7_tv_matched_generation_v1 as base

from hpcd_controller_v1 import HJTMModel
from hpcd_reference_aligned_v2 import (
    ReferenceAlignedHPCDController,
)


MODEL_PATH = Path(
    os.environ["HPCD_MODEL_PATH"]
)


HJTM_PATH = (
    ROOT
    / "results/hjtm/final_hjtm_v1/"
      "hjtm_final_v1.json"
)

MANIFEST = Path(
    os.environ["HPCD_MANIFEST"]
)


MANIFEST_SHA_FILE = Path(
    os.environ["HPCD_MANIFEST_SHA_FILE"]
)


OUT_DIR = Path(
    os.environ["HPCD_OUT_DIR"]
)


OUT_JSONL = (
    OUT_DIR
    / "candidates.jsonl"
)

SUMMARY = (
    OUT_DIR
    / "summary.json"
)


BASE_SEED = 2027

K = int(os.environ.get("HPCD_K", "4"))

ATTEMPTS = int(os.environ.get("HPCD_ATTEMPTS", "2"))

TEMPERATURE = 1.0

TOP_P = 0.95


CONDITIONS = (
    {
        "label": "vanilla_t10",
        "order": 1,
        "beta": 0.0,
        "temperature": 1.0,
    },
    {
        "label": "vanilla_t11",
        "order": 1,
        "beta": 0.0,
        "temperature": 1.1,
    },
    {
        "label": "hpcd_n0",
        "order": 0,
        "beta": 0.22,
        "temperature": 1.0,
    },
)


def sha256_file(path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def stable_seed(*parts):

    payload = "|".join(
        str(x)
        for x in parts
    ).encode("utf-8")

    digest = hashlib.sha256(
        payload
    ).digest()

    return int.from_bytes(
        digest[:4],
        "big",
        signed=False,
    )


def set_all_seeds(seed):

    random.seed(seed)

    np.random.seed(
        seed
        %
        (2**32)
    )

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(
        seed
    )


def load_jsonl(path):

    rows = []

    if not path.exists():
        return rows

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


def append_jsonl(path, row):

    with path.open(
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            +
            "\n"
        )

        f.flush()

        os.fsync(
            f.fileno()
        )


def main():

    print("=" * 100)
    print("FRESH95 — FROZEN HELD-OUT CANDIDATE GENERATION")
    print("=" * 100)

    expected_manifest_sha = (
        MANIFEST_SHA_FILE
        .read_text(
            encoding="utf-8"
        )
        .split()[0]
    )

    actual_manifest_sha = (
        sha256_file(
            MANIFEST
        )
    )

    print(
        "manifest SHA256:",
        actual_manifest_sha,
    )

    if (
        actual_manifest_sha
        !=
        expected_manifest_sha
    ):
        raise RuntimeError(
            "Manifest SHA mismatch"
        )

    manifest = json.loads(
        MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    docs = manifest[
        "documents"
    ]

    expected_docs = int(
        os.environ.get(
            "HPCD_EXPECTED_DOCS",
            "95",
        )
    )

    if len(docs) != expected_docs:
        raise RuntimeError(
            f"Expected {expected_docs} documents; "
            f"found {len(docs)}"
        )


    domain_counts = Counter(
        d["domain"]
        for d in docs
    )

    generator_counts = Counter(
        d["generator_key"]
        for d in docs
    )

    print(
        "domain counts:",
        dict(domain_counts),
    )

    print(
        "generator counts:",
        dict(generator_counts),
    )

    total_expected = (
        len(docs)
        *
        K
        *
        ATTEMPTS
        *
        len(CONDITIONS)
    )

    print(
        "expected candidate generations:",
        total_expected,
    )

    expected_total = (
        expected_docs
        * K
        * ATTEMPTS
        * len(CONDITIONS)
    )

    if total_expected != expected_total:
        raise RuntimeError(
            f"Expected exactly {expected_total} generations"
        )

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "Apple MPS"

    elif torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)

    else:
        raise RuntimeError(
            "Neither MPS nor CUDA is available"
        )

    print(
        "DEVICE:",
        device_name,
    )


    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing_rows = load_jsonl(
        OUT_JSONL
    )

    completed = {
        (
            row[
                "instance_id"
            ],
            int(
                row["slot"]
            ),
            int(
                row["attempt"]
            ),
            row[
                "condition"
            ],
        )
        for row in existing_rows
    }

    print(
        "already completed:",
        len(completed),
    )

    if len(completed) > total_expected:
        raise RuntimeError(
            "Output contains more rows than expected"
        )

    print()
    print(
        "Loading Qwen2.5-3B..."
    )

    load_start = time.time()

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
            torch_dtype=torch.float16,
            low_cpu_mem_usage=False,
        )
    )

    model.to(
        device
    )

    model.eval()

    print(
        "model load seconds:",
        round(
            time.time()
            -
            load_start,
            3,
        ),
    )

    hjtm = HJTMModel(
        HJTM_PATH
    )

    controller = (
        ReferenceAlignedHPCDController(
            hjtm
        )
    )

    base.TEMPERATURE = (
        TEMPERATURE
    )

    base.TOP_P = (
        TOP_P
    )

    run_start = time.time()

    newly_generated = 0

    global_index = len(
        completed
    )

    for doc in docs:

        instance_id = doc[
            "instance_id"
        ]

        generator_key = doc[
            "generator_key"
        ]

        sample_id = doc[
            "sample_id"
        ]

        domain = doc[
            "domain"
        ]

        source_text = doc[
            "source_text"
        ]

        for slot in range(
            K
        ):

            for attempt in range(
                ATTEMPTS
            ):

                seed = stable_seed(
                    str(
                        BASE_SEED
                    ),
                    generator_key,
                    sample_id,
                    str(
                        slot
                    ),
                    str(
                        attempt
                    ),
                )

                for condition in CONDITIONS:
                    current_temperature = float(
                        condition["temperature"]
                    )

                    base.TEMPERATURE = (
                        current_temperature
                    )


                    key = (
                        instance_id,
                        slot,
                        attempt,
                        condition[
                            "label"
                        ],
                    )

                    if key in completed:
                        continue

                    set_all_seeds(
                        seed
                    )

                    base.SEED = (
                        seed
                    )

                    t0 = time.time()

                    rec = base.generate_one(
                        source_text=
                            source_text,

                        domain=
                            domain,

                        order=
                            condition[
                                "order"
                            ],

                        beta=
                            condition[
                                "beta"
                            ],

                        label=
                            condition[
                                "label"
                            ],

                        mode=
                            "typicality",

                        model=
                            model,

                        tokenizer=
                            tokenizer,

                        hjtm=
                            hjtm,

                        controller=
                            controller,

                        device=
                            device,
                    )

                    elapsed = (
                        time.time()
                        -
                        t0
                    )

                    global_index += 1

                    newly_generated += 1

                    row = {
                        "instance_id":
                            instance_id,

                        "manifest_index":
                            int(
                                doc[
                                    "manifest_index"
                                ]
                            ),

                        "generator_key":
                            generator_key,

                        "sample_id":
                            sample_id,

                        "domain":
                            domain,

                        "source_text":
                            source_text,

                        "slot":
                            int(
                                slot
                            ),

                        "attempt":
                            int(
                                attempt
                            ),

                        "seed":
                            int(
                                seed
                            ),

                        "condition":
                            condition[
                                "label"
                            ],

                        "order":
                            int(
                                condition[
                                    "order"
                                ]
                            ),

                        "beta":
                            float(
                                condition[
                                    "beta"
                                ]
                            ),

                        "temperature":
                            current_temperature,

                        "top_p":
                            TOP_P,

                        "text":
                            rec[
                                "text"
                            ],

                        "trajectory_typicality":
                            float(
                                rec[
                                    "trajectory_typicality"
                                ]
                            ),

                        "mean_control_tv":
                            float(
                                rec[
                                    "mean_control_tv"
                                ]
                            ),

                        "mean_support_size":
                            float(
                                rec[
                                    "mean_support_size"
                                ]
                            ),

                        "controllable_fraction":
                            float(
                                rec[
                                    "controllable_fraction"
                                ]
                            ),

                        "generation_seconds":
                            float(
                                elapsed
                            ),
                    }

                    append_jsonl(
                        OUT_JSONL,
                        row,
                    )

                    completed.add(
                        key
                    )

                    print(
                        f"[{len(completed):03d}/"
                        f"{total_expected}] "
                        f"{domain:7s} "
                        f"{generator_key:11s} "
                        f"slot={slot} "
                        f"att={attempt} "
                        f"{condition['label']:8s} "
                        f"typ={row['trajectory_typicality']:.4f} "
                        f"TV={row['mean_control_tv']:.4f} "
                        f"{elapsed:.2f}s",
                        flush=True,
                    )

    rows = load_jsonl(
        OUT_JSONL
    )

    if len(rows) != total_expected:
        raise RuntimeError(
            f"Expected {total_expected} rows; "
            f"found {len(rows)}"
        )

    key_count = Counter(
        (
            row[
                "instance_id"
            ],
            row[
                "slot"
            ],
            row[
                "attempt"
            ],
            row[
                "condition"
            ],
        )
        for row in rows
    )

    duplicate_keys = [
        key
        for key, count
        in key_count.items()
        if count != 1
    ]

    if duplicate_keys:
        raise RuntimeError(
            f"Duplicate/missing key problem: "
            f"{duplicate_keys[:5]}"
        )

    condition_counts = Counter(
        row[
            "condition"
        ]
        for row in rows
    )

    elapsed_total = (
        time.time()
        -
        run_start
    )

    summary = {
        "status":
            "COMPLETE",

        "stage":
            "Fresh95 frozen held-out candidate generation",

        "manifest_sha256":
            actual_manifest_sha,

        "candidate_rows":
            len(rows),

        "expected_candidate_rows":
            total_expected,

        "conditions":
            list(CONDITIONS),

        "K":
            K,

        "attempts":
            ATTEMPTS,

        "base_seed":
            BASE_SEED,

        "temperature_by_condition":
            {
                c["label"]:
                    float(c["temperature"])
                for c in CONDITIONS
            },

        "top_p":
            TOP_P,

        "condition_counts":
            dict(
                condition_counts
            ),

        "domain_counts":
            dict(
                domain_counts
            ),

        "generator_counts":
            dict(
                generator_counts
            ),

        "newly_generated_this_run":
            newly_generated,

        "run_seconds":
            float(
                elapsed_total
            ),

        "candidates_sha256":
            sha256_file(
                OUT_JSONL
            ),
    }

    SUMMARY.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        +
        "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("FRESH95 CANDIDATE GENERATION COMPLETE")
    print("=" * 100)

    print(
        "rows:",
        len(rows),
    )

    print(
        "condition counts:",
        dict(
            condition_counts
        ),
    )

    print(
        "run seconds:",
        round(
            elapsed_total,
            3,
        ),
    )

    print(
        "candidates SHA256:",
        sha256_file(
            OUT_JSONL
        ),
    )

    print(
        "summary SHA256:",
        sha256_file(
            SUMMARY
        ),
    )


if __name__ == "__main__":
    main()
