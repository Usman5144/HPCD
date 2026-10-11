#!/usr/bin/env python3
from __future__ import annotations

import os

import ast
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(os.environ.get("HPCD_PROJECT_ROOT", str(Path(__file__).resolve().parents[3])))

sys.path.insert(
    0,
    str(ROOT / "code/hpcd"),
)

from hpcd_controller_v1 import HJTMModel
from reanalyze_t2b7_common_metrics_v1 import score_tokens


# ============================================================================
# Frozen artifacts
# ============================================================================

SOURCE = (
    ROOT
    / "results/confirm48_v1/source_generations/"
      "confirm48_machine_sources_v1.jsonl"
)

DIPPER = (
    ROOT
    / "results/extension_oct2026_v1/dipper/"
      "friend_return_v1/friend_gpu_package_v1/"
      "outputs/confirm48/full/"
      "dipper_confirm48_full_final.jsonl"
)

DIPPER_RETURN_FREEZE = (
    ROOT
    / "results/extension_oct2026_v1/dipper/"
      "dipper_friend_return_freeze_v1.json"
)

PRIMARY_FREEZE = (
    ROOT
    / "results/confirm48_v1/manifests/"
      "confirm48_primary_t0_freeze_v1.json"
)

PRIMARY_CACHE = (
    ROOT
    / "results/confirm48_v1/process/"
      "confirm48_primary_t0_v1/"
      "unique_text_t0_cache.jsonl"
)

PRIMARY_SELECTED_SCORES = (
    ROOT
    / "results/confirm48_v1/process/"
      "confirm48_primary_t0_v1/"
      "selected_outputs_t0_scores.jsonl"
)

HISTORICAL_SCORER = (
    ROOT
    / "code/hpcd/"
      "run_fresh95_primary_t0_v1.py"
)

SCORE_HELPER = (
    ROOT
    / "code/hpcd/"
      "reanalyze_t2b7_common_metrics_v1.py"
)

CONTROLLER = (
    ROOT
    / "code/hpcd/"
      "hpcd_controller_v1.py"
)

HJTM_PATH = (
    ROOT
    / "results/hjtm/final_hjtm_v1/"
      "hjtm_final_v1.json"
)

MODEL_PATH = Path(
    "/path/to/huggingface/hub/"
    "models--Qwen--Qwen2.5-3B-Instruct/"
    "snapshots/"
    "aa8e72537993ba99e69dfaafa59ed015b17504d1"
)

OUT = (
    ROOT
    / "results/extension_oct2026_v1/dipper/"
      "confirm48_primary_t0_v1"
)

CACHE = OUT / "original_dipper_t0_cache.jsonl"
NEW_SCORES = OUT / "original_dipper_primary_scores.jsonl"
ALL_SCORES = OUT / "primary_scores_all_four_methods.jsonl"
PAIRS = OUT / "pairwise_instance_effects.csv"
CLUSTERS = OUT / "pairwise_cluster_effects.csv"
SUMMARY = OUT / "summary.json"


# ============================================================================
# Immutable hashes
# ============================================================================

EXPECTED = {
    SOURCE:
        "d7d72918046b686f8c4554dd98b8036b"
        "cfb10ce18744bd0ced3216f01616b356",

    DIPPER:
        "aee8bea9a3d9cf3aa6680c97558ac716"
        "4f12c1401128ae9c2bdb0cad9de32127",

    DIPPER_RETURN_FREEZE:
        "da6a1ce6de6f8f6aa8d335dea3ac7d1e"
        "289bb9d3fccf21453b635aea74d7d32d",

    PRIMARY_FREEZE:
        "b1f941918340ba5ec80d82316aef5d973"
        "cc48c89d7bf6f905f6bc73841b5d3a4",

    PRIMARY_CACHE:
        "f9a4fb066f9cbba314149590a26f4bc0e"
        "5bc6111e08b5251a270c7e83bd8433c",

    PRIMARY_SELECTED_SCORES:
        "9b4c7354bfdc690d37cf6e42970297ad"
        "7ff9d4291fcdd0b7add58dabc68ae1a6",

    HISTORICAL_SCORER:
        "5321752a458e060591623acf62ec14ab6"
        "a2be0c94d7033bd9d742a80e000953d",

    SCORE_HELPER:
        "97adab7677936e1a528bb3a9d9418a11"
        "f35ca0e3f7e12849d89705d0552a4f83",

    CONTROLLER:
        "d9033d811267a52b3c66d0dd77be8e2c"
        "48f4270bb9d126396b7a8b5567a2ef0f",

    HJTM_PATH:
        "1abd60c38a8787c4d44cf4e81ec157e"
        "af18745ce1dec28c366417030b1a22c61",
}


B = 50_000
BOOT_SEED = 2027

COMPARATORS = (
    "original",
    "vanilla_t10",
    "hpcd_n0",
)


def sha(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


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


def write_jsonl(path: Path, rows):
    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


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


# ============================================================================
# Recover exact historical statistical/helper implementations.
# ============================================================================

def recover_historical_functions():

    source = HISTORICAL_SCORER.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source,
        filename=str(HISTORICAL_SCORER),
    )

    wanted = {
        "text_key",
        "bootstrap_stratified",
    }

    nodes = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.FunctionDef)
            and node.name in wanted
        )
    ]

    if {node.name for node in nodes} != wanted:
        raise RuntimeError(
            "Unable to recover exact historical functions"
        )

    module = ast.Module(
        body=nodes,
        type_ignores=[],
    )

    ast.fix_missing_locations(module)

    namespace = {
        "np":
            np,

        "hashlib":
            hashlib,

        "defaultdict":
            defaultdict,

        "B":
            B,

        "BOOT_SEED":
            BOOT_SEED,
    }

    exec(
        compile(
            module,
            str(HISTORICAL_SCORER),
            "exec",
        ),
        namespace,
    )

    return (
        namespace["text_key"],
        namespace["bootstrap_stratified"],
    )


def main():

    print("=" * 108)
    print(
        "CONFIRM48 DIPPER — "
        "FROZEN PRIMARY QWEN2.5-3B T0 EVALUATION"
    )
    print("=" * 108)

    # ========================================================================
    # 1. Provenance firewall
    # ========================================================================

    print()
    print("===== FROZEN PROVENANCE =====")

    for path, expected in EXPECTED.items():

        if not path.exists():
            raise RuntimeError(
                f"Missing frozen artifact: {path}"
            )

        actual = sha(path)

        print(
            actual,
            path.relative_to(ROOT),
        )

        if actual != expected:
            raise RuntimeError(
                f"SHA mismatch:\n{path}\n"
                f"expected={expected}\n"
                f"actual={actual}"
            )

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Missing frozen model: {MODEL_PATH}"
        )

    primary_freeze = json.loads(
        PRIMARY_FREEZE.read_text(
            encoding="utf-8"
        )
    )

    if (
        primary_freeze.get("status")
        != "FROZEN_COMPLETE"
    ):
        raise RuntimeError(
            "CONFIRM48 primary endpoint not frozen"
        )

    if not primary_freeze.get(
        "primary_endpoint_locked",
        False,
    ):
        raise RuntimeError(
            "CONFIRM48 primary endpoint not locked"
        )

    if (
        primary_freeze.get(
            "scientific_classification"
        )
        != "REPRODUCED"
    ):
        raise RuntimeError(
            "Unexpected primary classification"
        )

    dipper_freeze = json.loads(
        DIPPER_RETURN_FREEZE.read_text(
            encoding="utf-8"
        )
    )

    if (
        dipper_freeze.get("status")
        != "FROZEN_COMPLETE"
    ):
        raise RuntimeError(
            "DIPPER return not frozen"
        )

    print("PROVENANCE FIREWALL: PASS")

    (
        text_key,
        bootstrap_stratified,
    ) = recover_historical_functions()

    print(
        "historical text_key()             : RECOVERED"
    )

    print(
        "historical bootstrap_stratified(): RECOVERED"
    )

    # ========================================================================
    # 2. Load frozen datasets
    # ========================================================================

    sources = load_jsonl(
        SOURCE
    )

    dipper_rows = load_jsonl(
        DIPPER
    )

    selected = load_jsonl(
        PRIMARY_SELECTED_SCORES
    )

    primary_cache_rows = load_jsonl(
        PRIMARY_CACHE
    )

    if len(sources) != 95:
        raise RuntimeError(
            f"Source rows={len(sources)} != 95"
        )

    if len(dipper_rows) != 95:
        raise RuntimeError(
            f"DIPPER rows={len(dipper_rows)} != 95"
        )

    if len(selected) != 190:
        raise RuntimeError(
            f"Primary selected rows={len(selected)} != 190"
        )

    if Counter(
        r["condition"]
        for r in selected
    ) != Counter({
        "vanilla_t10": 95,
        "hpcd_n0": 95,
    }):
        raise RuntimeError(
            "Primary selected condition counts wrong"
        )

    if Counter(
        r["status"]
        for r in dipper_rows
    ) != Counter({
        "success": 95,
    }):
        raise RuntimeError(
            "DIPPER has non-success generations"
        )

    source_by_id = {
        r["machine_sample_id"]:
            r
        for r in sources
    }

    dipper_by_id = {
        r["machine_sample_id"]:
            r
        for r in dipper_rows
    }

    if len(source_by_id) != 95:
        raise RuntimeError(
            "Source machine IDs not unique"
        )

    if len(dipper_by_id) != 95:
        raise RuntimeError(
            "DIPPER machine IDs not unique"
        )

    if set(source_by_id) != set(dipper_by_id):
        raise RuntimeError(
            "Source/DIPPER membership mismatch"
        )

    selected_ids = {
        r["instance_id"]
        for r in selected
    }

    if selected_ids != set(source_by_id):
        raise RuntimeError(
            "Primary selected/source membership mismatch"
        )

    for iid in source_by_id:

        source = source_by_id[iid]
        dipper = dipper_by_id[iid]

        for field in (
            "base_id",
            "domain",
            "generator_key",
            "source_generator",
        ):

            if source[field] != dipper[field]:
                raise RuntimeError(
                    f"Source/DIPPER metadata mismatch: "
                    f"{iid}/{field}"
                )

    base_domain = {}

    for row in sources:

        bid = row["base_id"]
        dom = row["domain"]

        if (
            bid in base_domain
            and base_domain[bid] != dom
        ):
            raise RuntimeError(
                f"Base spans domains: {bid}"
            )

        base_domain[bid] = dom

    if len(base_domain) != 48:
        raise RuntimeError(
            f"Expected 48 bases, got {len(base_domain)}"
        )

    if Counter(
        base_domain.values()
    ) != Counter({
        "news": 12,
        "reddit": 12,
        "reviews": 12,
        "wiki": 12,
    }):
        raise RuntimeError(
            "Base domain structure mismatch"
        )

    print()
    print("===== MEMBERSHIP =====")
    print("sources          :", len(sources))
    print("DIPPER           :", len(dipper_rows))
    print("Vanilla + HPCD   :", len(selected))
    print("machine instances:", len(source_by_id))
    print("base clusters    :", len(base_domain))
    print("membership       : EXACT MATCH")

    # ========================================================================
    # 3. Build required Original + DIPPER text universe
    # ========================================================================

    original_required = {}

    dipper_required = {}

    required = {}

    for iid in sorted(source_by_id):

        src = source_by_id[iid]

        key = text_key(
            src["domain"],
            src["source_text"],
        )

        original_required[iid] = key

        required[key] = {
            "domain":
                src["domain"],

            "text":
                src["source_text"],
        }

        dp = dipper_by_id[iid]

        text = str(
            dp["output_text"]
        )

        if not text.strip():
            raise RuntimeError(
                f"Empty DIPPER text: {iid}"
            )

        key = text_key(
            dp["domain"],
            text,
        )

        dipper_required[iid] = key

        required[key] = {
            "domain":
                dp["domain"],

            "text":
                text,
        }

    # ========================================================================
    # 4. Existing primary cache is immutable reusable evidence
    # ========================================================================

    primary_cache = {
        r["text_key"]:
            r
        for r in primary_cache_rows
    }

    if len(primary_cache) != len(
        primary_cache_rows
    ):
        raise RuntimeError(
            "Duplicate primary cache key"
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if SUMMARY.exists():
        raise RuntimeError(
            "Final CONFIRM48 DIPPER summary already exists; "
            "refusing frozen-result overwrite"
        )

    extension_cache = {}

    if CACHE.exists():

        for r in load_jsonl(
            CACHE
        ):

            key = r["text_key"]

            if key in extension_cache:
                raise RuntimeError(
                    f"Duplicate extension cache key: {key}"
                )

            extension_cache[key] = r

    if (
        set(extension_cache)
        -
        set(required)
    ):
        raise RuntimeError(
            "Extension cache contains texts outside "
            "the frozen Original+DIPPER universe"
        )

    # Copy required scores already available in the frozen primary cache.
    reused = 0

    for key in required:

        if key in extension_cache:
            continue

        if key not in primary_cache:
            continue

        old = primary_cache[key]

        rec = {
            "text_key":
                key,

            "domain":
                old["domain"],

            "scored_tokens":
                int(
                    old["scored_tokens"]
                ),

            "nll":
                float(
                    old["nll"]
                ),

            "T0":
                float(
                    old["T0"]
                ),

            "cache_origin":
                "frozen_confirm48_primary_cache",
        }

        append_jsonl(
            CACHE,
            rec,
        )

        extension_cache[
            key
        ] = rec

        reused += 1

    remaining = [
        (key, obj)
        for key, obj in required.items()
        if key not in extension_cache
    ]

    print()
    print("===== ORIGINAL + DIPPER T0 CACHE =====")
    print("required unique texts :", len(required))
    print("reused primary cache  :", reused)
    print("cached total          :", len(extension_cache))
    print("remaining inference   :", len(remaining))

    # ========================================================================
    # 5. Score only genuinely missing Original/DIPPER texts
    # ========================================================================

    if remaining:

        if torch.backends.mps.is_available():

            device = torch.device(
                "mps"
            )

            dtype = torch.float16

        elif torch.cuda.is_available():

            device = torch.device(
                "cuda"
            )

            dtype = torch.bfloat16

        else:

            raise RuntimeError(
                "Need MPS or CUDA"
            )

        print()
        print("device :", device)
        print("dtype  :", dtype)
        print()
        print("Loading frozen Qwen2.5-3B...")

        tokenizer = (
            AutoTokenizer
            .from_pretrained(
                str(MODEL_PATH),
                local_files_only=True,
                trust_remote_code=True,
            )
        )

        model = (
            AutoModelForCausalLM
            .from_pretrained(
                str(MODEL_PATH),
                local_files_only=True,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
        )

        model.to(
            device
        )

        model.eval()

        hjtm = HJTMModel(
            HJTM_PATH
        )

        total = len(
            remaining
        )

        for i, (key, obj) in enumerate(
            remaining,
            start=1,
        ):

            token_ids = tokenizer(
                obj["text"],
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]

            result = score_tokens(
                token_ids,
                model=model,
                hjtm=hjtm,
                device=device,
                domain=obj["domain"],
                order=0,
                eos_token_id=
                    tokenizer.eos_token_id,
            )

            rec = {
                "text_key":
                    key,

                "domain":
                    obj["domain"],

                "scored_tokens":
                    int(
                        result["scored_tokens"]
                    ),

                "nll":
                    float(
                        result["nll"]
                    ),

                "T0":
                    float(
                        result["typicality"]
                    ),

                "cache_origin":
                    "confirm48_dipper_extension",
            }

            append_jsonl(
                CACHE,
                rec,
            )

            extension_cache[
                key
            ] = rec

            print(
                f"[{i:03d}/{total:03d}] "
                f"{obj['domain']:7s} "
                f"tokens={rec['scored_tokens']:4d} "
                f"T0={rec['T0']:.6f}",
                flush=True,
            )

    if set(extension_cache) != set(required):
        raise RuntimeError(
            "Final extension cache does not exactly "
            "cover required Original+DIPPER texts"
        )

    # ========================================================================
    # 6. Canonical Original + DIPPER score records
    # ========================================================================

    new_scores = []

    for iid in sorted(
        source_by_id
    ):

        src = source_by_id[iid]

        key = original_required[
            iid
        ]

        score = extension_cache[
            key
        ]

        new_scores.append({
            "instance_id":
                iid,

            "base_id":
                src["base_id"],

            "domain":
                src["domain"],

            "generator_key":
                src["generator_key"],

            "source_generator":
                src["source_generator"],

            "method":
                "original",

            "text":
                src["source_text"],

            "text_hash":
                hashlib.sha256(
                    src[
                        "source_text"
                    ].encode("utf-8")
                ).hexdigest(),

            "token_count_scored":
                int(
                    score[
                        "scored_tokens"
                    ]
                ),

            "trajectory_nll":
                float(
                    score["nll"]
                ),

            "T0":
                float(
                    score["T0"]
                ),

            "scoring_status":
                "OK",
        })

        dp = dipper_by_id[
            iid
        ]

        key = dipper_required[
            iid
        ]

        score = extension_cache[
            key
        ]

        new_scores.append({
            "instance_id":
                iid,

            "base_id":
                dp["base_id"],

            "domain":
                dp["domain"],

            "generator_key":
                dp["generator_key"],

            "source_generator":
                dp["source_generator"],

            "method":
                "dipper_l60_o60",

            "text":
                dp["output_text"],

            "text_hash":
                hashlib.sha256(
                    dp[
                        "output_text"
                    ].encode("utf-8")
                ).hexdigest(),

            "token_count_scored":
                int(
                    score[
                        "scored_tokens"
                    ]
                ),

            "trajectory_nll":
                float(
                    score["nll"]
                ),

            "T0":
                float(
                    score["T0"]
                ),

            "scoring_status":
                "OK",
        })

    if len(new_scores) != 190:
        raise RuntimeError(
            f"Expected 190 Original+DIPPER score rows, "
            f"got {len(new_scores)}"
        )

    write_jsonl(
        NEW_SCORES,
        new_scores,
    )

    # ========================================================================
    # 7. Canonicalize frozen Vanilla + HPCD scores
    # ========================================================================

    selected_canonical = []

    for row in selected:

        selected_canonical.append({
            "instance_id":
                row["instance_id"],

            "base_id":
                row["base_id"],

            "domain":
                row["domain"],

            "generator_key":
                row["generator_key"],

            "source_generator":
                row[
                    "source_generator"
                ],

            "method":
                row["condition"],

            "text":
                row["output_text"],

            "text_hash":
                hashlib.sha256(
                    row[
                        "output_text"
                    ].encode("utf-8")
                ).hexdigest(),

            "token_count_scored":
                int(
                    row[
                        "common_scored_tokens_qwen25_3b"
                    ]
                ),

            "trajectory_nll":
                float(
                    row[
                        "common_nll_qwen25_3b"
                    ]
                ),

            "T0":
                float(
                    row[
                        "common_t0_qwen25_3b"
                    ]
                ),

            "scoring_status":
                "OK",

            "itt_status":
                row["status"],
        })

    all_scores = (
        new_scores
        +
        selected_canonical
    )

    if len(all_scores) != 380:
        raise RuntimeError(
            f"Expected 380 all-method rows, "
            f"got {len(all_scores)}"
        )

    method_counts = Counter(
        r["method"]
        for r in all_scores
    )

    if method_counts != Counter({
        "original":
            95,

        "vanilla_t10":
            95,

        "hpcd_n0":
            95,

        "dipper_l60_o60":
            95,
    }):
        raise RuntimeError(
            f"All-method counts wrong: "
            f"{method_counts}"
        )

    write_jsonl(
        ALL_SCORES,
        all_scores,
    )

    score_map = {
        (
            r["instance_id"],
            r["method"],
        ):
            r

        for r in all_scores
    }

    if len(score_map) != 380:
        raise RuntimeError(
            "Duplicate all-method score key"
        )

    # ========================================================================
    # 8. Paired DIPPER contrasts
    # ========================================================================

    pair_rows = []

    for iid in sorted(
        source_by_id
    ):

        d = score_map[
            iid,
            "dipper_l60_o60",
        ]

        rec = {
            "instance_id":
                iid,

            "base_id":
                d["base_id"],

            "domain":
                d["domain"],

            "generator_key":
                d["generator_key"],

            "dipper_T0":
                float(
                    d["T0"]
                ),
        }

        for comparator in COMPARATORS:

            x = score_map[
                iid,
                comparator,
            ]

            for field in (
                "base_id",
                "domain",
                "generator_key",
            ):

                if d[field] != x[field]:
                    raise RuntimeError(
                        f"Metadata mismatch: "
                        f"{iid}/{comparator}/{field}"
                    )

            rec[
                f"{comparator}_T0"
            ] = float(
                x["T0"]
            )

            rec[
                f"delta_dipper_minus_{comparator}"
            ] = (
                rec["dipper_T0"]
                -
                rec[
                    f"{comparator}_T0"
                ]
            )

        pair_rows.append(
            rec
        )

    if len(pair_rows) != 95:
        raise RuntimeError(
            "Expected 95 paired rows"
        )

    with PAIRS.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                pair_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            pair_rows
        )

    # ========================================================================
    # 9. Collapse generator variants to the 48 RAID base IDs
    # ========================================================================

    by_base = defaultdict(
        list
    )

    for row in pair_rows:
        by_base[
            row["base_id"]
        ].append(row)

    cluster_rows = []

    for base_id, rr in sorted(
        by_base.items()
    ):

        domains = {
            x["domain"]
            for x in rr
        }

        if len(domains) != 1:
            raise RuntimeError(
                f"Base spans domains: {base_id}"
            )

        rec = {
            "base_id":
                base_id,

            "domain":
                next(iter(domains)),

            "n_generator_variants":
                len(rr),
        }

        for comparator in COMPARATORS:

            key = (
                f"delta_dipper_minus_{comparator}"
            )

            rec[
                key
            ] = float(
                np.mean([
                    x[key]
                    for x in rr
                ])
            )

        cluster_rows.append(
            rec
        )

    if len(cluster_rows) != 48:
        raise RuntimeError(
            f"Expected 48 clusters, "
            f"got {len(cluster_rows)}"
        )

    if Counter(
        r["domain"]
        for r in cluster_rows
    ) != Counter({
        "news": 12,
        "reddit": 12,
        "reviews": 12,
        "wiki": 12,
    }):
        raise RuntimeError(
            "Cluster domain structure mismatch"
        )

    if Counter(
        r["n_generator_variants"]
        for r in cluster_rows
    ) != Counter({
        2: 47,
        1: 1,
    }):
        raise RuntimeError(
            "Generator-variant structure mismatch"
        )

    with CLUSTERS.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                cluster_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            cluster_rows
        )

    # ========================================================================
    # 10. Exact historical 48-cluster bootstrap
    # ========================================================================

    comparisons = {}

    for comparator in COMPARATORS:

        delta_key = (
            f"delta_dipper_minus_{comparator}"
        )

        instance_values = np.asarray([
            r[
                delta_key
            ]
            for r in pair_rows
        ])

        boot = bootstrap_stratified(
            cluster_rows,
            delta_key,
        )

        comparisons[
            f"dipper_l60_o60 - {comparator}"
        ] = {
            "n_instances":
                95,

            "instance_mean_delta":
                float(
                    instance_values.mean()
                ),

            "instance_median_delta":
                float(
                    np.median(
                        instance_values
                    )
                ),

            "improved":
                int(
                    np.sum(
                        instance_values < 0
                    )
                ),

            "worsened":
                int(
                    np.sum(
                        instance_values > 0
                    )
                ),

            "ties":
                int(
                    np.sum(
                        instance_values == 0
                    )
                ),

            "cluster_bootstrap":
                boot,
        }

    # ========================================================================
    # 11. Absolute T0
    # ========================================================================

    absolute = {}

    for method in (
        "original",
        "vanilla_t10",
        "dipper_l60_o60",
        "hpcd_n0",
    ):

        values = np.asarray([
            score_map[
                iid,
                method,
            ]["T0"]
            for iid in sorted(
                source_by_id
            )
        ], dtype=float)

        absolute[
            method
        ] = {
            "n":
                95,

            "mean_T0":
                float(
                    values.mean()
                ),

            "median_T0":
                float(
                    np.median(values)
                ),
        }

    # ========================================================================
    # 12. Descriptive generator/domain effects
    # ========================================================================

    subgroup = {}

    for comparator in (
        "vanilla_t10",
        "hpcd_n0",
    ):

        key = (
            f"delta_dipper_minus_{comparator}"
        )

        by_generator = {}

        for generator in sorted({
            r["generator_key"]
            for r in pair_rows
        }):

            vals = [
                r[key]
                for r in pair_rows
                if (
                    r["generator_key"]
                    == generator
                )
            ]

            by_generator[
                generator
            ] = {
                "n":
                    len(vals),

                "mean_delta":
                    float(
                        np.mean(vals)
                    ),
            }

        by_domain = {}

        for domain in (
            "news",
            "reddit",
            "reviews",
            "wiki",
        ):

            vals = [
                r[key]
                for r in cluster_rows
                if (
                    r["domain"]
                    == domain
                )
            ]

            by_domain[
                domain
            ] = {
                "n_base_clusters":
                    len(vals),

                "mean_cluster_delta":
                    float(
                        np.mean(vals)
                    ),
            }

        subgroup[
            f"dipper_l60_o60 - {comparator}"
        ] = {
            "by_source_generator":
                by_generator,

            "by_domain":
                by_domain,
        }

    # ========================================================================
    # 13. Independent HPCD-vs-DIPPER replication classification
    #
    # Our contrast is DIPPER - HPCD:
    # positive => HPCD has lower T0.
    # ========================================================================

    hpcd_comparison = comparisons[
        "dipper_l60_o60 - hpcd_n0"
    ][
        "cluster_bootstrap"
    ]

    hpcd_point = float(
        hpcd_comparison[
            "point"
        ]
    )

    hpcd_ci_low = float(
        hpcd_comparison[
            "ci95"
        ][0]
    )

    hpcd_ci_high = float(
        hpcd_comparison[
            "ci95"
        ][1]
    )

    if (
        hpcd_point > 0
        and hpcd_ci_low > 0
    ):

        external_replication_status = (
            "HPCD_LOWER_T0_THAN_DIPPER"
        )

        external_replication_sentence = (
            "CONFIRM48 independently reproduces "
            "the Fresh95 external-baseline ordering: "
            "HPCD has lower T0 than DIPPER, and the "
            "95% domain-stratified base-cluster "
            "bootstrap CI excludes zero."
        )

    elif (
        hpcd_point > 0
        and hpcd_ci_low <= 0 <= hpcd_ci_high
    ):

        external_replication_status = (
            "DIRECTIONALLY_HPCD_LOWER_"
            "INCONCLUSIVE"
        )

        external_replication_sentence = (
            "CONFIRM48 is directionally consistent "
            "with HPCD having lower T0 than DIPPER, "
            "but the 95% domain-stratified "
            "base-cluster bootstrap CI includes zero."
        )

    elif hpcd_point < 0:

        external_replication_status = (
            "DIPPER_LOWER_T0_THAN_HPCD"
        )

        external_replication_sentence = (
            "CONFIRM48 does not reproduce the "
            "Fresh95 HPCD-versus-DIPPER ordering; "
            "the point estimate favors DIPPER."
        )

    else:

        external_replication_status = (
            "NO_DIFFERENCE"
        )

        external_replication_sentence = (
            "CONFIRM48 has a zero cluster-point "
            "difference between DIPPER and HPCD."
        )

    # ========================================================================
    # 14. Summary
    # ========================================================================

    summary = {
        "status":
            "COMPLETE",

        "stage":
            (
                "CONFIRM48 DIPPER L60/O60 "
                "external-baseline frozen T0 evaluation"
            ),

        "scientific_role":
            (
                "post-primary-freeze external "
                "paraphrasing baseline evaluation"
            ),

        "primary_confirm48_was_locked_before_evaluation":
            True,

        "primary_confirm48_freeze_sha256":
            sha(PRIMARY_FREEZE),

        "dipper_return_freeze_sha256":
            sha(DIPPER_RETURN_FREEZE),

        "source_sha256":
            sha(SOURCE),

        "dipper_output_sha256":
            sha(DIPPER),

        "primary_selected_scores_sha256":
            sha(PRIMARY_SELECTED_SCORES),

        "primary_cache_sha256":
            sha(PRIMARY_CACHE),

        "historical_primary_scorer_sha256":
            sha(HISTORICAL_SCORER),

        "score_tokens_helper_sha256":
            sha(SCORE_HELPER),

        "hjtm_sha256":
            sha(HJTM_PATH),

        "model":
            "Qwen2.5-3B-Instruct",

        "n_instances":
            95,

        "n_base_clusters":
            48,

        "absolute_T0":
            absolute,

        "comparisons":
            comparisons,

        "subgroups":
            subgroup,

        "external_baseline_replication_status":
            external_replication_status,

        "external_baseline_replication_interpretation":
            external_replication_sentence,

        "sign_convention":
            (
                "delta = DIPPER T0 - comparator T0; "
                "negative favors DIPPER; positive "
                "versus HPCD favors HPCD"
            ),

        "bootstrap": {
            "unit":
                "48 RAID base IDs",

            "generator_handling":
                (
                    "average available paired generator "
                    "effects within each base ID before "
                    "bootstrap"
                ),

            "stratification":
                "domain",

            "base_ids_per_domain":
                12,

            "B":
                50000,

            "seed":
                2027,

            "implementation":
                (
                    "verbatim bootstrap_stratified() "
                    "from frozen Fresh95 primary scorer"
                ),
        },

        "evaluation_rules": {
            "dipper_regeneration":
                False,

            "dipper_filtering":
                False,

            "quality_selection":
                False,

            "t0_selection":
                False,

            "detector_selection":
                False,

            "parameter_tuning":
                False,
        },

        "artifacts": {
            "cache_sha256":
                sha(CACHE),

            "new_scores_sha256":
                sha(NEW_SCORES),

            "all_scores_sha256":
                sha(ALL_SCORES),

            "pairs_sha256":
                sha(PAIRS),

            "clusters_sha256":
                sha(CLUSTERS),
        },
    }

    SUMMARY.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    # ========================================================================
    # 15. Final report
    # ========================================================================

    print()
    print("=" * 108)
    print("CONFIRM48 DIPPER PRIMARY T0 RESULT")
    print("=" * 108)

    print()
    print("ABSOLUTE MEAN T0:")

    for method in (
        "original",
        "vanilla_t10",
        "dipper_l60_o60",
        "hpcd_n0",
    ):

        print(
            f"  {method:18s}",
            f"{absolute[method]['mean_T0']:.6f}",
        )

    print()
    print("PAIRWISE CLUSTER RESULTS")
    print(
        "(delta = DIPPER - comparator)"
    )

    for name, info in comparisons.items():

        boot = info[
            "cluster_bootstrap"
        ]

        print()
        print(name)

        print(
            "  instance mean :",
            info[
                "instance_mean_delta"
            ],
        )

        print(
            "  improved/worsened/ties:",
            info["improved"],
            info["worsened"],
            info["ties"],
        )

        print(
            "  cluster point :",
            boot["point"],
        )

        print(
            "  95% CI        :",
            boot["ci95"],
        )

        print(
            "  P(effect<0)   :",
            boot[
                "p_effect_lt_0"
            ],
        )

    print()
    print("DESCRIPTIVE SUBGROUPS:")

    for name, info in subgroup.items():

        print()
        print(name)

        print(
            "  generators:",
            info[
                "by_source_generator"
            ],
        )

        print(
            "  domains   :",
            info[
                "by_domain"
            ],
        )

    print()
    print(
        "EXTERNAL BASELINE STATUS:",
        external_replication_status,
    )

    print(
        external_replication_sentence
    )

    print()
    print("=" * 108)
    print("ARTIFACT HASHES")
    print("=" * 108)

    print(
        "cache      :",
        sha(CACHE),
    )

    print(
        "new scores :",
        sha(NEW_SCORES),
    )

    print(
        "all scores :",
        sha(ALL_SCORES),
    )

    print(
        "pairs      :",
        sha(PAIRS),
    )

    print(
        "clusters   :",
        sha(CLUSTERS),
    )

    print(
        "summary    :",
        sha(SUMMARY),
    )

    print()
    print(
        "CONFIRM48 DIPPER PRIMARY T0 EVALUATION: PASS"
    )


if __name__ == "__main__":
    main()
