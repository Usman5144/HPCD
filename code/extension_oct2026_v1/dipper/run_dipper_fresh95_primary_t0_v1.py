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


DIPPER = (
    ROOT
    / "results/extension_oct2026_v1/dipper/"
      "friend_return_v1/friend_gpu_package_v1/"
      "outputs/fresh95/full/"
      "dipper_fresh95_full_final.jsonl"
)

DIPPER_FREEZE = (
    ROOT
    / "results/extension_oct2026_v1/dipper/"
      "dipper_friend_return_freeze_v1.json"
)

OLD_SCORES = (
    ROOT
    / "results/hpcd/final_copa_t0_eval_v1/"
      "primary_scores_all_methods.jsonl"
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
      "fresh95_primary_t0_v1"
)

CACHE = OUT / "dipper_t0_cache.jsonl"
DIPPER_SCORES = OUT / "dipper_primary_scores.jsonl"
ALL_SCORES = OUT / "primary_scores_with_dipper.jsonl"
PAIRS = OUT / "pairwise_instance_effects.csv"
CLUSTERS = OUT / "pairwise_cluster_effects.csv"
SUMMARY = OUT / "summary.json"


EXPECTED = {
    DIPPER:
        "7ff45d5472025ac7883a922590859dacd"
        "b5bc3d4a398c2789c1f995a05661622",

    DIPPER_FREEZE:
        "da6a1ce6de6f8f6aa8d335dea3ac7d1e"
        "289bb9d3fccf21453b635aea74d7d32d",

    OLD_SCORES:
        "f0ce714c7c2d616125e48bf7e7a1598e"
        "3ca0176a46fe393626550c6f696e0f20",

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

BASELINES = (
    "original",
    "vanilla_t10",
    "copa_3b",
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


def recover_historical_functions():

    source = HISTORICAL_SCORER.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(source)

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
            "Could not recover exact historical functions"
        )

    module = ast.Module(
        body=nodes,
        type_ignores=[],
    )
    ast.fix_missing_locations(module)

    namespace = {
        "np": np,
        "hashlib": hashlib,
        "defaultdict": defaultdict,
        "B": B,
        "BOOT_SEED": BOOT_SEED,
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

    print("=" * 104)
    print(
        "DIPPER FRESH95 — "
        "FROZEN PRIMARY QWEN2.5-3B T0"
    )
    print("=" * 104)

    print()
    print("===== FROZEN PROVENANCE =====")

    for path, expected in EXPECTED.items():

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

    freeze = json.loads(
        DIPPER_FREEZE.read_text(
            encoding="utf-8"
        )
    )

    if freeze["status"] != "FROZEN_COMPLETE":
        raise RuntimeError(
            "DIPPER return is not frozen"
        )

    print("PROVENANCE: PASS")

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

    dipper_rows = load_jsonl(DIPPER)
    old = load_jsonl(OLD_SCORES)

    if len(dipper_rows) != 95:
        raise RuntimeError(
            f"DIPPER rows={len(dipper_rows)} != 95"
        )

    if len(old) != 475:
        raise RuntimeError(
            f"Historical rows={len(old)} != 475"
        )

    if Counter(
        r["status"]
        for r in dipper_rows
    ) != Counter({"success": 95}):
        raise RuntimeError(
            "DIPPER contains non-success rows"
        )

    methods = Counter(
        r["method"]
        for r in old
    )

    expected_methods = Counter({
        "original": 95,
        "vanilla_t10": 95,
        "vanilla_t11": 95,
        "copa_3b": 95,
        "hpcd_n0": 95,
    })

    if methods != expected_methods:
        raise RuntimeError(
            f"Historical method counts wrong: {methods}"
        )

    old_instance_ids = {
        r["instance_id"]
        for r in old
    }

    dipper_ids = {
        r["machine_sample_id"]
        for r in dipper_rows
    }

    if len(dipper_ids) != 95:
        raise RuntimeError(
            "DIPPER machine_sample_id not unique"
        )

    if dipper_ids != old_instance_ids:

        missing = sorted(
            old_instance_ids - dipper_ids
        )

        extra = sorted(
            dipper_ids - old_instance_ids
        )

        raise RuntimeError(
            "DIPPER/Fresh95 membership mismatch\n"
            f"missing={missing[:5]}\n"
            f"extra={extra[:5]}"
        )

    print()
    print("===== MEMBERSHIP =====")
    print("DIPPER rows       :", len(dipper_rows))
    print("historical rows   :", len(old))
    print("machine instances :", len(dipper_ids))
    print("membership        : EXACT MATCH")

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if SUMMARY.exists():
        raise RuntimeError(
            "Final summary already exists; "
            "refusing frozen-result overwrite"
        )

    cache = {}

    if CACHE.exists():

        for row in load_jsonl(CACHE):

            key = row["text_key"]

            if key in cache:
                raise RuntimeError(
                    f"Duplicate cache key: {key}"
                )

            cache[key] = row

    needed = {}

    for row in dipper_rows:

        text = str(
            row["output_text"]
        )

        if not text.strip():
            raise RuntimeError(
                f"Empty DIPPER output: "
                f"{row['machine_sample_id']}"
            )

        key = text_key(
            row["domain"],
            text,
        )

        needed[key] = {
            "domain":
                row["domain"],

            "text":
                text,
        }

    extras = set(cache) - set(needed)

    if extras:
        raise RuntimeError(
            "T0 cache contains texts outside "
            "frozen Fresh95 DIPPER input"
        )

    remaining = [
        (key, obj)
        for key, obj in needed.items()
        if key not in cache
    ]

    print()
    print("===== DIPPER T0 CACHE =====")
    print("DIPPER rows   :", len(dipper_rows))
    print("unique texts  :", len(needed))
    print("cached        :", len(cache))
    print("remaining     :", len(remaining))

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
        print("Loading Qwen2.5-3B...")

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

        model.to(device)
        model.eval()

        hjtm = HJTMModel(
            HJTM_PATH
        )

        total = len(remaining)

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
            }

            append_jsonl(
                CACHE,
                rec,
            )

            cache[key] = rec

            print(
                f"[{i:03d}/{total:03d}] "
                f"{obj['domain']:7s} "
                f"tokens={rec['scored_tokens']:4d} "
                f"T0={rec['T0']:.6f}",
                flush=True,
            )

    scored_dipper = []

    for row in dipper_rows:

        key = text_key(
            row["domain"],
            row["output_text"],
        )

        score = cache[key]

        scored_dipper.append({
            "instance_id":
                row["machine_sample_id"],

            "base_id":
                row["base_id"],

            "domain":
                row["domain"],

            "generator_key":
                row["generator_key"],

            "source_generator":
                row["source_generator"],

            "method":
                "dipper_l60_o60",

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
                    score["scored_tokens"]
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

    write_jsonl(
        DIPPER_SCORES,
        scored_dipper,
    )

    write_jsonl(
        ALL_SCORES,
        old + scored_dipper,
    )

    old_map = {
        (
            r["instance_id"],
            r["method"],
        ):
            r
        for r in old
    }

    dipper_map = {
        r["instance_id"]:
            r
        for r in scored_dipper
    }

    for iid in sorted(dipper_ids):

        d = dipper_map[iid]

        for method in BASELINES:

            x = old_map[
                iid,
                method,
            ]

            for field in (
                "base_id",
                "domain",
                "generator_key",
            ):

                if d[field] != x[field]:
                    raise RuntimeError(
                        f"Metadata mismatch: "
                        f"{iid}/{method}/{field}"
                    )

    pair_rows = []

    for iid in sorted(dipper_ids):

        d = dipper_map[iid]

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

        for method in BASELINES:

            comparator = float(
                old_map[
                    iid,
                    method,
                ]["T0"]
            )

            rec[
                f"{method}_T0"
            ] = comparator

            rec[
                f"delta_dipper_minus_{method}"
            ] = (
                rec["dipper_T0"]
                -
                comparator
            )

        pair_rows.append(rec)

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
        writer.writerows(pair_rows)

    by_base = defaultdict(list)

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

        for method in BASELINES:

            key = (
                f"delta_dipper_minus_{method}"
            )

            rec[key] = float(
                np.mean([
                    x[key]
                    for x in rr
                ])
            )

        cluster_rows.append(rec)

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
            "Base-domain structure mismatch"
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
        writer.writerows(cluster_rows)

    comparisons = {}

    for method in BASELINES:

        delta_key = (
            f"delta_dipper_minus_{method}"
        )

        vals = np.asarray([
            r[delta_key]
            for r in pair_rows
        ])

        boot = bootstrap_stratified(
            cluster_rows,
            delta_key,
        )

        comparisons[
            f"dipper_l60_o60 - {method}"
        ] = {
            "n_instances":
                95,

            "instance_mean_delta":
                float(
                    vals.mean()
                ),

            "instance_median_delta":
                float(
                    np.median(vals)
                ),

            "improved":
                int(
                    np.sum(vals < 0)
                ),

            "worsened":
                int(
                    np.sum(vals > 0)
                ),

            "ties":
                int(
                    np.sum(vals == 0)
                ),

            "cluster_bootstrap":
                boot,
        }

    absolute = {}

    for method in BASELINES:

        values = np.asarray([
            float(
                old_map[
                    iid,
                    method,
                ]["T0"]
            )
            for iid in sorted(
                dipper_ids
            )
        ])

        absolute[method] = {
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

    dipper_values = np.asarray([
        r["T0"]
        for r in scored_dipper
    ])

    absolute[
        "dipper_l60_o60"
    ] = {
        "n":
            95,

        "mean_T0":
            float(
                dipper_values.mean()
            ),

        "median_T0":
            float(
                np.median(
                    dipper_values
                )
            ),
    }

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
                if r["domain"] == domain
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

    summary = {
        "status":
            "COMPLETE",

        "stage":
            (
                "Fresh95 DIPPER L60/O60 "
                "primary frozen Qwen2.5-3B "
                "T0 evaluation"
            ),

        "scientific_role":
            (
                "evaluation-only external "
                "paraphrasing baseline"
            ),

        "dipper_output_sha256":
            sha(DIPPER),

        "dipper_return_freeze_sha256":
            sha(DIPPER_FREEZE),

        "historical_scores_sha256":
            sha(OLD_SCORES),

        "historical_primary_scorer_sha256":
            sha(HISTORICAL_SCORER),

        "score_tokens_helper_sha256":
            sha(SCORE_HELPER),

        "hjtm_sha256":
            sha(HJTM_PATH),

        "n_instances":
            95,

        "n_base_clusters":
            48,

        "dipper_condition":
            "L60/O60",

        "absolute_T0":
            absolute,

        "comparisons":
            comparisons,

        "subgroups":
            subgroup,

        "sign_convention":
            (
                "delta = DIPPER T0 - comparator T0; "
                "negative means DIPPER has lower T0"
            ),

        "bootstrap": {
            "B":
                B,

            "seed":
                BOOT_SEED,

            "unit":
                "48 RAID base IDs",

            "stratification":
                "domain",

            "implementation":
                (
                    "verbatim historical Fresh95 "
                    "bootstrap_stratified()"
                ),
        },

        "artifacts": {
            "cache_sha256":
                sha(CACHE),

            "dipper_scores_sha256":
                sha(DIPPER_SCORES),

            "all_scores_sha256":
                sha(ALL_SCORES),

            "pairs_sha256":
                sha(PAIRS),

            "clusters_sha256":
                sha(CLUSTERS),
        },

        "no_selection":
            True,

        "no_filtering":
            True,

        "no_detector_feedback":
            True,
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

    print()
    print("=" * 104)
    print("FRESH95 DIPPER PRIMARY T0 RESULT")
    print("=" * 104)

    print()
    print("ABSOLUTE MEAN T0:")

    for method, info in absolute.items():
        print(
            f"  {method:18s}",
            f"{info['mean_T0']:.6f}",
        )

    print()
    print("PAIRWISE CLUSTER RESULTS")
    print(
        "(delta = DIPPER - comparator; "
        "negative favors DIPPER)"
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
    print("=" * 104)
    print("ARTIFACT HASHES")
    print("=" * 104)

    print(
        "cache       :",
        sha(CACHE),
    )

    print(
        "DIPPER score:",
        sha(DIPPER_SCORES),
    )

    print(
        "all scores  :",
        sha(ALL_SCORES),
    )

    print(
        "pairs       :",
        sha(PAIRS),
    )

    print(
        "clusters    :",
        sha(CLUSTERS),
    )

    print(
        "summary     :",
        sha(SUMMARY),
    )

    print()
    print(
        "FRESH95 DIPPER PRIMARY T0 EVALUATION: PASS"
    )


if __name__ == "__main__":
    main()
