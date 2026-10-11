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


# =====================================================================
# FROZEN PATHS
# =====================================================================

ROOT = Path(os.environ.get("HPCD_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))

sys.path.insert(
    0,
    str(ROOT / "code/hpcd"),
)

from hpcd_controller_v1 import HJTMModel
from reanalyze_t2b7_common_metrics_v1 import score_tokens


INPUT = (
    ROOT
    / "results/confirm48_v1/process/"
      "confirm48_selected_itt_v1.jsonl"
)

SELECTION_FREEZE = (
    ROOT
    / "results/confirm48_v1/manifests/"
      "confirm48_quality_selection_freeze_v1.json"
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
    / "results/confirm48_v1/process/"
      "confirm48_primary_t0_v1"
)

CACHE = OUT / "unique_text_t0_cache.jsonl"
SCORES = OUT / "selected_outputs_t0_scores.jsonl"
PAIRS = OUT / "primary_pairs.csv"
CLUSTERS = OUT / "base_cluster_effects.csv"
SUMMARY = OUT / "summary.json"


# =====================================================================
# IMMUTABLE HASHES
# =====================================================================

EXPECTED = {
    INPUT:
        "6e1beccdb415279fa6e4ff25a2f983f6"
        "9fd725a2deee87ecb21936fb49aaf06e",

    SELECTION_FREEZE:
        "2cbf81902e7d47a3197bf9522fea3a32"
        "65e9c3eb85d9975d2eddc16b8ceece0f",

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

CONDITIONS = (
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


# =====================================================================
# EXTRACT EXACT HISTORICAL FUNCTIONS VERBATIM
#
# We do NOT rewrite:
#   - text_key()
#   - bootstrap_stratified()
#
# Their AST nodes are compiled directly from the frozen historical
# scorer whose SHA is checked above.
# =====================================================================

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
            isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            )
            and node.name in wanted
        )
    ]

    names = {
        node.name
        for node in nodes
    }

    if names != wanted:
        raise RuntimeError(
            "Could not recover exact historical "
            f"functions; found={names}"
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


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("=" * 104)
    print(
        "CONFIRM48 PHASE 8B — "
        "PRIMARY QWEN2.5-3B FROZEN T0"
    )
    print("=" * 104)

    # -----------------------------------------------------------------
    # 1. Provenance
    # -----------------------------------------------------------------

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
                f"SHA mismatch: {path}\n"
                f"expected={expected}\n"
                f"actual={actual}"
            )

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Missing frozen model: {MODEL_PATH}"
        )

    print("PROVENANCE: PASS")

    # -----------------------------------------------------------------
    # 2. Recover exact historical utility/statistical functions
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # 3. Frozen CONFIRM48 input validation
    # -----------------------------------------------------------------

    rows = load_jsonl(INPUT)

    if len(rows) != 190:
        raise RuntimeError(
            f"Expected 190 rows, got {len(rows)}"
        )

    counts = Counter(
        r["condition"]
        for r in rows
    )

    if counts != Counter({
        "vanilla_t10": 95,
        "hpcd_n0": 95,
    }):
        raise RuntimeError(
            f"Condition mismatch: {counts}"
        )

    instances = {
        r["instance_id"]
        for r in rows
    }

    if len(instances) != 95:
        raise RuntimeError(
            f"Expected 95 instances, got "
            f"{len(instances)}"
        )

    base_ids = {
        r["base_id"]
        for r in rows
    }

    if len(base_ids) != 48:
        raise RuntimeError(
            f"Expected 48 base IDs, got "
            f"{len(base_ids)}"
        )

    grid = Counter(
        (
            r["instance_id"],
            r["condition"],
        )
        for r in rows
    )

    if (
        len(grid) != 190
        or any(x != 1 for x in grid.values())
    ):
        raise RuntimeError(
            "Selected ITT grid is not 95 × 2"
        )

    status_counts = Counter(
        (
            r["condition"],
            r["status"],
        )
        for r in rows
    )

    expected_status = Counter({
        ("vanilla_t10", "REWRITE"):
            67,

        ("vanilla_t10", "ABSTAIN"):
            28,

        ("hpcd_n0", "REWRITE"):
            69,

        ("hpcd_n0", "ABSTAIN"):
            26,
    })

    if status_counts != expected_status:
        raise RuntimeError(
            f"Frozen coverage mismatch: "
            f"{status_counts}"
        )

    base_domain = {}

    for row in rows:

        bid = row["base_id"]
        dom = row["domain"]

        if (
            bid in base_domain
            and base_domain[bid] != dom
        ):
            raise RuntimeError(
                f"Base ID spans domains: {bid}"
            )

        base_domain[bid] = dom

    base_domain_counts = Counter(
        base_domain.values()
    )

    if base_domain_counts != Counter({
        "news": 12,
        "reddit": 12,
        "reviews": 12,
        "wiki": 12,
    }):
        raise RuntimeError(
            "Base-domain structure mismatch: "
            f"{base_domain_counts}"
        )

    print()
    print("===== CONFIRM48 STRUCTURE =====")
    print("rows           :", len(rows))
    print("instances      :", len(instances))
    print("base clusters  :", len(base_ids))
    print("conditions     :", dict(counts))
    print("status         :", dict(status_counts))
    print("base domains   :", dict(base_domain_counts))

    # -----------------------------------------------------------------
    # 4. Output / resume rules
    # -----------------------------------------------------------------

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if SUMMARY.exists():
        raise RuntimeError(
            "Final summary already exists. "
            "Refusing to rerun frozen evaluation."
        )

    # -----------------------------------------------------------------
    # 5. Common unique-text cache
    #
    # Exact same domain+text keying used historically.
    # -----------------------------------------------------------------

    cache = {}

    if CACHE.exists():

        for r in load_jsonl(CACHE):

            key = r["text_key"]

            if key in cache:
                raise RuntimeError(
                    f"Duplicate cache key: {key}"
                )

            cache[key] = r

    needed = {}

    for r in rows:

        key = text_key(
            r["domain"],
            r["output_text"],
        )

        needed[key] = {
            "domain":
                r["domain"],

            "text":
                r["output_text"],
        }

    extras = (
        set(cache)
        -
        set(needed)
    )

    if extras:
        raise RuntimeError(
            "Cache contains keys outside frozen "
            "CONFIRM48 input. Refusing contamination."
        )

    remaining = [
        (key, obj)
        for key, obj in needed.items()
        if key not in cache
    ]

    print()
    print("===== T0 CACHE =====")
    print("selected rows :", len(rows))
    print("unique texts  :", len(needed))
    print("cached        :", len(cache))
    print("remaining     :", len(remaining))

    # -----------------------------------------------------------------
    # 6. Exact historical primary model scoring
    # -----------------------------------------------------------------

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
                "Need MPS or CUDA for frozen "
                "Qwen2.5-3B scorer."
            )

        print()
        print("device        :", device)
        print("dtype         :", dtype)
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

            # CRITICAL:
            # This is the EXACT historical score_tokens().
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
                        result[
                            "scored_tokens"
                        ]
                    ),

                "nll":
                    float(
                        result["nll"]
                    ),

                "T0":
                    float(
                        result[
                            "typicality"
                        ]
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

        del model

        if (
            device.type == "mps"
            and hasattr(torch.mps, "empty_cache")
        ):
            torch.mps.empty_cache()

    # -----------------------------------------------------------------
    # 7. Attach common T0 to all 190 frozen ITT outputs
    # -----------------------------------------------------------------

    scored = []

    for r in rows:

        key = text_key(
            r["domain"],
            r["output_text"],
        )

        if key not in cache:
            raise RuntimeError(
                f"Missing score cache key: {key}"
            )

        c = cache[key]

        rec = dict(r)

        rec[
            "common_t0_qwen25_3b"
        ] = float(
            c["T0"]
        )

        rec[
            "common_nll_qwen25_3b"
        ] = float(
            c["nll"]
        )

        rec[
            "common_scored_tokens_qwen25_3b"
        ] = int(
            c["scored_tokens"]
        )

        scored.append(rec)

    write_jsonl(
        SCORES,
        scored,
    )

    # -----------------------------------------------------------------
    # 8. Exact paired ITT comparison
    # -----------------------------------------------------------------

    score_map = {
        (
            r["instance_id"],
            r["condition"],
        ):
            r

        for r in scored
    }

    if len(score_map) != 190:
        raise RuntimeError(
            "Scored 95 × 2 grid incomplete"
        )

    pair_rows = []

    for iid in sorted(instances):

        v = score_map[
            iid,
            "vanilla_t10",
        ]

        h = score_map[
            iid,
            "hpcd_n0",
        ]

        for field in (
            "base_id",
            "domain",
            "generator_key",
        ):
            if v[field] != h[field]:
                raise RuntimeError(
                    f"Pair metadata mismatch "
                    f"{iid}/{field}"
                )

        rec = {
            "instance_id":
                iid,

            "base_id":
                h["base_id"],

            "generator_key":
                h["generator_key"],

            "source_generator":
                h.get(
                    "source_generator"
                ),

            "domain":
                h["domain"],

            "vanilla_t10_status":
                v["status"],

            "hpcd_n0_status":
                h["status"],

            "vanilla_t10_T0":
                float(
                    v[
                        "common_t0_qwen25_3b"
                    ]
                ),

            "hpcd_n0_T0":
                float(
                    h[
                        "common_t0_qwen25_3b"
                    ]
                ),
        }

        rec[
            "delta_hpcd_minus_vanilla_t10"
        ] = (
            rec["hpcd_n0_T0"]
            -
            rec["vanilla_t10_T0"]
        )

        pair_rows.append(rec)

    if len(pair_rows) != 95:
        raise RuntimeError(
            f"Expected 95 pairs, got "
            f"{len(pair_rows)}"
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
        writer.writerows(pair_rows)

    # -----------------------------------------------------------------
    # 9. Correct experimental unit:
    #    collapse available generator variants inside each base ID
    # -----------------------------------------------------------------

    by_base = defaultdict(list)

    for r in pair_rows:
        by_base[
            r["base_id"]
        ].append(r)

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
                f"Base cluster spans domains: "
                f"{base_id}"
            )

        cluster_rows.append({
            "base_id":
                base_id,

            "domain":
                next(iter(domains)),

            "n_generator_variants":
                len(rr),

            "delta_hpcd_minus_vanilla_t10":
                float(
                    np.mean([
                        x[
                            "delta_hpcd_minus_vanilla_t10"
                        ]
                        for x in rr
                    ])
                ),
        })

    if len(cluster_rows) != 48:
        raise RuntimeError(
            f"Expected 48 base clusters, got "
            f"{len(cluster_rows)}"
        )

    cluster_domains = Counter(
        r["domain"]
        for r in cluster_rows
    )

    if cluster_domains != Counter({
        "news": 12,
        "reddit": 12,
        "reviews": 12,
        "wiki": 12,
    }):
        raise RuntimeError(
            f"Bad cluster domains: "
            f"{cluster_domains}"
        )

    variant_counts = Counter(
        r["n_generator_variants"]
        for r in cluster_rows
    )

    # 95 source instances across 48 bases:
    # 47 bases × 2 generators + 1 base × 1 generator.
    if variant_counts != Counter({
        2: 47,
        1: 1,
    }):
        raise RuntimeError(
            f"Unexpected generator-variant "
            f"structure: {variant_counts}"
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

    # -----------------------------------------------------------------
    # 10. EXACT historical domain-stratified bootstrap
    # -----------------------------------------------------------------

    primary_boot = bootstrap_stratified(
        cluster_rows,
        "delta_hpcd_minus_vanilla_t10",
    )

    # -----------------------------------------------------------------
    # 11. Descriptives
    # -----------------------------------------------------------------

    primary_deltas = np.asarray([
        r[
            "delta_hpcd_minus_vanilla_t10"
        ]
        for r in pair_rows
    ])

    vanilla_t0 = np.asarray([
        r["vanilla_t10_T0"]
        for r in pair_rows
    ])

    hpcd_t0 = np.asarray([
        r["hpcd_n0_T0"]
        for r in pair_rows
    ])

    per_generator = {}

    for generator in sorted({
        r["generator_key"]
        for r in pair_rows
    }):

        vals = [
            r[
                "delta_hpcd_minus_vanilla_t10"
            ]
            for r in pair_rows
            if (
                r["generator_key"]
                == generator
            )
        ]

        per_generator[generator] = {
            "n":
                len(vals),

            "mean_delta":
                float(
                    np.mean(vals)
                ),

            "median_delta":
                float(
                    np.median(vals)
                ),

            "improved":
                int(
                    sum(x < 0 for x in vals)
                ),

            "worsened":
                int(
                    sum(x > 0 for x in vals)
                ),

            "ties":
                int(
                    sum(x == 0 for x in vals)
                ),
        }

    per_domain = {}

    for domain in (
        "news",
        "reddit",
        "reviews",
        "wiki",
    ):

        vals = [
            r[
                "delta_hpcd_minus_vanilla_t10"
            ]
            for r in cluster_rows
            if r["domain"] == domain
        ]

        per_domain[domain] = {
            "n_base_clusters":
                len(vals),

            "mean_cluster_delta":
                float(
                    np.mean(vals)
                ),
        }

    coverage = {}

    for condition in CONDITIONS:

        rr = [
            r
            for r in rows
            if r["condition"] == condition
        ]

        rewrites = sum(
            r["status"] == "REWRITE"
            for r in rr
        )

        coverage[condition] = {
            "rewrites":
                rewrites,

            "abstentions":
                95 - rewrites,

            "coverage":
                rewrites / 95.0,
        }

    # -----------------------------------------------------------------
    # 12. Confirmatory interpretation — protocol-fixed
    # -----------------------------------------------------------------

    point = float(
        primary_boot["point"]
    )

    ci_low = float(
        primary_boot["ci95"][0]
    )

    ci_high = float(
        primary_boot["ci95"][1]
    )

    if (
        point < 0
        and ci_high < 0
    ):
        interpretation_class = (
            "REPRODUCED"
        )

        interpretation_sentence = (
            "CONFIRM48 independently reproduces "
            "the preregistered direction: HPCD "
            "has lower T0 than Vanilla, and the "
            "95% domain-stratified cluster "
            "bootstrap CI excludes zero."
        )

    elif (
        point < 0
        and ci_low <= 0 <= ci_high
    ):
        interpretation_class = (
            "DIRECTIONALLY_CONSISTENT_"
            "INCONCLUSIVE"
        )

        interpretation_sentence = (
            "CONFIRM48 is directionally "
            "consistent with the preregistered "
            "effect, but the 95% "
            "domain-stratified cluster bootstrap "
            "CI includes zero."
        )

    elif point > 0:

        interpretation_class = (
            "DOES_NOT_REPRODUCE"
        )

        interpretation_sentence = (
            "CONFIRM48 does not reproduce the "
            "preregistered direction: the pooled "
            "cluster point estimate favors "
            "Vanilla rather than HPCD."
        )

    else:

        interpretation_class = (
            "NO_POOLED_DIFFERENCE"
        )

        interpretation_sentence = (
            "CONFIRM48 has a pooled cluster "
            "point estimate of zero."
        )

    # -----------------------------------------------------------------
    # 13. Summary
    # -----------------------------------------------------------------

    summary = {
        "status":
            "COMPLETE",

        "stage":
            (
                "CONFIRM48 primary frozen "
                "Qwen2.5-3B common T0 evaluation"
            ),

        "confirmatory_status":
            interpretation_class,

        "confirmatory_interpretation":
            interpretation_sentence,

        "input_sha256":
            sha(INPUT),

        "selection_freeze_sha256":
            sha(SELECTION_FREEZE),

        "historical_scorer_sha256":
            sha(HISTORICAL_SCORER),

        "score_tokens_helper_sha256":
            sha(SCORE_HELPER),

        "controller_sha256":
            sha(CONTROLLER),

        "hjtm_sha256":
            sha(HJTM_PATH),

        "model":
            "Qwen2.5-3B-Instruct",

        "model_path":
            str(MODEL_PATH),

        "historical_functions_reused": [
            "text_key",
            "bootstrap_stratified",
        ],

        "historical_scientific_function_reused":
            "score_tokens",

        "scoring": {
            "add_special_tokens":
                False,

            "truncation":
                False,

            "hjtm_order":
                0,

            "eos_handling":
                "historical score_tokens",

            "text_cache_key":
                "historical domain+text SHA",

            "detector_feedback":
                False,
        },

        "bootstrap": {
            "unit":
                "48 RAID base IDs",

            "generator_handling":
                (
                    "average all available paired "
                    "generator effects within each "
                    "base ID before bootstrap"
                ),

            "stratification":
                "domain",

            "base_ids_per_domain":
                12,

            "B":
                B,

            "seed":
                BOOT_SEED,

            "implementation":
                (
                    "verbatim bootstrap_stratified "
                    "from frozen Fresh95 scorer"
                ),
        },

        "n_selected_rows":
            190,

        "n_instances":
            95,

        "n_base_clusters":
            48,

        "cluster_generator_variant_counts":
            dict(
                sorted(
                    variant_counts.items()
                )
            ),

        "coverage":
            coverage,

        "absolute_T0": {
            "vanilla_t10": {
                "n":
                    95,

                "mean":
                    float(
                        vanilla_t0.mean()
                    ),

                "median":
                    float(
                        np.median(
                            vanilla_t0
                        )
                    ),
            },

            "hpcd_n0": {
                "n":
                    95,

                "mean":
                    float(
                        hpcd_t0.mean()
                    ),

                "median":
                    float(
                        np.median(
                            hpcd_t0
                        )
                    ),
            },
        },

        "primary_instance_level": {
            "comparison":
                "hpcd_n0 - vanilla_t10",

            "n":
                95,

            "mean_delta":
                float(
                    primary_deltas.mean()
                ),

            "median_delta":
                float(
                    np.median(
                        primary_deltas
                    )
                ),

            "improved":
                int(
                    np.sum(
                        primary_deltas < 0
                    )
                ),

            "worsened":
                int(
                    np.sum(
                        primary_deltas > 0
                    )
                ),

            "ties":
                int(
                    np.sum(
                        primary_deltas == 0
                    )
                ),
        },

        "primary_domain_stratified_cluster_bootstrap":
            primary_boot,

        "per_generator_primary":
            per_generator,

        "per_domain_primary":
            per_domain,

        "sign_convention":
            (
                "delta = HPCD T0 - Vanilla T0; "
                "negative favors HPCD"
            ),

        "scores_sha256":
            sha(SCORES),

        "pairs_sha256":
            sha(PAIRS),

        "clusters_sha256":
            sha(CLUSTERS),

        "cache_sha256":
            sha(CACHE),
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

    # -----------------------------------------------------------------
    # 14. Final report
    # -----------------------------------------------------------------

    print()
    print("=" * 104)
    print("CONFIRM48 PRIMARY 3B RESULT")
    print("=" * 104)

    print(
        "Vanilla mean T0 :",
        summary[
            "absolute_T0"
        ][
            "vanilla_t10"
        ][
            "mean"
        ],
    )

    print(
        "HPCD mean T0    :",
        summary[
            "absolute_T0"
        ][
            "hpcd_n0"
        ][
            "mean"
        ],
    )

    print()
    print(
        "instance mean delta HPCD-Vanilla:",
        summary[
            "primary_instance_level"
        ][
            "mean_delta"
        ],
    )

    print(
        "instance median delta:",
        summary[
            "primary_instance_level"
        ][
            "median_delta"
        ],
    )

    print(
        "improved / worsened / ties:",
        summary[
            "primary_instance_level"
        ][
            "improved"
        ],
        summary[
            "primary_instance_level"
        ][
            "worsened"
        ],
        summary[
            "primary_instance_level"
        ][
            "ties"
        ],
    )

    print()
    print(
        "CLUSTER POINT :",
        primary_boot["point"],
    )

    print(
        "95% CI        :",
        primary_boot["ci95"],
    )

    print(
        "P(effect < 0):",
        primary_boot[
            "p_effect_lt_0"
        ],
    )

    print()
    print("PER GENERATOR:")

    for key, value in per_generator.items():
        print(
            " ",
            key,
            value,
        )

    print()
    print("PER DOMAIN:")

    for key, value in per_domain.items():
        print(
            " ",
            key,
            value,
        )

    print()
    print(
        "CONFIRMATORY STATUS:",
        interpretation_class,
    )

    print(
        interpretation_sentence
    )

    print()
    print("=" * 104)
    print("ARTIFACT HASHES")
    print("=" * 104)

    print(
        "cache    :",
        sha(CACHE),
    )

    print(
        "scores   :",
        sha(SCORES),
    )

    print(
        "pairs    :",
        sha(PAIRS),
    )

    print(
        "clusters :",
        sha(CLUSTERS),
    )

    print(
        "summary  :",
        sha(SUMMARY),
    )

    print()
    print(
        "CONFIRM48 PRIMARY T0 EVALUATION: PASS"
    )


if __name__ == "__main__":
    main()
