#!/usr/bin/env python3
from __future__ import annotations

import os

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(os.environ.get("HPCD_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))

sys.path.insert(
    0,
    str(ROOT / "code/hpcd"),
)

from hpcd_controller_v1 import HJTMModel
from reanalyze_t2b7_common_metrics_v1 import score_tokens


INPUT = (
    ROOT
    / "results/hpcd/fresh95_surface_entity_selection_v1/"
      "selected_outputs.jsonl"
)

EXPECTED_INPUT_SHA = (
    "ed203b1b948267a201f14b6195d49cb24563999070cbcfd128de9a8c40ed879e"
)

HJTM_PATH = (
    ROOT
    / "results/hjtm/final_hjtm_v1/"
      "hjtm_final_v1.json"
)

EXPECTED_HJTM_SHA = (
    "1abd60c38a8787c4d44cf4e81ec157e"
    "af18745ce1dec28c366417030b1a22c61"
)

MODEL_PATH = Path(
    "/path/to/huggingface/hub/"
    "models--Qwen--Qwen2.5-3B-Instruct/"
    "snapshots/"
    "aa8e72537993ba99e69dfaafa59ed015b17504d1"
)

OUT = (
    ROOT
    / "results/hpcd/fresh95_primary_t0_v1"
)

CACHE = OUT / "unique_text_t0_cache.jsonl"
SCORES = OUT / "selected_outputs_t0_scores.jsonl"
PAIRS = OUT / "primary_pairs.csv"
CLUSTERS = OUT / "base_cluster_effects.csv"
SUMMARY = OUT / "summary.json"

B = 50_000
BOOT_SEED = 2027

CONDITIONS = (
    "vanilla_t10",
    "vanilla_t11",
    "hpcd_n0",
)


def sha(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
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


def text_key(domain, text):
    payload = (
        domain
        + "\0"
        + text
    ).encode("utf-8")

    return hashlib.sha256(
        payload
    ).hexdigest()


def bootstrap_stratified(
    cluster_rows,
    value_key,
):
    by_domain = defaultdict(list)

    for r in cluster_rows:
        by_domain[
            r["domain"]
        ].append(
            float(r[value_key])
        )

    assert set(by_domain) == {
        "news",
        "reddit",
        "reviews",
        "wiki",
    }

    # Fresh48 = 12 base IDs/domain.
    assert all(
        len(v) == 12
        for v in by_domain.values()
    )

    point = float(
        np.mean([
            r[value_key]
            for r in cluster_rows
        ])
    )

    rng = np.random.default_rng(
        BOOT_SEED
    )

    boots = np.empty(
        B,
        dtype=np.float64,
    )

    domains = [
        "news",
        "reddit",
        "reviews",
        "wiki",
    ]

    arrays = {
        d: np.asarray(
            by_domain[d],
            dtype=np.float64,
        )
        for d in domains
    }

    for b in range(B):
        domain_means = []

        for d in domains:
            a = arrays[d]

            sample = rng.choice(
                a,
                size=len(a),
                replace=True,
            )

            domain_means.append(
                float(sample.mean())
            )

        boots[b] = float(
            np.mean(domain_means)
        )

    lo, hi = np.quantile(
        boots,
        [0.025, 0.975],
    )

    return {
        "point":
            point,

        "ci95":
            [
                float(lo),
                float(hi),
            ],

        "p_effect_lt_0":
            float(
                np.mean(
                    boots < 0
                )
            ),

        "bootstrap_B":
            B,

        "bootstrap_seed":
            BOOT_SEED,
    }


print("=" * 88)
print("FRESH95 PRIMARY QWEN2.5-3B COMMON T0")
print("=" * 88)

assert sha(INPUT) == EXPECTED_INPUT_SHA
assert sha(HJTM_PATH) == EXPECTED_HJTM_SHA
assert MODEL_PATH.exists()

rows = load_jsonl(INPUT)

assert len(rows) == 285

assert Counter(
    r["condition"]
    for r in rows
) == Counter({
    "vanilla_t10": 95,
    "vanilla_t11": 95,
    "hpcd_n0": 95,
})

assert len({
    r["instance_id"]
    for r in rows
}) == 95

assert len({
    r["sample_id"]
    for r in rows
}) == 48

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

# ------------------------------------------------------------
# Resume-safe common T0 cache.
# Same domain+text gets exactly the same score, regardless
# of condition. This is computational caching only.
# ------------------------------------------------------------

cache = {}

if CACHE.exists():
    for r in load_jsonl(CACHE):
        cache[
            r["text_key"]
        ] = r

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

remaining = [
    (k, v)
    for k, v in needed.items()
    if k not in cache
]

print(
    "selected rows :",
    len(rows),
)

print(
    "unique texts  :",
    len(needed),
)

print(
    "cached        :",
    len(cache),
)

print(
    "remaining     :",
    len(remaining),
)

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

    print(
        "device        :",
        device,
    )

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


# ------------------------------------------------------------
# Attach common T0 to all selected ITT outputs.
# ------------------------------------------------------------

scored = []

for r in rows:
    key = text_key(
        r["domain"],
        r["output_text"],
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

score_map = {
    (
        r["instance_id"],
        r["condition"],
    ):
    r
    for r in scored
}

assert len(score_map) == 285


# ------------------------------------------------------------
# Instance-level paired ITT effects.
# Negative = HPCD closer to human T0 target.
# ------------------------------------------------------------

pair_rows = []

instances = sorted({
    r["instance_id"]
    for r in scored
})

for iid in instances:

    v10 = score_map[
        iid,
        "vanilla_t10",
    ]

    v11 = score_map[
        iid,
        "vanilla_t11",
    ]

    h = score_map[
        iid,
        "hpcd_n0",
    ]

    rec = {
        "instance_id":
            iid,

        "base_id":
            h["sample_id"],

        "generator_key":
            h["generator_key"],

        "domain":
            h["domain"],

        "vanilla_t10_status":
            v10["status"],

        "vanilla_t11_status":
            v11["status"],

        "hpcd_n0_status":
            h["status"],

        "vanilla_t10_T0":
            float(
                v10[
                    "common_t0_qwen25_3b"
                ]
            ),

        "vanilla_t11_T0":
            float(
                v11[
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

    rec[
        "delta_hpcd_minus_vanilla_t11"
    ] = (
        rec["hpcd_n0_T0"]
        -
        rec["vanilla_t11_T0"]
    )

    rec[
        "delta_vanilla_t11_minus_t10"
    ] = (
        rec["vanilla_t11_T0"]
        -
        rec["vanilla_t10_T0"]
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


# ------------------------------------------------------------
# Collapse generator variants within each RAID base cluster.
# ------------------------------------------------------------

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

    assert len(domains) == 1

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

        "delta_hpcd_minus_vanilla_t11":
            float(
                np.mean([
                    x[
                        "delta_hpcd_minus_vanilla_t11"
                    ]
                    for x in rr
                ])
            ),

        "delta_vanilla_t11_minus_t10":
            float(
                np.mean([
                    x[
                        "delta_vanilla_t11_minus_t10"
                    ]
                    for x in rr
                ])
            ),
    })

assert len(cluster_rows) == 48

assert Counter(
    r["domain"]
    for r in cluster_rows
) == Counter({
    "news": 12,
    "reddit": 12,
    "reviews": 12,
    "wiki": 12,
})


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


primary_boot = bootstrap_stratified(
    cluster_rows,
    "delta_hpcd_minus_vanilla_t10",
)

temp_boot = bootstrap_stratified(
    cluster_rows,
    "delta_hpcd_minus_vanilla_t11",
)

t11_boot = bootstrap_stratified(
    cluster_rows,
    "delta_vanilla_t11_minus_t10",
)


# ------------------------------------------------------------
# Descriptives
# ------------------------------------------------------------

primary_deltas = np.asarray([
    r[
        "delta_hpcd_minus_vanilla_t10"
    ]
    for r in pair_rows
])

per_generator = {}

for g in sorted({
    r["generator_key"]
    for r in pair_rows
}):
    vals = [
        r[
            "delta_hpcd_minus_vanilla_t10"
        ]
        for r in pair_rows
        if r["generator_key"] == g
    ]

    per_generator[g] = {
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

for d in [
    "news",
    "reddit",
    "reviews",
    "wiki",
]:
    vals = [
        r[
            "delta_hpcd_minus_vanilla_t10"
        ]
        for r in cluster_rows
        if r["domain"] == d
    ]

    per_domain[d] = {
        "n_base_clusters":
            len(vals),

        "mean_cluster_delta":
            float(
                np.mean(vals)
            ),
    }


coverage = {}

for c in CONDITIONS:
    rr = [
        r
        for r in rows
        if r["condition"] == c
    ]

    rewrites = sum(
        r["status"] == "REWRITE"
        for r in rr
    )

    coverage[c] = {
        "rewrites":
            rewrites,

        "abstentions":
            95 - rewrites,

        "coverage":
            rewrites / 95.0,
    }


summary = {
    "status":
        "COMPLETE",

    "stage":
        "Fresh95 primary common T0 evaluation",

    "input_sha256":
        sha(INPUT),

    "scores_sha256":
        sha(SCORES),

    "model":
        "Qwen2.5-3B-Instruct",

    "hjtm_sha256":
        sha(HJTM_PATH),

    "n_selected_rows":
        285,

    "n_instances":
        95,

    "n_base_clusters":
        48,

    "coverage":
        coverage,

    "primary_instance_level": {
        "comparison":
            "hpcd_n0 - vanilla_t10",

        "n":
            len(pair_rows),

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

    "hpcd_vs_temperature_t11_cluster_bootstrap":
        temp_boot,

    "temperature_t11_vs_t10_cluster_bootstrap":
        t11_boot,

    "per_generator_primary":
        per_generator,

    "per_domain_primary":
        per_domain,

    "interpretation":
        "Negative delta means HPCD output is closer to the human T0 typicality target.",

    "primary_pass_rule":
        "point < 0 and 95% CI upper bound < 0",
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
print("=" * 88)
print("PRIMARY 3B RESULT")
print("=" * 88)

print(
    "instance n:",
    len(pair_rows),
)

print(
    "mean delta HPCD-Vanilla:",
    summary[
        "primary_instance_level"
    ][
        "mean_delta"
    ],
)

print(
    "median delta:",
    summary[
        "primary_instance_level"
    ][
        "median_delta"
    ],
)

print(
    "improved/worsened/ties:",
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
    "CLUSTER POINT:",
    primary_boot["point"],
)

print(
    "95% CI:",
    primary_boot["ci95"],
)

print(
    "p(effect<0):",
    primary_boot[
        "p_effect_lt_0"
    ],
)

print()
print("PER GENERATOR:")
for k, v in per_generator.items():
    print(
        " ",
        k,
        v,
    )

print()
print("PER DOMAIN:")
for k, v in per_domain.items():
    print(
        " ",
        k,
        v,
    )

print()
print("TEMPERATURE CONTROL:")
print(
    " HPCD - T1.1:",
    temp_boot,
)

print(
    " T1.1 - T1.0:",
    t11_boot,
)

print()
print(
    "SCORES SHA:",
    sha(SCORES),
)

print(
    "SUMMARY SHA:",
    sha(SUMMARY),
)

strong_pass = (
    primary_boot["point"] < 0
    and
    primary_boot["ci95"][1] < 0
)

print()
print(
    "PRIMARY 3B GATE:",
    "PASS"
    if strong_pass
    else "FAIL",
)
