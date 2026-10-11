#!/usr/bin/env python3

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

RAW = (
    ROOT
    / "results/hpcd/"
      "t2c4_dev111_threegen_candidates_v3/"
      "candidates.jsonl"
)

SCORED = (
    ROOT
    / "results/hpcd/"
      "t2c4_dev111_quality_selection_v1/"
      "scored_candidates.jsonl"
)

RESCORED = (
    ROOT
    / "results/hpcd/"
      "t2c4_dev111_surface_entity_selection_v2/"
      "rescored_candidates.jsonl"
)

SELECTED = (
    ROOT
    / "results/hpcd/"
      "t2c4_dev111_surface_entity_selection_v2/"
      "selected_outputs.jsonl"
)

MANIFEST = (
    ROOT
    / "results/hpcd/"
      "t2c4_dev111_threegen_manifest_v3.json"
)

COMMON = (
    ROOT
    / "results/hpcd/"
      "t2c4_dev111_common_rescore_v1.json"
)

OUT = (
    ROOT
    / "results/hpcd/factorial_dev111"
)

FIG_DIR = (
    ROOT
    / "figures/hpcd"
)

PAIR_CSV = OUT / "candidate_pair_analysis.csv"
FOUR_CSV = OUT / "four_condition_outputs.csv"
SUMMARY_JSON = OUT / "factorial_summary.json"
BOOT_CSV = OUT / "cluster_bootstrap.csv"
AUDIT_JSON = OUT / "artifact_audit.json"

FIG_PDF = (
    FIG_DIR
    / "decoding_vs_selection_factorial.pdf"
)


EXPECTED_RAW_SHA = (
    "9027ef59f6c1cb3dc70e1f6155871ba"
    "46b3fdedf9dd4df0ddaa46d101763eb07"
)

EXPECTED_SCORED_SHA = (
    "5ca94afa38220a1d6b02df5b02708127"
    "7231b15f4756bc10c90c578cc88fd1e0"
)

EXPECTED_COMMON_SHA = (
    "2510585baacc293959677d85089cd02ba"
    "49d337f4a651ae63de533eb89726c46"
)

EXPECTED_RAW_ROWS = 7992
EXPECTED_INSTANCES = 333
EXPECTED_BASE_IDS = 111
EXPECTED_PER_CONDITION = 2664
EXPECTED_PER_INSTANCE_CONDITION = 8


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def load_jsonl(path: Path):
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                rows.append(
                    json.loads(line)
                )
            except Exception as exc:
                raise RuntimeError(
                    f"{path}: bad JSON line {ln}: {exc}"
                )

    return rows


def load_manifest(path: Path):
    obj = json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )

    if isinstance(obj, list):
        rows = obj
    elif isinstance(obj, dict):
        candidates = []

        for value in obj.values():
            if (
                isinstance(value, list)
                and value
                and isinstance(value[0], dict)
            ):
                score = sum(
                    "instance_id" in r
                    for r in value
                    if isinstance(r, dict)
                )

                candidates.append(
                    (score, value)
                )

        if not candidates:
            raise RuntimeError(
                "Could not locate document list "
                "inside manifest JSON."
            )

        candidates.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        rows = candidates[0][1]

    else:
        raise RuntimeError(
            "Unexpected manifest structure."
        )

    rows = [
        r
        for r in rows
        if isinstance(r, dict)
        and "instance_id" in r
    ]

    return rows


def first_present(row, names):
    for name in names:
        if name in row:
            return name

    return None


def resolve_field(
    rows,
    exact_names,
    contains_all=(),
    contains_any=(),
    type_check=None,
    label="field",
):
    keys = set()

    for row in rows[:100]:
        keys.update(row.keys())

    for name in exact_names:
        if name in keys:
            return name

    candidates = []

    for key in sorted(keys):
        low = key.lower()

        if contains_all and not all(
            x.lower() in low
            for x in contains_all
        ):
            continue

        if contains_any and not any(
            x.lower() in low
            for x in contains_any
        ):
            continue

        if type_check is not None:
            vals = [
                r.get(key)
                for r in rows[:100]
                if r.get(key) is not None
            ]

            if vals and not all(
                type_check(v)
                for v in vals[:20]
            ):
                continue

        candidates.append(key)

    if len(candidates) == 1:
        return candidates[0]

    raise RuntimeError(
        f"Cannot uniquely resolve {label}. "
        f"Candidates={candidates}. "
        f"Available keys={sorted(keys)}"
    )


def instance_id(row):
    value = row.get("instance_id")

    if value is not None:
        return str(value)

    g = row.get("generator_key")
    sid = row.get("sample_id")

    if (
        g is not None
        and sid is not None
    ):
        return f"{g}::{sid}"

    raise RuntimeError(
        "Cannot resolve instance_id."
    )


def base_id(row):
    sid = row.get("sample_id")

    if sid is not None:
        return str(sid)

    iid = instance_id(row)

    if "::" in iid:
        return iid.split("::", 1)[1]

    raise RuntimeError(
        f"Cannot resolve base sample ID: {iid}"
    )


def condition(row):
    value = row.get("condition")

    if value is None:
        value = row.get("label")

    if value is None:
        value = row.get("condition_label")

    return str(value)


def slot(row):
    return int(row["slot"])


def attempt(row):
    return int(row["attempt"])


def stable_key(row):
    return (
        instance_id(row),
        slot(row),
        attempt(row),
        condition(row),
    )


def pair_key(row):
    return (
        instance_id(row),
        slot(row),
        attempt(row),
    )


def candidate_order(row):
    # Historical candidate ordering:
    # slot first, then attempt.
    return (
        slot(row),
        attempt(row),
    )


def safe_float(value):
    if value is None:
        return None

    try:
        x = float(value)
    except Exception:
        return None

    if not math.isfinite(x):
        return None

    return x


def mean_or_none(values):
    vals = [
        float(x)
        for x in values
        if x is not None
        and math.isfinite(float(x))
    ]

    if not vals:
        return None

    return float(
        statistics.fmean(vals)
    )


def median_or_none(values):
    vals = [
        float(x)
        for x in values
        if x is not None
        and math.isfinite(float(x))
    ]

    if not vals:
        return None

    return float(
        statistics.median(vals)
    )


def percentile_ci(values):
    vals = np.asarray(
        values,
        dtype=float,
    )

    if len(vals) == 0:
        return [None, None]

    return [
        float(np.quantile(vals, 0.025)),
        float(np.quantile(vals, 0.975)),
    ]


OUT.mkdir(
    parents=True,
    exist_ok=True,
)

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print("=" * 100)
print("STEP 1 — DEV111 DECODER × SELECTOR FACTORIAL")
print("NO GENERATION / NO MODEL INFERENCE")
print("=" * 100)


# ==============================================================================
# 1. Artifact verification
# ==============================================================================

for p in (
    RAW,
    SCORED,
    RESCORED,
    SELECTED,
    MANIFEST,
    COMMON,
):
    if not p.exists():
        raise RuntimeError(
            f"Missing required input: {p}"
        )


raw_sha = sha256_file(RAW)
scored_sha = sha256_file(SCORED)
common_sha = sha256_file(COMMON)

print()
print("RAW SHA   :", raw_sha)
print("SCORED SHA:", scored_sha)
print("COMMON SHA:", common_sha)

if raw_sha != EXPECTED_RAW_SHA:
    raise RuntimeError(
        "Frozen raw-candidate SHA mismatch."
    )

if scored_sha != EXPECTED_SCORED_SHA:
    raise RuntimeError(
        "Frozen scored-candidate SHA mismatch."
    )

if common_sha != EXPECTED_COMMON_SHA:
    raise RuntimeError(
        "Frozen common-rescore SHA mismatch."
    )


raw = load_jsonl(RAW)
scored = load_jsonl(SCORED)
rescored = load_jsonl(RESCORED)
selected = load_jsonl(SELECTED)
manifest = load_manifest(MANIFEST)

print()
print("rows")
print(" raw     :", len(raw))
print(" scored  :", len(scored))
print(" rescored:", len(rescored))
print(" selected:", len(selected))
print(" manifest:", len(manifest))

if len(raw) != EXPECTED_RAW_ROWS:
    raise RuntimeError(
        f"Expected {EXPECTED_RAW_ROWS} raw rows, "
        f"found {len(raw)}"
    )

if len(scored) != EXPECTED_RAW_ROWS:
    raise RuntimeError(
        f"Expected {EXPECTED_RAW_ROWS} scored rows, "
        f"found {len(scored)}"
    )

if len(rescored) != EXPECTED_RAW_ROWS:
    raise RuntimeError(
        f"Expected {EXPECTED_RAW_ROWS} corrected "
        f"candidate rows, found {len(rescored)}"
    )


# ==============================================================================
# 2. Resolve authoritative corrected-candidate schema
# ==============================================================================

rows = rescored

text_field = resolve_field(
    rows,
    exact_names=(
        "rewritten_text",
        "output_text",
        "generated_text",
        "candidate_text",
        "rewrite",
        "text",
    ),
    contains_any=(
        "text",
        "rewrite",
        "output",
    ),
    type_check=lambda x: isinstance(x, str),
    label="candidate text field",
)

t0_field = resolve_field(
    rows,
    exact_names=(
        "common_typicality_n0",
        "typicality_n0",
        "hjtm_typicality_n0",
        "trajectory_typicality_n0",
        "t0",
        "T0",
    ),
    contains_all=(
        "n0",
    ),
    contains_any=(
        "typ",
        "hjtm",
        "trajectory",
        "score",
    ),
    type_check=lambda x: isinstance(
        x,
        (int, float),
    ),
    label="candidate T0 field",
)

t1_field = resolve_field(
    rows,
    exact_names=(
        "common_typicality_n1",
        "typicality_n1",
        "hjtm_typicality_n1",
        "trajectory_typicality_n1",
        "t1",
        "T1",
    ),
    contains_all=(
        "n1",
    ),
    contains_any=(
        "typ",
        "hjtm",
        "trajectory",
        "score",
    ),
    type_check=lambda x: isinstance(
        x,
        (int, float),
    ),
    label="candidate T1 field",
)


# Prefer corrected/final quality-validity field.
keys = sorted({
    k
    for r in rows[:200]
    for k in r.keys()
})

valid_candidates = [
    k
    for k in keys
    if "valid" in k.lower()
    and (
        "quality" in k.lower()
        or "surface" in k.lower()
        or "final" in k.lower()
        or "correct" in k.lower()
    )
]

# Ranking: corrected/final/surface > generic quality.
def valid_rank(k):
    low = k.lower()

    return (
        int("correct" in low),
        int("surface" in low),
        int("final" in low),
        int("quality" in low),
    )


valid_candidates.sort(
    key=valid_rank,
    reverse=True,
)

valid_field = None

for k in valid_candidates:
    vals = [
        r.get(k)
        for r in rows[:200]
        if r.get(k) is not None
    ]

    if vals and all(
        isinstance(x, bool)
        for x in vals
    ):
        valid_field = k
        break


if valid_field is None:
    raise RuntimeError(
        "Could not resolve corrected quality-valid "
        f"boolean field. Available keys={keys}"
    )


print()
print("RESOLVED CANDIDATE SCHEMA")
print(" text :", text_field)
print(" T0   :", t0_field)
print(" T1   :", t1_field)
print(" valid:", valid_field)


# ==============================================================================
# 3. Candidate invariants
# ==============================================================================

all_keys = [
    stable_key(r)
    for r in rows
]

if len(set(all_keys)) != EXPECTED_RAW_ROWS:
    raise RuntimeError(
        "Corrected candidate keys are not unique."
    )


cond_counts = Counter(
    condition(r)
    for r in rows
)

print()
print("condition counts:", dict(cond_counts))

for cond in (
    "vanilla",
    "hpcd_n0",
    "hpcd_n1",
):
    if cond_counts[cond] != EXPECTED_PER_CONDITION:
        raise RuntimeError(
            f"{cond}: expected "
            f"{EXPECTED_PER_CONDITION}, "
            f"found {cond_counts[cond]}"
        )


instance_set = {
    instance_id(r)
    for r in rows
}

base_set = {
    base_id(r)
    for r in rows
}

if len(instance_set) != EXPECTED_INSTANCES:
    raise RuntimeError(
        f"Expected {EXPECTED_INSTANCES} instances, "
        f"found {len(instance_set)}"
    )

if len(base_set) != EXPECTED_BASE_IDS:
    raise RuntimeError(
        f"Expected {EXPECTED_BASE_IDS} base IDs, "
        f"found {len(base_set)}"
    )


per_ic = Counter(
    (
        instance_id(r),
        condition(r),
    )
    for r in rows
)

bad = {
    k: v
    for k, v in per_ic.items()
    if v != EXPECTED_PER_INSTANCE_CONDITION
}

if bad:
    raise RuntimeError(
        "Expected exactly 8 candidates per "
        f"instance/condition. Examples={list(bad.items())[:5]}"
    )


# ==============================================================================
# 4. Manifest/source invariant
# ==============================================================================

manifest_map = {
    instance_id(r): r
    for r in manifest
}

if len(manifest_map) != EXPECTED_INSTANCES:
    raise RuntimeError(
        f"Manifest expected 333 instances, "
        f"found {len(manifest_map)}"
    )


source_field = None

for candidate in (
    "source_text",
    "generated_text",
    "text",
):
    vals = [
        r.get(candidate)
        for r in manifest[:50]
    ]

    if any(
        isinstance(x, str)
        and x.strip()
        for x in vals
    ):
        source_field = candidate
        break


if source_field is None:
    raise RuntimeError(
        "Could not resolve source text field "
        "from manifest."
    )

print("source field:", source_field)


# ==============================================================================
# 5. Build Vanilla and HPCD-n0 candidate dictionaries
# ==============================================================================

vanilla = {}
hpcd = {}

for r in rows:
    c = condition(r)

    if c == "vanilla":
        vanilla[pair_key(r)] = r

    elif c == "hpcd_n0":
        hpcd[pair_key(r)] = r


if len(vanilla) != EXPECTED_PER_CONDITION:
    raise RuntimeError(
        "Vanilla candidate dictionary != 2664"
    )

if len(hpcd) != EXPECTED_PER_CONDITION:
    raise RuntimeError(
        "HPCD-n0 candidate dictionary != 2664"
    )

if set(vanilla) != set(hpcd):
    raise RuntimeError(
        "Vanilla and HPCD candidate pair-key sets differ."
    )


# Verify shared matched seeds.
seed_field = first_present(
    rows[0],
    ("seed", "stable_seed"),
)

if seed_field is None:
    # Search broader sample.
    for r in rows[:100]:
        seed_field = first_present(
            r,
            ("seed", "stable_seed"),
        )
        if seed_field is not None:
            break

if seed_field is None:
    raise RuntimeError(
        "Could not resolve matched seed field."
    )


seed_mismatches = []

for k in vanilla:
    if (
        vanilla[k].get(seed_field)
        !=
        hpcd[k].get(seed_field)
    ):
        seed_mismatches.append(k)

if seed_mismatches:
    raise RuntimeError(
        "Matched Vanilla/HPCD seeds differ. "
        f"Examples={seed_mismatches[:5]}"
    )

print(
    "Matched Vanilla↔HPCD pair keys:",
    len(vanilla),
)

print(
    "Matched seeds:",
    "PASS",
)


# ==============================================================================
# 6. Raw matched-candidate decoder analysis
# ==============================================================================

pair_rows = []

for k in sorted(vanilla):
    v = vanilla[k]
    h = hpcd[k]

    v_valid = bool(v[valid_field])
    h_valid = bool(h[valid_field])

    vt0 = safe_float(v.get(t0_field))
    ht0 = safe_float(h.get(t0_field))

    vt1 = safe_float(v.get(t1_field))
    ht1 = safe_float(h.get(t1_field))

    both_valid = (
        v_valid
        and h_valid
    )

    row = {
        "instance_id": k[0],
        "base_sample_id": base_id(v),
        "generator_key": v.get("generator_key"),
        "domain": v.get("domain"),
        "slot": k[1],
        "attempt": k[2],
        "seed": v.get(seed_field),
        "vanilla_valid": v_valid,
        "hpcd_valid": h_valid,
        "both_valid": both_valid,
        "vanilla_t0": vt0,
        "hpcd_t0": ht0,
        "delta_t0_hpcd_minus_vanilla":
            (
                ht0 - vt0
                if both_valid
                and ht0 is not None
                and vt0 is not None
                else None
            ),
        "vanilla_t1": vt1,
        "hpcd_t1": ht1,
        "delta_t1_hpcd_minus_vanilla":
            (
                ht1 - vt1
                if both_valid
                and ht1 is not None
                and vt1 is not None
                else None
            ),
    }

    pair_rows.append(row)


if len(pair_rows) != 2664:
    raise RuntimeError(
        f"Expected 2664 candidate-pair rows, "
        f"got {len(pair_rows)}"
    )


with PAIR_CSV.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(pair_rows[0]),
    )

    writer.writeheader()
    writer.writerows(pair_rows)


# ==============================================================================
# 7. Four-condition factorial reconstruction
# ==============================================================================

by_instance_condition = defaultdict(list)

for r in rows:
    c = condition(r)

    if c not in (
        "vanilla",
        "hpcd_n0",
    ):
        continue

    by_instance_condition[
        (
            instance_id(r),
            c,
        )
    ].append(r)


def valid_candidates_for(iid, cond):
    candidates = sorted(
        by_instance_condition[
            (iid, cond)
        ],
        key=candidate_order,
    )

    return [
        r
        for r in candidates
        if bool(r[valid_field])
    ]


def select_first_valid(iid, cond):
    valid = valid_candidates_for(
        iid,
        cond,
    )

    if not valid:
        return None

    return valid[0]


def select_hjtm(iid, cond):
    valid = valid_candidates_for(
        iid,
        cond,
    )

    if not valid:
        return None

    scored_valid = [
        r
        for r in valid
        if safe_float(
            r.get(t0_field)
        ) is not None
    ]

    if not scored_valid:
        raise RuntimeError(
            f"{iid}/{cond}: quality-valid candidates "
            "exist but none has T0."
        )

    return min(
        scored_valid,
        key=lambda r: (
            float(r[t0_field]),
            slot(r),
            attempt(r),
        ),
    )


def make_factorial_row(
    iid,
    factorial_condition,
    selected_row,
):
    m = manifest_map[iid]

    source_text = str(
        m[source_field]
    )

    if selected_row is None:
        return {
            "instance_id": iid,
            "base_sample_id": base_id(m),
            "generator_key": m.get("generator_key"),
            "domain": m.get("domain"),
            "factorial_condition":
                factorial_condition,
            "decoder":
                (
                    "vanilla"
                    if factorial_condition.startswith("V_")
                    else "hpcd_n0"
                ),
            "selector":
                (
                    "first_valid"
                    if factorial_condition.endswith("_FV")
                    else "hjtm_n0"
                ),
            "abstained": True,
            "slot": None,
            "attempt": None,
            "seed": None,
            "t0": None,
            "t1": None,
            "output_text": source_text,
        }

    return {
        "instance_id": iid,
        "base_sample_id": base_id(m),
        "generator_key": m.get("generator_key"),
        "domain": m.get("domain"),
        "factorial_condition":
            factorial_condition,
        "decoder":
            (
                "vanilla"
                if factorial_condition.startswith("V_")
                else "hpcd_n0"
            ),
        "selector":
            (
                "first_valid"
                if factorial_condition.endswith("_FV")
                else "hjtm_n0"
            ),
        "abstained": False,
        "slot": slot(selected_row),
        "attempt": attempt(selected_row),
        "seed": selected_row.get(seed_field),
        "t0": safe_float(
            selected_row.get(t0_field)
        ),
        "t1": safe_float(
            selected_row.get(t1_field)
        ),
        "output_text":
            str(selected_row[text_field]),
    }


four_rows = []

for iid in sorted(instance_set):
    specs = (
        (
            "V_FV",
            select_first_valid(
                iid,
                "vanilla",
            ),
        ),
        (
            "V_HJTM",
            select_hjtm(
                iid,
                "vanilla",
            ),
        ),
        (
            "H_FV",
            select_first_valid(
                iid,
                "hpcd_n0",
            ),
        ),
        (
            "H_HJTM",
            select_hjtm(
                iid,
                "hpcd_n0",
            ),
        ),
    )

    for name, chosen in specs:
        four_rows.append(
            make_factorial_row(
                iid,
                name,
                chosen,
            )
        )


if len(four_rows) != 1332:
    raise RuntimeError(
        f"Expected 1332 factorial rows, "
        f"found {len(four_rows)}"
    )


with FOUR_CSV.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(four_rows[0]),
    )

    writer.writeheader()
    writer.writerows(four_rows)


# ==============================================================================
# 8. Validate reconstruction against frozen selected outputs
#
# V_FV must reproduce frozen Vanilla.
# H_HJTM must reproduce frozen HPCD-n0.
# ==============================================================================

selected_keys = sorted({
    k
    for r in selected[:100]
    for k in r.keys()
})

selected_text_field = None

for candidate in (
    "selected_text",
    "output_text",
    "rewritten_text",
    "generated_text",
    "text",
):
    if candidate in selected_keys:
        selected_text_field = candidate
        break


if selected_text_field is None:
    candidates = [
        k
        for k in selected_keys
        if (
            "text" in k.lower()
            or "output" in k.lower()
            or "rewrite" in k.lower()
        )
    ]

    if len(candidates) == 1:
        selected_text_field = candidates[0]
    else:
        raise RuntimeError(
            "Could not resolve frozen selected-output "
            f"text field. Candidates={candidates}"
        )


frozen = {}

for r in selected:
    c = condition(r)

    if c in (
        "vanilla",
        "hpcd_n0",
    ):
        frozen[
            (
                instance_id(r),
                c,
            )
        ] = str(
            r[selected_text_field]
        )


four_lookup = {
    (
        r["instance_id"],
        r["factorial_condition"],
    ): r
    for r in four_rows
}


mismatch_v = []
mismatch_h = []

for iid in sorted(instance_set):
    key_v = (
        iid,
        "vanilla",
    )

    key_h = (
        iid,
        "hpcd_n0",
    )

    if key_v not in frozen:
        raise RuntimeError(
            f"Frozen Vanilla output missing: {iid}"
        )

    if key_h not in frozen:
        raise RuntimeError(
            f"Frozen HPCD-n0 output missing: {iid}"
        )

    if (
        four_lookup[
            (iid, "V_FV")
        ]["output_text"]
        !=
        frozen[key_v]
    ):
        mismatch_v.append(iid)

    if (
        four_lookup[
            (iid, "H_HJTM")
        ]["output_text"]
        !=
        frozen[key_h]
    ):
        mismatch_h.append(iid)


print()
print(
    "Frozen Vanilla reconstruction mismatches:",
    len(mismatch_v),
)

print(
    "Frozen HPCD-n0 reconstruction mismatches:",
    len(mismatch_h),
)


if mismatch_v:
    raise RuntimeError(
        "V_FV does not reproduce frozen Vanilla. "
        f"Examples={mismatch_v[:5]}"
    )

if mismatch_h:
    raise RuntimeError(
        "H_HJTM does not reproduce frozen HPCD-n0. "
        f"Examples={mismatch_h[:5]}"
    )


# ==============================================================================
# 9. Factorial contrasts
# ==============================================================================

by_fc = {
    fc: {}
    for fc in (
        "V_FV",
        "V_HJTM",
        "H_FV",
        "H_HJTM",
    )
}

for r in four_rows:
    by_fc[
        r["factorial_condition"]
    ][r["instance_id"]] = r


contrasts = {
    "decoder_first_valid":
        (
            "H_FV",
            "V_FV",
        ),

    "selector_under_vanilla":
        (
            "V_HJTM",
            "V_FV",
        ),

    "decoder_same_hjtm_selector":
        (
            "H_HJTM",
            "V_HJTM",
        ),

    "full_pipeline":
        (
            "H_HJTM",
            "V_FV",
        ),
}


contrast_results = {}


for name, (a, b) in contrasts.items():
    deltas = []
    rows_delta = []

    for iid in sorted(instance_set):
        ra = by_fc[a][iid]
        rb = by_fc[b][iid]

        if (
            ra["abstained"]
            or rb["abstained"]
            or ra["t0"] is None
            or rb["t0"] is None
        ):
            continue

        delta = (
            float(ra["t0"])
            -
            float(rb["t0"])
        )

        deltas.append(delta)

        rows_delta.append(
            {
                "instance_id": iid,
                "base_sample_id":
                    ra["base_sample_id"],
                "generator_key":
                    ra["generator_key"],
                "domain":
                    ra["domain"],
                "delta": delta,
            }
        )

    contrast_results[name] = {
        "condition_a": a,
        "condition_b": b,
        "n": len(deltas),
        "mean_delta":
            mean_or_none(deltas),
        "median_delta":
            median_or_none(deltas),
        "improved_negative":
            sum(x < 0 for x in deltas),
        "worsened_positive":
            sum(x > 0 for x in deltas),
        "ties":
            sum(x == 0 for x in deltas),
        "_rows": rows_delta,
    }


# Candidate-pair raw decoder result.
raw_pair_deltas = [
    r[
        "delta_t0_hpcd_minus_vanilla"
    ]
    for r in pair_rows
    if r[
        "delta_t0_hpcd_minus_vanilla"
    ] is not None
]


raw_decoder_summary = {
    "n_all_matched_seed_pairs":
        len(pair_rows),

    "n_both_quality_valid":
        len(raw_pair_deltas),

    "mean_delta_t0":
        mean_or_none(
            raw_pair_deltas
        ),

    "median_delta_t0":
        median_or_none(
            raw_pair_deltas
        ),

    "improved_negative":
        sum(
            x < 0
            for x in raw_pair_deltas
        ),

    "worsened_positive":
        sum(
            x > 0
            for x in raw_pair_deltas
        ),

    "ties":
        sum(
            x == 0
            for x in raw_pair_deltas
        ),
}


# ==============================================================================
# 10. Factorial-specific cluster bootstrap
#
# Cluster = underlying base sample ID.
# This is NOT the final all-333 ITT analysis from Step 2.
# ==============================================================================

rng = np.random.default_rng(2027)
B = 10000

boot_rows = []


def cluster_effect_rows(rows_delta):
    groups = defaultdict(list)

    for r in rows_delta:
        groups[
            r["base_sample_id"]
        ].append(
            float(r["delta"])
        )

    return {
        bid: float(
            statistics.fmean(vals)
        )
        for bid, vals in groups.items()
    }


# Candidate pair cluster effects:
pair_cluster = defaultdict(list)

for r in pair_rows:
    delta = r[
        "delta_t0_hpcd_minus_vanilla"
    ]

    if delta is not None:
        pair_cluster[
            r["base_sample_id"]
        ].append(float(delta))


pair_cluster_effects = {
    k: float(
        statistics.fmean(v)
    )
    for k, v in pair_cluster.items()
}


all_boot_targets = {
    "raw_candidate_decoder":
        pair_cluster_effects,
}

for name, result in contrast_results.items():
    all_boot_targets[
        name
    ] = cluster_effect_rows(
        result["_rows"]
    )


for target, effects in all_boot_targets.items():
    ids = sorted(effects)

    vals = np.asarray(
        [
            effects[i]
            for i in ids
        ],
        dtype=float,
    )

    if not len(vals):
        continue

    draws = np.empty(
        B,
        dtype=float,
    )

    for b in range(B):
        idx = rng.integers(
            0,
            len(vals),
            size=len(vals),
        )

        draws[b] = float(
            vals[idx].mean()
        )

    ci = percentile_ci(draws)

    boot_rows.append(
        {
            "contrast": target,
            "cluster_unit":
                "base_sample_id",
            "n_clusters":
                len(vals),
            "point_estimate":
                float(vals.mean()),
            "bootstrap_replicates":
                B,
            "ci_low":
                ci[0],
            "ci_high":
                ci[1],
            "p_boot_effect_lt_zero":
                float(
                    np.mean(
                        draws < 0
                    )
                ),
        }
    )


with BOOT_CSV.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(boot_rows[0]),
    )

    writer.writeheader()
    writer.writerows(boot_rows)


# Remove private rows from JSON output.
for result in contrast_results.values():
    del result["_rows"]


# ==============================================================================
# 11. Coverage summary
# ==============================================================================

coverage = {}

for fc in (
    "V_FV",
    "V_HJTM",
    "H_FV",
    "H_HJTM",
):
    vals = list(
        by_fc[fc].values()
    )

    rewritten = sum(
        not r["abstained"]
        for r in vals
    )

    coverage[fc] = {
        "rewritten":
            rewritten,
        "abstained":
            EXPECTED_INSTANCES
            - rewritten,
        "coverage":
            rewritten
            / EXPECTED_INSTANCES,
        "mean_t0_nonabstained":
            mean_or_none(
                [
                    r["t0"]
                    for r in vals
                    if not r["abstained"]
                ]
            ),
    }


# ==============================================================================
# 12. Audit + summary
# ==============================================================================

audit = {
    "input_files": {
        "raw": str(
            RAW.relative_to(ROOT)
        ),
        "scored": str(
            SCORED.relative_to(ROOT)
        ),
        "rescored": str(
            RESCORED.relative_to(ROOT)
        ),
        "selected": str(
            SELECTED.relative_to(ROOT)
        ),
        "manifest": str(
            MANIFEST.relative_to(ROOT)
        ),
        "common": str(
            COMMON.relative_to(ROOT)
        ),
    },

    "hashes": {
        "raw": raw_sha,
        "scored": scored_sha,
        "rescored":
            sha256_file(RESCORED),
        "selected":
            sha256_file(SELECTED),
        "manifest":
            sha256_file(MANIFEST),
        "common": common_sha,
    },

    "resolved_schema": {
        "candidate_text":
            text_field,
        "candidate_t0":
            t0_field,
        "candidate_t1":
            t1_field,
        "quality_valid":
            valid_field,
        "source_text":
            source_field,
        "selected_text":
            selected_text_field,
        "seed":
            seed_field,
    },

    "counts": {
        "raw_candidates":
            len(raw),
        "scored_candidates":
            len(scored),
        "corrected_candidates":
            len(rescored),
        "source_instances":
            len(instance_set),
        "base_sample_ids":
            len(base_set),
        "candidate_pairs":
            len(pair_rows),
        "factorial_rows":
            len(four_rows),
    },

    "reconstruction": {
        "V_FV_vs_frozen_vanilla_mismatches":
            len(mismatch_v),
        "H_HJTM_vs_frozen_n0_mismatches":
            len(mismatch_h),
    },
}


AUDIT_JSON.write_text(
    json.dumps(
        audit,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)


summary = {
    "status":
        "COMPLETE",

    "analysis":
        "DEV111 decoder × selector factorial",

    "interpretation": {
        "negative_delta":
            "first condition is more human-typical",
        "primary_decoder_contrast":
            "H_FV - V_FV",
        "raw_candidate_decoder_contrast":
            "matched HPCD candidate T0 - Vanilla candidate T0",
    },

    "coverage":
        coverage,

    "raw_candidate_decoder":
        raw_decoder_summary,

    "factorial_contrasts":
        contrast_results,

    "artifacts": {
        "candidate_pair_analysis":
            str(
                PAIR_CSV.relative_to(ROOT)
            ),
        "four_condition_outputs":
            str(
                FOUR_CSV.relative_to(ROOT)
            ),
        "cluster_bootstrap":
            str(
                BOOT_CSV.relative_to(ROOT)
            ),
        "audit":
            str(
                AUDIT_JSON.relative_to(ROOT)
            ),
    },
}


SUMMARY_JSON.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)


# ==============================================================================
# 13. Figure on common complete-case set
# ==============================================================================

try:
    import matplotlib.pyplot as plt

    complete_ids = []

    for iid in sorted(instance_set):
        if all(
            not by_fc[fc][iid]["abstained"]
            and
            by_fc[fc][iid]["t0"]
            is not None
            for fc in (
                "V_FV",
                "V_HJTM",
                "H_FV",
                "H_HJTM",
            )
        ):
            complete_ids.append(iid)

    labels = [
        "V_FV",
        "V_HJTM",
        "H_FV",
        "H_HJTM",
    ]

    means = []

    for fc in labels:
        means.append(
            statistics.fmean(
                float(
                    by_fc[fc][iid]["t0"]
                )
                for iid in complete_ids
            )
        )

    fig, ax = plt.subplots(
        figsize=(7.2, 4.8)
    )

    ax.bar(
        labels,
        means,
    )

    ax.set_ylabel(
        "Mean human-process typicality T0 ↓"
    )

    ax.set_title(
        "DEV111 Decoder × Selector Factorial"
    )

    ax.text(
        0.02,
        0.98,
        f"Common complete cases: n={len(complete_ids)}",
        transform=ax.transAxes,
        va="top",
    )

    fig.tight_layout()

    fig.savefig(
        FIG_PDF,
        bbox_inches="tight",
    )

    plt.close(fig)

except Exception as exc:
    print(
        "WARNING: figure generation failed:",
        repr(exc),
    )


# ==============================================================================
# 14. Final output
# ==============================================================================

print()
print("=" * 100)
print("FACTORIAL RESULT")
print("=" * 100)

print()
print("Candidate-level raw decoder effect")
print(
    json.dumps(
        raw_decoder_summary,
        indent=2,
    )
)

print()
print("Four-condition coverage")
print(
    json.dumps(
        coverage,
        indent=2,
    )
)

print()
print("Factorial contrasts")
print(
    json.dumps(
        contrast_results,
        indent=2,
    )
)

print()
print("Cluster bootstrap")
for row in boot_rows:
    print(row)

print()
print("=" * 100)
print("INVARIANTS")
print("=" * 100)
print(
    "raw candidates                 :",
    len(raw),
)
print(
    "unique source instances        :",
    len(instance_set),
)
print(
    "underlying base IDs            :",
    len(base_set),
)
print(
    "matched candidate pairs        :",
    len(pair_rows),
)
print(
    "factorial rows                 :",
    len(four_rows),
)
print(
    "V_FV frozen Vanilla mismatches :",
    len(mismatch_v),
)
print(
    "H_HJTM frozen n0 mismatches    :",
    len(mismatch_h),
)

print()
print("=" * 100)
print("ARTIFACTS")
print("=" * 100)

for p in (
    AUDIT_JSON,
    PAIR_CSV,
    FOUR_CSV,
    SUMMARY_JSON,
    BOOT_CSV,
):
    print(
        p.relative_to(ROOT),
        sha256_file(p),
    )

if FIG_PDF.exists():
    print(
        FIG_PDF.relative_to(ROOT),
        sha256_file(FIG_PDF),
    )

print()
print("=" * 100)
print("STEP 1 COMPLETE — STOP HERE")
print("=" * 100)

