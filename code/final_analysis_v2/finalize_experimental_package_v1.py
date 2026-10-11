#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import spacy


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(os.environ.get("HPCD_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))

FINAL_ROOT = ROOT / "results/final_analysis_v2"

QUALITY_OUT = FINAL_ROOT / "dipper_quality"
TABLE_OUT = FINAL_ROOT / "tables"
FIGURE_OUT = FINAL_ROOT / "figure_data"

FRESH_COMMON = (
    ROOT
    / "results/quality/fresh95_common_quality_v1/per_instance.jsonl"
)

FRESH_ENTITY = (
    ROOT
    / "results/quality/final_semantic_v1/entity/entity_scores_v1.jsonl"
)

FRESH_NLI = (
    ROOT
    / "results/quality/final_semantic_v1/nli/nli_scores_v1.jsonl"
)

FRESH_DIPPER = (
    ROOT
    / "results/extension_oct2026_v1/dipper/"
      "friend_return_v1/friend_gpu_package_v1/outputs/"
      "fresh95/full/dipper_fresh95_full_final.jsonl"
)

CONFIRM_DIPPER = (
    ROOT
    / "results/extension_oct2026_v1/dipper/"
      "friend_return_v1/friend_gpu_package_v1/outputs/"
      "confirm48/full/dipper_confirm48_full_final.jsonl"
)

CONFIRM_SOURCE = (
    ROOT
    / "results/confirm48_v1/source_generations/"
      "confirm48_machine_sources_v1.jsonl"
)

CONFIRM_SELECTED = (
    ROOT
    / "results/confirm48_v1/process/"
      "confirm48_selected_itt_v1.jsonl"
)

FRESH_MPNET = (
    ROOT
    / "results/final_analysis_v2/"
      "dipper_mpnet_cuda_v2_bs16_return/"
      "fresh95_dipper_mpnet_cuda_v2_bs16.jsonl"
)

CONFIRM_MPNET = (
    ROOT
    / "results/final_analysis_v2/"
      "dipper_mpnet_cuda_v2_bs16_return/"
      "confirm48_vanilla_hpcd_dipper_mpnet_cuda_v2_bs16.jsonl"
)

NEW_NLI = (
    ROOT
    / "results/final_analysis_v2/dipper_nli_v1/"
      "nli_scores_v1.jsonl"
)

FRESH_DIPPER_T0_SUMMARY = (
    ROOT
    / "results/extension_oct2026_v1/dipper/"
      "fresh95_primary_t0_v1/summary.json"
)

CONFIRM_DIPPER_T0_SUMMARY = (
    ROOT
    / "results/extension_oct2026_v1/dipper/"
      "confirm48_primary_t0_v1/summary.json"
)

HIST_ENTITY_CODE = (
    ROOT
    / "code/hpcd/finalize_fresh95_surface_entity_selection_v1.py"
)

FINAL_METHOD_FREEZE = (
    ROOT
    / "configs/hpcd/FINAL_METHOD_FREEZE.json"
)


# =============================================================================
# EXPECTED FROZEN HASHES
# =============================================================================

EXPECTED_SHA = {
    FRESH_COMMON:
        "935ad844fa0c189080b947b8d1d8959c685b30911aa7b6e34e71a4a8d7057513",

    FRESH_ENTITY:
        "ab06655c4dd5679eb36c7d16f4f915514ba530f2a4d69208afded72bc548c8eb",

    FRESH_NLI:
        "d0e4ab27af8f83c42b318fb83020ee2ccb2e4ed57fce3138591ad5be849c0e88",

    FRESH_DIPPER:
        "7ff45d5472025ac7883a922590859dacdb5bc3d4a398c2789c1f995a05661622",

    CONFIRM_DIPPER:
        "aee8bea9a3d9cf3aa6680c97558ac7164f12c1401128ae9c2bdb0cad9de32127",

    CONFIRM_SOURCE:
        "d7d72918046b686f8c4554dd98b8036bcfb10ce18744bd0ced3216f01616b356",

    CONFIRM_SELECTED:
        "6e1beccdb415279fa6e4ff25a2f983f69fd725a2deee87ecb21936fb49aaf06e",

    FRESH_MPNET:
        "0260c176c7e6e2489f45c8fd26999ea790047656b7be119d5f8f21d1e209664d",

    CONFIRM_MPNET:
        "acbf991dc300d9d66860bd339929d380f83abe6d3b56a36f672f4dcd0a1f311f",

    NEW_NLI:
        "4e45cad14b44616d37137a5d9635b77f647e02433853ddccc7e63e303dcf50fa",

    FRESH_DIPPER_T0_SUMMARY:
        "1f9d2a6c90ff27b6500ff61cb9a29d858b8efa9d4f578f66b958481f92d06917",

    CONFIRM_DIPPER_T0_SUMMARY:
        "d3b897f55096101abbfbbba42b550c3cbb15762a9a113547b3ea49c23893e7cf",

    HIST_ENTITY_CODE:
        "319fc639a7c705c4eeaf779e4ae50740fbd919b98bdfc0390c45ffebe9e78b0f",

    FINAL_METHOD_FREEZE:
        "da8610f8acbcc0e0c3c2be4132db63dd013493889559dc518ccc9315777ed035",
}


# =============================================================================
# OUTPUTS
# =============================================================================

FRESH_DIPPER_QUALITY_JSONL = (
    QUALITY_OUT / "fresh95_dipper_quality_v1.jsonl"
)
FRESH_DIPPER_QUALITY_CSV = (
    QUALITY_OUT / "fresh95_dipper_quality_v1.csv"
)

CONFIRM_DIPPER_QUALITY_JSONL = (
    QUALITY_OUT / "confirm48_dipper_quality_v1.jsonl"
)
CONFIRM_DIPPER_QUALITY_CSV = (
    QUALITY_OUT / "confirm48_dipper_quality_v1.csv"
)

ALL_QUALITY_JSONL = (
    QUALITY_OUT / "quality_all_methods_v1.jsonl"
)

PAIRWISE_CSV = (
    QUALITY_OUT / "quality_pairwise_dipper_v1.csv"
)
PAIRWISE_MD = (
    QUALITY_OUT / "quality_pairwise_dipper_v1.md"
)

DOMAIN_CSV = (
    QUALITY_OUT / "quality_by_domain_v1.csv"
)

GENERATOR_CSV = (
    QUALITY_OUT / "quality_by_generator_v1.csv"
)

MAIN_TABLE_CSV = (
    TABLE_OUT / "table_main_process_fidelity_with_dipper_v1.csv"
)
MAIN_TABLE_MD = (
    TABLE_OUT / "table_main_process_fidelity_with_dipper_v1.md"
)

CONFIRM_TABLE_CSV = (
    TABLE_OUT / "table_confirm48_process_fidelity_with_dipper_v1.csv"
)
CONFIRM_TABLE_MD = (
    TABLE_OUT / "table_confirm48_process_fidelity_with_dipper_v1.md"
)

CROSS_SPLIT_CSV = (
    TABLE_OUT / "table_cross_split_replication_v1.csv"
)
CROSS_SPLIT_MD = (
    TABLE_OUT / "table_cross_split_replication_v1.md"
)

FIGURE_DATA = (
    FIGURE_OUT / "figure_process_fidelity_with_dipper_v1.csv"
)

PAPER_NUMBERS_JSON = (
    FINAL_ROOT / "paper_numbers_final_v1.json"
)
PAPER_NUMBERS_CSV = (
    FINAL_ROOT / "paper_numbers_final_v1.csv"
)
PAPER_NUMBERS_MD = (
    FINAL_ROOT / "paper_numbers_final_v1.md"
)

CONSISTENCY_AUDIT = (
    FINAL_ROOT / "consistency_audit_final_v1.md"
)
CLAIM_AUDIT = (
    FINAL_ROOT / "claim_audit_final_v1.md"
)
FINAL_HANDOFF = (
    FINAL_ROOT / "FINAL_PAPER_HANDOFF.md"
)

FREEZE_MANIFEST = (
    FINAL_ROOT / "final_experimental_freeze_manifest_v1.json"
)


ALL_TARGETS = [
    FRESH_DIPPER_QUALITY_JSONL,
    FRESH_DIPPER_QUALITY_CSV,
    CONFIRM_DIPPER_QUALITY_JSONL,
    CONFIRM_DIPPER_QUALITY_CSV,
    ALL_QUALITY_JSONL,
    PAIRWISE_CSV,
    PAIRWISE_MD,
    DOMAIN_CSV,
    GENERATOR_CSV,
    MAIN_TABLE_CSV,
    MAIN_TABLE_MD,
    CONFIRM_TABLE_CSV,
    CONFIRM_TABLE_MD,
    CROSS_SPLIT_CSV,
    CROSS_SPLIT_MD,
    FIGURE_DATA,
    PAPER_NUMBERS_JSON,
    PAPER_NUMBERS_CSV,
    PAPER_NUMBERS_MD,
    CONSISTENCY_AUDIT,
    CLAIM_AUDIT,
    FINAL_HANDOFF,
    FREEZE_MANIFEST,
]


# =============================================================================
# CONSTANTS
# =============================================================================

B = 50_000
BOOT_SEED = 2027

DOMAINS = (
    "news",
    "reddit",
    "reviews",
    "wiki",
)

METHOD_DISPLAY = {
    "original": "Original",
    "vanilla": "Vanilla",
    "temperature": "Temperature Control",
    "copa": "CoPA-3B",
    "dipper": "DIPPER",
    "hpcd": "HPCD",
}

FRESH_T0 = {
    "Original": 1.343030,
    "Vanilla": 1.110840,
    "Temperature Control": 1.030768,
    "CoPA-3B": 1.028039,
    "DIPPER": 1.017539,
    "HPCD": 0.730614,
}

CONFIRM_T0 = {
    "Original": 1.401584923221229,
    "Vanilla": 1.2194043123127005,
    "DIPPER": 1.039241,
    "HPCD": 0.77451596574869,
}


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(
        s.encode("utf-8")
    ).hexdigest()


def load_jsonl(p: Path):
    with p.open(
        "r",
        encoding="utf-8",
    ) as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def write_jsonl(p: Path, rows):
    with p.open(
        "w",
        encoding="utf-8",
    ) as f:
        for r in rows:
            f.write(
                json.dumps(
                    r,
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_csv(p: Path, rows):
    rows = list(rows)

    if not rows:
        raise RuntimeError(
            f"No rows for {p}"
        )

    fields = []
    seen = set()

    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)

    with p.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        w.writeheader()

        for r in rows:
            w.writerow(r)


def finite(v):
    if v is None:
        return None

    try:
        x = float(v)
    except Exception:
        return None

    if not math.isfinite(x):
        return None

    return x


def mean_valid(values):
    x = [
        float(v)
        for v in values
        if finite(v) is not None
    ]

    if not x:
        return None

    return float(
        np.mean(
            np.asarray(
                x,
                dtype=np.float64,
            )
        )
    )


def median_valid(values):
    x = [
        float(v)
        for v in values
        if finite(v) is not None
    ]

    if not x:
        return None

    return float(
        np.median(
            np.asarray(
                x,
                dtype=np.float64,
            )
        )
    )


def std_valid(values):
    x = [
        float(v)
        for v in values
        if finite(v) is not None
    ]

    if len(x) < 2:
        return 0.0 if x else None

    return float(
        np.std(
            np.asarray(
                x,
                dtype=np.float64,
            ),
            ddof=1,
        )
    )


def fmt(v, digits=4):
    if v is None:
        return "--"

    if isinstance(v, str):
        return v

    try:
        x = float(v)
    except Exception:
        return str(v)

    if not math.isfinite(x):
        return "--"

    return f"{x:.{digits}f}"


def generator_key(s: str) -> str:
    low = str(s).casefold()

    if "deepseek" in low:
        return "deepseek"

    if "qwen" in low:
        return "qwen"

    raise RuntimeError(
        f"Unknown source generator: {s!r}"
    )


def method_key(s: str) -> str:
    low = str(s).strip().casefold()

    if "dipper" in low:
        return "dipper"

    if "hpcd" in low:
        return "hpcd"

    if "copa" in low:
        return "copa"

    if (
        "temperature" in low
        or "vanilla_t11" in low
        or "t1.1" in low
    ):
        return "temperature"

    if "vanilla" in low:
        return "vanilla"

    if "original" in low:
        return "original"

    raise RuntimeError(
        f"Unknown condition/method: {s!r}"
    )


def markdown_table(rows, columns):
    header = "| " + " | ".join(
        label
        for _, label in columns
    ) + " |"

    sep = "| " + " | ".join(
        "---"
        for _ in columns
    ) + " |"

    lines = [
        header,
        sep,
    ]

    for r in rows:
        vals = []

        for key, _ in columns:
            value = r.get(key)

            if isinstance(value, float):
                vals.append(
                    fmt(value, 4)
                )
            elif value is None:
                vals.append("--")
            else:
                vals.append(
                    str(value)
                )

        lines.append(
            "| "
            + " | ".join(vals)
            + " |"
        )

    return "\n".join(lines)


# =============================================================================
# PROVENANCE / NON-OVERWRITE
# =============================================================================

print("=" * 110)
print("HPCD FINAL EXPERIMENTAL PACKAGE")
print("DETERMINISTIC INTEGRATION / NO GENERATION / NO DETECTOR RERUN")
print("=" * 110)

for target in ALL_TARGETS:
    if target.exists():
        raise RuntimeError(
            "REFUSING TO OVERWRITE FINAL TARGET: "
            f"{target}"
        )

QUALITY_OUT.mkdir(
    parents=True,
    exist_ok=True,
)

TABLE_OUT.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_OUT.mkdir(
    parents=True,
    exist_ok=True,
)

print()
print("===== FROZEN INPUT SHA CONTRACT =====")

for p, expected in EXPECTED_SHA.items():

    if not p.is_file():
        raise RuntimeError(
            f"Missing frozen input: {p}"
        )

    observed = sha256_file(p)

    print(
        observed,
        p.relative_to(ROOT),
    )

    if observed != expected:
        raise RuntimeError(
            f"SHA mismatch: {p}\n"
            f"expected={expected}\n"
            f"observed={observed}"
        )

print("FROZEN INPUT SHA CONTRACT: PASS")


# =============================================================================
# FROZEN DETERMINISTIC COMMON-QUALITY IMPLEMENTATION
# =============================================================================

MONTHS = (
    "January|February|March|April|May|June|"
    "July|August|September|October|November|December"
)

DATE_RE = re.compile(
    rf"\b(?:{MONTHS})\s+"
    r"\d{1,2}(?:st|nd|rd|th)?,\s*"
    r"\d{4}\b",
    flags=re.IGNORECASE,
)

NUMBER_RE = re.compile(
    r"(?<!\w)"
    r"[+-]?"
    r"(?:[$£€¥])?"
    r"(?:"
        r"\d{1,3}(?:,\d{3})+"
        r"|"
        r"\d+"
    r")"
    r"(?:\.\d+)?"
    r"%?"
    r"(?!\w)",
    flags=re.UNICODE,
)


def unique_in_order(values):
    out = []
    seen = set()

    for value in values:
        key = value.casefold()

        if key in seen:
            continue

        seen.add(key)
        out.append(value)

    return out


def extract_numbers(text: str):
    return unique_in_order(
        [
            m.group(0)
            for m in NUMBER_RE.finditer(
                text
            )
        ]
    )


def extract_dates(text: str):
    return unique_in_order(
        [
            m.group(0)
            for m in DATE_RE.finditer(
                text
            )
        ]
    )


def recall_exact_source_strings(
    source_values,
    output: str,
):
    if not source_values:
        return None

    low = output.casefold()

    preserved = sum(
        1
        for value in source_values
        if value.casefold() in low
    )

    return (
        preserved
        / len(source_values)
    )


def all_preserved(
    source_values,
    output: str,
):
    if not source_values:
        return False

    low = output.casefold()

    return all(
        value.casefold() in low
        for value in source_values
    )


def combined_numdate_preserved(
    source_numbers,
    source_dates,
    output,
):
    if (
        not source_numbers
        and not source_dates
    ):
        return False

    number_ok = (
        all_preserved(
            source_numbers,
            output,
        )
        if source_numbers
        else True
    )

    date_ok = (
        all_preserved(
            source_dates,
            output,
        )
        if source_dates
        else True
    )

    return bool(
        number_ok
        and date_ok
    )


def edit_tokens(text: str):
    return re.findall(
        r"\w+|[^\w\s]",
        text.casefold(),
        flags=re.UNICODE,
    )


def levenshtein_distance(a, b):
    if len(a) > len(b):
        a, b = b, a

    prev = list(
        range(
            len(a) + 1
        )
    )

    for j, bj in enumerate(
        b,
        1,
    ):
        cur = [j]

        for i, ai in enumerate(
            a,
            1,
        ):
            cur.append(
                min(
                    cur[-1] + 1,
                    prev[i] + 1,
                    prev[i - 1]
                    + (
                        0
                        if ai == bj
                        else 1
                    ),
                )
            )

        prev = cur

    return prev[-1]


def token_edit_fraction(
    source: str,
    output: str,
):
    a = edit_tokens(source)
    b = edit_tokens(output)

    denom = max(
        len(a),
        len(b),
    )

    if denom == 0:
        return 0.0

    return (
        levenshtein_distance(
            a,
            b,
        )
        / denom
    )


def whitespace_words(text: str):
    return len(
        text.split()
    )


# =============================================================================
# EXACT HISTORICAL ENTITY VALIDATOR
# =============================================================================

def normalize_surface(text):
    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text = (
        text
        .replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", "-")
        .casefold()
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


ENTITY_IGNORE = {
    "CARDINAL",
    "ORDINAL",
    "PERCENT",
    "MONEY",
    "DATE",
    "TIME",
    "QUANTITY",
}


def source_entities(nlp, text):
    entities = []
    seen = set()

    for ent in nlp(text).ents:

        if ent.label_ in ENTITY_IGNORE:
            continue

        value = normalize_surface(
            ent.text
        )

        if not value:
            continue

        if value not in seen:
            seen.add(value)
            entities.append(value)

    return entities


def entity_present(
    entity,
    normalized_output,
):
    pattern = (
        r"(?<!\w)"
        + re.escape(entity)
        + r"(?!\w)"
    )

    return (
        re.search(
            pattern,
            normalized_output,
        )
        is not None
    )


def entity_metrics(
    entities,
    output,
):
    normalized_output = (
        normalize_surface(
            output
        )
    )

    missing = [
        entity
        for entity in entities
        if not entity_present(
            entity,
            normalized_output,
        )
    ]

    preserved = (
        len(entities)
        - len(missing)
    )

    rate = (
        preserved
        / len(entities)
        if entities
        else None
    )

    return {
        "source_entity_count":
            len(entities),

        "preserved_entity_count":
            preserved,

        "missing_entity_count":
            len(missing),

        "missing_source_entity_surfaces":
            missing,

        "entity_preservation":
            rate,

        # Historical convention:
        # zero-entity source is a vacuous all-preserved pass.
        "all_entities_preserved":
            len(missing) == 0,
    }


# =============================================================================
# REGRESSION 1: DETERMINISTIC COMMON QUALITY AGAINST 475 FROZEN ROWS
# =============================================================================

fresh_common = load_jsonl(
    FRESH_COMMON
)

if len(fresh_common) != 475:
    raise RuntimeError(
        f"Expected 475 Fresh95 common rows, "
        f"got {len(fresh_common)}"
    )

print()
print("===== 475-ROW DETERMINISTIC QUALITY REGRESSION =====")

det_mismatches = []

for r in fresh_common:

    source = r["source_text"]
    output = r["output_text"]

    numbers = extract_numbers(source)
    dates = extract_dates(source)

    num_recall = (
        recall_exact_source_strings(
            numbers,
            output,
        )
    )

    date_recall = (
        recall_exact_source_strings(
            dates,
            output,
        )
    )

    num_ok = all_preserved(
        numbers,
        output,
    )

    date_ok = all_preserved(
        dates,
        output,
    )

    combo_ok = (
        combined_numdate_preserved(
            numbers,
            dates,
            output,
        )
    )

    edit = token_edit_fraction(
        source,
        output,
    )

    sw = whitespace_words(source)
    ow = whitespace_words(output)

    ratio = (
        ow / sw
        if sw
        else 1.0
    )

    errors = []

    if (
        numbers
        !=
        r["source_number_strings"]
    ):
        errors.append(
            (
                "source_number_strings",
                numbers,
                r["source_number_strings"],
            )
        )

    if (
        dates
        !=
        r["source_date_strings"]
    ):
        errors.append(
            (
                "source_date_strings",
                dates,
                r["source_date_strings"],
            )
        )

    checks = [
        (
            "all_numbers_preserved",
            bool(num_ok),
            bool(
                r[
                    "all_numbers_preserved"
                ]
            ),
        ),
        (
            "all_dates_preserved",
            bool(date_ok),
            bool(
                r[
                    "all_dates_preserved"
                ]
            ),
        ),
        (
            "all_number_date_strings_preserved",
            bool(combo_ok),
            bool(
                r[
                    "all_number_date_strings_preserved"
                ]
            ),
        ),
        (
            "source_words",
            sw,
            int(r["source_words"]),
        ),
        (
            "output_words",
            ow,
            int(r["output_words"]),
        ),
    ]

    for name, got, expected in checks:
        if got != expected:
            errors.append(
                (
                    name,
                    got,
                    expected,
                )
            )

    float_checks = [
        (
            "number_string_recall",
            num_recall,
            r["number_string_recall"],
        ),
        (
            "date_string_recall",
            date_recall,
            r["date_string_recall"],
        ),
        (
            "token_edit_fraction",
            edit,
            r["token_edit_fraction"],
        ),
        (
            "word_length_ratio",
            ratio,
            r["word_length_ratio"],
        ),
    ]

    for name, got, expected in float_checks:

        if (
            got is None
            and expected is None
        ):
            continue

        if (
            got is None
            or expected is None
        ):
            errors.append(
                (
                    name,
                    got,
                    expected,
                )
            )
            continue

        if abs(
            float(got)
            - float(expected)
        ) > 1e-12:
            errors.append(
                (
                    name,
                    got,
                    expected,
                )
            )

    if errors:
        det_mismatches.append(
            {
                "instance_id":
                    r["instance_id"],
                "condition":
                    r["condition"],
                "errors":
                    errors,
            }
        )

if det_mismatches:

    print(
        json.dumps(
            det_mismatches[:5],
            indent=2,
            ensure_ascii=False,
        )
    )

    raise RuntimeError(
        "DETERMINISTIC QUALITY REGRESSION FAILED: "
        f"{len(det_mismatches)} rows"
    )

print(
    "475/475 rows exact: PASS"
)


# =============================================================================
# REGRESSION 2: ENTITY VALIDATOR AGAINST FROZEN ENTITY ARTIFACT
# =============================================================================

print()
print("===== FROZEN ENTITY REGRESSION =====")

print(
    "spaCy:",
    spacy.__version__,
)

nlp = spacy.load(
    "en_core_web_sm"
)

print(
    "en_core_web_sm:",
    nlp.meta.get("version"),
)

fresh_entity = load_jsonl(
    FRESH_ENTITY
)

if len(fresh_entity) != 475:
    raise RuntimeError(
        "Unexpected frozen entity row count"
    )

# Original source rows uniquely identify all 95 source realizations.
common_original_by_bg = {}

for r in fresh_common:

    if method_key(
        r["condition"]
    ) != "original":
        continue

    bg = (
        r["base_id"],
        generator_key(
            r["source_generator"]
        ),
    )

    common_original_by_bg[bg] = r

if len(common_original_by_bg) != 95:
    raise RuntimeError(
        "Expected 95 unique Fresh95 Original sources"
    )

entity_source_cache = {}

for bg, r in common_original_by_bg.items():
    entity_source_cache[bg] = (
        source_entities(
            nlp,
            r["source_text"],
        )
    )

entity_frozen_map = {}

for r in fresh_entity:

    mk = method_key(
        r["condition"]
    )

    bgm = (
        r["base_id"],
        generator_key(
            r["source_generator"]
        ),
        mk,
    )

    if bgm in entity_frozen_map:
        raise RuntimeError(
            f"Duplicate entity key: {bgm}"
        )

    entity_frozen_map[bgm] = r

if len(entity_frozen_map) != 475:
    raise RuntimeError(
        "Entity key map not 475"
    )

entity_mismatches = []

for r in fresh_common:

    mk = method_key(
        r["condition"]
    )

    bg = (
        r["base_id"],
        generator_key(
            r["source_generator"]
        ),
    )

    expected = entity_frozen_map[
        (
            bg[0],
            bg[1],
            mk,
        )
    ]

    entities = entity_source_cache[bg]

    observed = entity_metrics(
        entities,
        r["output_text"],
    )

    expected_surfaces = (
        expected[
            "source_entity_surfaces"
        ]
    )

    if entities != expected_surfaces:
        entity_mismatches.append(
            {
                "key": [
                    bg[0],
                    bg[1],
                    mk,
                ],
                "reason":
                    "source_entity_surfaces",
                "observed":
                    entities,
                "expected":
                    expected_surfaces,
            }
        )
        continue

    expected_rate = expected[
        "entity_preservation_rate"
    ]

    observed_rate = observed[
        "entity_preservation"
    ]

    rate_ok = (
        (
            expected_rate is None
            and observed_rate is None
        )
        or
        (
            expected_rate is not None
            and observed_rate is not None
            and abs(
                float(expected_rate)
                - float(observed_rate)
            )
            <= 1e-12
        )
    )

    if (
        not rate_ok
        or
        bool(
            expected[
                "all_entities_preserved"
            ]
        )
        !=
        bool(
            observed[
                "all_entities_preserved"
            ]
        )
    ):
        entity_mismatches.append(
            {
                "key": [
                    bg[0],
                    bg[1],
                    mk,
                ],
                "observed":
                    observed,
                "expected_rate":
                    expected_rate,
                "expected_all":
                    expected[
                        "all_entities_preserved"
                    ],
            }
        )

if entity_mismatches:

    print(
        json.dumps(
            entity_mismatches[:5],
            indent=2,
            ensure_ascii=False,
        )
    )

    raise RuntimeError(
        "ENTITY REGRESSION FAILED: "
        f"{len(entity_mismatches)} rows"
    )

print(
    "475/475 frozen entity rows: PASS"
)


# =============================================================================
# LOAD FINAL MPNET — ADAPTIVE SCHEMA, STRICT UNIVERSE
# =============================================================================

def mpnet_numeric(row):
    candidates = [
        "mpnet_cosine",
        "cosine_similarity",
        "cosine",
        "similarity",
        "score",
    ]

    found = [
        k
        for k in candidates
        if k in row
        and finite(row[k]) is not None
    ]

    if not found:
        raise RuntimeError(
            "Cannot find MPNet score field. Keys: "
            f"{sorted(row.keys())}"
        )

    return float(
        row[found[0]]
    )


def mpnet_index(
    rows,
    default_method=None,
):
    out = {}

    for r in rows:

        raw_method = (
            r.get("method")
            or r.get("condition")
            or default_method
        )

        if raw_method is None:
            raise RuntimeError(
                "MPNet row has no method/condition"
            )

        mk = method_key(
            raw_method
        )

        bgm = (
            r["base_id"],
            generator_key(
                r["source_generator"]
            ),
            mk,
        )

        if bgm in out:
            raise RuntimeError(
                f"Duplicate MPNet key: {bgm}"
            )

        out[bgm] = mpnet_numeric(r)

    return out


fresh_mpnet_rows = load_jsonl(
    FRESH_MPNET
)

confirm_mpnet_rows = load_jsonl(
    CONFIRM_MPNET
)

print()
print("===== FINAL MPNET ARTIFACTS =====")
print(
    "Fresh rows:",
    len(fresh_mpnet_rows),
    "keys:",
    sorted(
        fresh_mpnet_rows[0].keys()
    ),
)
print(
    "Confirm rows:",
    len(confirm_mpnet_rows),
    "keys:",
    sorted(
        confirm_mpnet_rows[0].keys()
    ),
)

fresh_mpnet = mpnet_index(
    fresh_mpnet_rows,
    default_method="DIPPER",
)

confirm_mpnet = mpnet_index(
    confirm_mpnet_rows,
)

if len(fresh_mpnet) != 95:
    raise RuntimeError(
        f"Fresh MPNet universe={len(fresh_mpnet)}, expected 95"
    )

if len(confirm_mpnet) != 285:
    raise RuntimeError(
        f"Confirm MPNet universe={len(confirm_mpnet)}, expected 285"
    )


# Regression against already accepted aggregate values.
fresh_mpnet_mean = float(
    np.mean(
        list(
            fresh_mpnet.values()
        )
    )
)

confirm_mpnet_means = {}

for mk in (
    "vanilla",
    "hpcd",
    "dipper",
):
    vals = [
        v
        for (
            _base,
            _gen,
            method
        ), v in confirm_mpnet.items()
        if method == mk
    ]

    confirm_mpnet_means[mk] = (
        float(
            np.mean(vals)
        )
    )

assert abs(
    fresh_mpnet_mean
    - 0.8649406627604835
) < 1e-12

assert abs(
    confirm_mpnet_means["vanilla"]
    - 0.9765788153598183
) < 1e-12

assert abs(
    confirm_mpnet_means["hpcd"]
    - 0.9730712100079185
) < 1e-12

assert abs(
    confirm_mpnet_means["dipper"]
    - 0.8792775147839597
) < 1e-12

print(
    "FINAL MPNET AGGREGATE REGRESSION: PASS"
)


# =============================================================================
# LOAD FINAL NLI
# =============================================================================

historical_nli = load_jsonl(
    FRESH_NLI
)

new_nli = load_jsonl(
    NEW_NLI
)


def nli_index():
    out = {}

    for split, rows in [
        (
            "fresh95",
            historical_nli,
        ),
        (
            None,
            new_nli,
        ),
    ]:

        for r in rows:

            this_split = (
                r.get("split")
                or split
            )

            mk = method_key(
                r.get("method")
                or r.get("condition")
            )

            bgm = (
                this_split,
                r["base_id"],
                generator_key(
                    r["source_generator"]
                ),
                mk,
            )

            if bgm in out:
                raise RuntimeError(
                    f"Duplicate NLI key: {bgm}"
                )

            out[bgm] = r

    return out


nli_map = nli_index()

new_nli_means = {}

for split, mk in [
    (
        "fresh95",
        "dipper",
    ),
    (
        "confirm48",
        "vanilla",
    ),
    (
        "confirm48",
        "hpcd",
    ),
    (
        "confirm48",
        "dipper",
    ),
]:
    vals = [
        float(r["nli_min"])
        for (
            s,
            _b,
            _g,
            m
        ), r in nli_map.items()
        if s == split
        and m == mk
    ]

    new_nli_means[
        (
            split,
            mk,
        )
    ] = float(
        np.mean(vals)
    )

assert abs(
    new_nli_means[
        (
            "fresh95",
            "dipper",
        )
    ]
    - 0.8365679576013966
) < 1e-12

assert abs(
    new_nli_means[
        (
            "confirm48",
            "vanilla",
        )
    ]
    - 0.9739505462316673
) < 1e-12

assert abs(
    new_nli_means[
        (
            "confirm48",
            "hpcd",
        )
    ]
    - 0.9604914327051902
) < 1e-12

assert abs(
    new_nli_means[
        (
            "confirm48",
            "dipper",
        )
    ]
    - 0.7766300766290807
) < 1e-12

print()
print("FINAL NLI AGGREGATE REGRESSION: PASS")


# =============================================================================
# BUILD STANDARD QUALITY ROW
# =============================================================================

def deterministic_quality(
    source,
    output,
    entities,
):
    numbers = extract_numbers(source)
    dates = extract_dates(source)

    num_recall = (
        recall_exact_source_strings(
            numbers,
            output,
        )
    )

    date_recall = (
        recall_exact_source_strings(
            dates,
            output,
        )
    )

    number_ok = all_preserved(
        numbers,
        output,
    )

    date_ok = all_preserved(
        dates,
        output,
    )

    numdate_ok = (
        combined_numdate_preserved(
            numbers,
            dates,
            output,
        )
    )

    source_words = whitespace_words(
        source
    )

    output_words = whitespace_words(
        output
    )

    length_ratio = (
        output_words
        / source_words
        if source_words
        else 1.0
    )

    em = entity_metrics(
        entities,
        output,
    )

    return {
        "source_number_strings":
            numbers,

        "source_date_strings":
            dates,

        "number_string_recall":
            num_recall,

        "date_string_recall":
            date_recall,

        "all_numbers_preserved":
            bool(number_ok),

        "all_dates_preserved":
            bool(date_ok),

        "number_date_applicable":
            bool(
                numbers
                or dates
            ),

        "all_number_date_strings_preserved":
            bool(numdate_ok),

        "numdate_preservation":
            (
                float(numdate_ok)
                if (
                    numbers
                    or dates
                )
                else None
            ),

        "edit_fraction":
            float(
                token_edit_fraction(
                    source,
                    output,
                )
            ),

        "source_words":
            int(source_words),

        "output_words":
            int(output_words),

        "length_ratio":
            float(length_ratio),

        "length_deviation":
            float(
                abs(
                    length_ratio
                    - 1.0
                )
            ),

        **em,
    }


def make_quality_row(
    *,
    split,
    instance_id,
    base_id,
    domain,
    source_generator,
    method,
    source,
    output,
    entities,
    mpnet,
    nli,
):
    d = deterministic_quality(
        source,
        output,
        entities,
    )

    return {
        "split":
            split,

        "instance_id":
            instance_id,

        "base_id":
            base_id,

        "domain":
            domain,

        "source_generator":
            source_generator,

        "generator_key":
            generator_key(
                source_generator
            ),

        "method":
            method,

        "source_text_sha256":
            sha256_text(source),

        "output_text_sha256":
            sha256_text(output),

        "mpnet":
            (
                float(mpnet)
                if mpnet is not None
                else None
            ),

        "nli_forward":
            (
                float(
                    nli[
                        "forward_entailment"
                    ]
                )
                if nli is not None
                else None
            ),

        "nli_reverse":
            (
                float(
                    nli[
                        "reverse_entailment"
                    ]
                )
                if nli is not None
                else None
            ),

        "nli_min":
            (
                float(
                    nli["nli_min"]
                )
                if nli is not None
                else None
            ),

        "nli_mean":
            (
                float(
                    nli["nli_mean"]
                )
                if nli is not None
                else None
            ),

        **d,
    }


# =============================================================================
# FRESH95 QUALITY ROWS
# =============================================================================

print()
print("===== BUILD FRESH95 QUALITY ROWS =====")

fresh_rows = []

fresh_common_methods = {
    "original",
    "vanilla",
    "copa",
    "hpcd",
}

for r in fresh_common:

    mk = method_key(
        r["condition"]
    )

    if mk not in fresh_common_methods:
        continue

    bg = (
        r["base_id"],
        generator_key(
            r["source_generator"]
        ),
    )

    entity_row = entity_frozen_map[
        (
            bg[0],
            bg[1],
            mk,
        )
    ]

    nli = (
        None
        if mk == "original"
        else nli_map[
            (
                "fresh95",
                bg[0],
                bg[1],
                mk,
            )
        ]
    )

    # Use frozen historical metrics directly for historical methods.
    fresh_rows.append(
        {
            "split":
                "fresh95",

            "instance_id":
                r["instance_id"],

            "base_id":
                r["base_id"],

            "domain":
                r["domain"],

            "source_generator":
                r["source_generator"],

            "generator_key":
                bg[1],

            "method":
                METHOD_DISPLAY[mk],

            "source_text_sha256":
                sha256_text(
                    r["source_text"]
                ),

            "output_text_sha256":
                sha256_text(
                    r["output_text"]
                ),

            "mpnet":
                float(
                    r["mpnet_cosine"]
                ),

            "nli_forward":
                (
                    float(
                        nli[
                            "forward_entailment"
                        ]
                    )
                    if nli is not None
                    else None
                ),

            "nli_reverse":
                (
                    float(
                        nli[
                            "reverse_entailment"
                        ]
                    )
                    if nli is not None
                    else None
                ),

            "nli_min":
                (
                    float(
                        nli["nli_min"]
                    )
                    if nli is not None
                    else None
                ),

            "nli_mean":
                (
                    float(
                        nli["nli_mean"]
                    )
                    if nli is not None
                    else None
                ),

            "source_number_strings":
                r[
                    "source_number_strings"
                ],

            "source_date_strings":
                r[
                    "source_date_strings"
                ],

            "number_string_recall":
                r[
                    "number_string_recall"
                ],

            "date_string_recall":
                r[
                    "date_string_recall"
                ],

            "all_numbers_preserved":
                bool(
                    r[
                        "all_numbers_preserved"
                    ]
                ),

            "all_dates_preserved":
                bool(
                    r[
                        "all_dates_preserved"
                    ]
                ),

            "number_date_applicable":
                bool(
                    r[
                        "source_number_strings"
                    ]
                    or
                    r[
                        "source_date_strings"
                    ]
                ),

            "all_number_date_strings_preserved":
                bool(
                    r[
                        "all_number_date_strings_preserved"
                    ]
                ),

            "numdate_preservation":
                (
                    float(
                        bool(
                            r[
                                "all_number_date_strings_preserved"
                            ]
                        )
                    )
                    if (
                        r[
                            "source_number_strings"
                        ]
                        or
                        r[
                            "source_date_strings"
                        ]
                    )
                    else None
                ),

            "edit_fraction":
                float(
                    r[
                        "token_edit_fraction"
                    ]
                ),

            "source_words":
                int(
                    r["source_words"]
                ),

            "output_words":
                int(
                    r["output_words"]
                ),

            "length_ratio":
                float(
                    r[
                        "word_length_ratio"
                    ]
                ),

            "length_deviation":
                abs(
                    float(
                        r[
                            "word_length_ratio"
                        ]
                    )
                    - 1.0
                ),

            "source_entity_count":
                int(
                    entity_row[
                        "source_entity_count"
                    ]
                ),

            "preserved_entity_count":
                int(
                    entity_row[
                        "preserved_entity_count"
                    ]
                ),

            "missing_entity_count":
                int(
                    entity_row[
                        "missing_entity_count"
                    ]
                ),

            "missing_source_entity_surfaces":
                entity_row[
                    "missing_source_entity_surfaces"
                ],

            "entity_preservation":
                (
                    float(
                        entity_row[
                            "entity_preservation_rate"
                        ]
                    )
                    if entity_row[
                        "entity_preservation_rate"
                    ]
                    is not None
                    else None
                ),

            "all_entities_preserved":
                bool(
                    entity_row[
                        "all_entities_preserved"
                    ]
                ),
        }
    )


# Fresh DIPPER.
fresh_dipper_raw = load_jsonl(
    FRESH_DIPPER
)

if len(fresh_dipper_raw) != 95:
    raise RuntimeError(
        "Fresh DIPPER count != 95"
    )

fresh_dipper_quality = []

for r in fresh_dipper_raw:

    if r.get("status") != "success":
        raise RuntimeError(
            "Fresh DIPPER contains non-success row"
        )

    bg = (
        r["base_id"],
        generator_key(
            r["source_generator"]
        ),
    )

    original = common_original_by_bg[
        bg
    ]

    if (
        sha256_text(
            r["source_text"]
        )
        !=
        sha256_text(
            original[
                "source_text"
            ]
        )
    ):
        raise RuntimeError(
            f"Fresh DIPPER source mismatch: {bg}"
        )

    entities = entity_source_cache[
        bg
    ]

    nli = nli_map[
        (
            "fresh95",
            bg[0],
            bg[1],
            "dipper",
        )
    ]

    mpnet = fresh_mpnet[
        (
            bg[0],
            bg[1],
            "dipper",
        )
    ]

    q = make_quality_row(
        split="fresh95",
        instance_id=r[
            "machine_sample_id"
        ],
        base_id=r["base_id"],
        domain=r["domain"],
        source_generator=r[
            "source_generator"
        ],
        method="DIPPER",
        source=r["source_text"],
        output=r["output_text"],
        entities=entities,
        mpnet=mpnet,
        nli=nli,
    )

    fresh_rows.append(q)
    fresh_dipper_quality.append(q)

if len(fresh_rows) != 475:
    raise RuntimeError(
        f"Fresh final quality universe "
        f"{len(fresh_rows)} != 475"
    )

print(
    "Fresh95 final rows:",
    len(fresh_rows),
)
print(
    "Fresh95 DIPPER rows:",
    len(fresh_dipper_quality),
)


# =============================================================================
# CONFIRM48 QUALITY ROWS
# =============================================================================

print()
print("===== BUILD CONFIRM48 QUALITY ROWS =====")

confirm_dipper_raw = load_jsonl(
    CONFIRM_DIPPER
)

if len(confirm_dipper_raw) != 95:
    raise RuntimeError(
        "Confirm DIPPER count != 95"
    )

confirm_source_by_mid = {}
confirm_source_by_bg = {}

for r in confirm_dipper_raw:

    if r.get("status") != "success":
        raise RuntimeError(
            "Confirm DIPPER contains non-success row"
        )

    mid = r[
        "machine_sample_id"
    ]

    bg = (
        r["base_id"],
        generator_key(
            r["source_generator"]
        ),
    )

    if mid in confirm_source_by_mid:
        raise RuntimeError(
            f"Duplicate Confirm machine ID: {mid}"
        )

    if bg in confirm_source_by_bg:
        raise RuntimeError(
            f"Duplicate Confirm BG: {bg}"
        )

    confirm_source_by_mid[
        mid
    ] = r

    confirm_source_by_bg[
        bg
    ] = r

if len(confirm_source_by_bg) != 95:
    raise RuntimeError(
        "Confirm source map != 95"
    )


# Cross-check frozen source universe itself.
confirm_source_frozen = load_jsonl(
    CONFIRM_SOURCE
)

if len(confirm_source_frozen) != 95:
    raise RuntimeError(
        "Frozen Confirm source count != 95"
    )

for r in confirm_source_frozen:

    bg = (
        r["base_id"],
        generator_key(
            r["source_generator"]
        ),
    )

    if bg not in confirm_source_by_bg:
        raise RuntimeError(
            f"Confirm source missing from DIPPER: {bg}"
        )

    dip = confirm_source_by_bg[bg]

    if (
        sha256_text(
            r["source_text"]
        )
        !=
        sha256_text(
            dip["source_text"]
        )
    ):
        raise RuntimeError(
            f"Confirm frozen/DIPPER source mismatch: {bg}"
        )

print(
    "CONFIRM48 SOURCE CROSS-CHECK: PASS"
)


confirm_entity_cache = {}

for bg, r in confirm_source_by_bg.items():
    confirm_entity_cache[bg] = (
        source_entities(
            nlp,
            r["source_text"],
        )
    )


confirm_rows = []

# Original self rows.
for bg, src in confirm_source_by_bg.items():

    instance_id = src[
        "machine_sample_id"
    ]

    entities = confirm_entity_cache[
        bg
    ]

    q = make_quality_row(
        split="confirm48",
        instance_id=instance_id,
        base_id=src["base_id"],
        domain=src["domain"],
        source_generator=src[
            "source_generator"
        ],
        method="Original",
        source=src["source_text"],
        output=src["source_text"],
        entities=entities,
        mpnet=1.0,
        nli=None,
    )

    confirm_rows.append(q)


# Vanilla + HPCD frozen selected outputs.
confirm_selected = load_jsonl(
    CONFIRM_SELECTED
)

if len(confirm_selected) != 190:
    raise RuntimeError(
        "Confirm selected count != 190"
    )

for r in confirm_selected:

    raw_method = (
        r.get("condition")
        or r.get("method")
    )

    mk = method_key(
        raw_method
    )

    if mk not in {
        "vanilla",
        "hpcd",
    }:
        raise RuntimeError(
            f"Unexpected Confirm selected method: "
            f"{raw_method}"
        )

    candidate_mid = (
        r.get("instance_id")
        or
        r.get("machine_sample_id")
    )

    src = (
        confirm_source_by_mid.get(
            candidate_mid
        )
        if candidate_mid
        else None
    )

    if src is None:

        bg = (
            r["base_id"],
            generator_key(
                r[
                    "source_generator"
                ]
            ),
        )

        src = confirm_source_by_bg[
            bg
        ]

    bg = (
        src["base_id"],
        generator_key(
            src["source_generator"]
        ),
    )

    instance_id = src[
        "machine_sample_id"
    ]

    entities = confirm_entity_cache[
        bg
    ]

    nli = nli_map[
        (
            "confirm48",
            bg[0],
            bg[1],
            mk,
        )
    ]

    mpnet = confirm_mpnet[
        (
            bg[0],
            bg[1],
            mk,
        )
    ]

    q = make_quality_row(
        split="confirm48",
        instance_id=instance_id,
        base_id=src["base_id"],
        domain=src["domain"],
        source_generator=src[
            "source_generator"
        ],
        method=METHOD_DISPLAY[mk],
        source=src["source_text"],
        output=r["output_text"],
        entities=entities,
        mpnet=mpnet,
        nli=nli,
    )

    confirm_rows.append(q)


# DIPPER.
confirm_dipper_quality = []

for r in confirm_dipper_raw:

    bg = (
        r["base_id"],
        generator_key(
            r["source_generator"]
        ),
    )

    entities = confirm_entity_cache[
        bg
    ]

    nli = nli_map[
        (
            "confirm48",
            bg[0],
            bg[1],
            "dipper",
        )
    ]

    mpnet = confirm_mpnet[
        (
            bg[0],
            bg[1],
            "dipper",
        )
    ]

    q = make_quality_row(
        split="confirm48",
        instance_id=r[
            "machine_sample_id"
        ],
        base_id=r["base_id"],
        domain=r["domain"],
        source_generator=r[
            "source_generator"
        ],
        method="DIPPER",
        source=r["source_text"],
        output=r["output_text"],
        entities=entities,
        mpnet=mpnet,
        nli=nli,
    )

    confirm_rows.append(q)
    confirm_dipper_quality.append(q)


if len(confirm_rows) != 380:
    raise RuntimeError(
        f"Confirm final quality universe "
        f"{len(confirm_rows)} != 380"
    )

print(
    "CONFIRM48 final rows:",
    len(confirm_rows),
)
print(
    "CONFIRM48 DIPPER rows:",
    len(confirm_dipper_quality),
)


# =============================================================================
# WRITE PER-INSTANCE QUALITY ARTIFACTS
# =============================================================================

write_jsonl(
    FRESH_DIPPER_QUALITY_JSONL,
    fresh_dipper_quality,
)

write_csv(
    FRESH_DIPPER_QUALITY_CSV,
    fresh_dipper_quality,
)

write_jsonl(
    CONFIRM_DIPPER_QUALITY_JSONL,
    confirm_dipper_quality,
)

write_csv(
    CONFIRM_DIPPER_QUALITY_CSV,
    confirm_dipper_quality,
)

all_quality_rows = (
    fresh_rows
    + confirm_rows
)

write_jsonl(
    ALL_QUALITY_JSONL,
    all_quality_rows,
)


# =============================================================================
# QUALITY SUMMARIES
# =============================================================================

METRICS = [
    "mpnet",
    "nli_min",
    "entity_preservation",
    "numdate_preservation",
    "edit_fraction",
    "length_deviation",
]


def summarize_method_rows(rows):
    return {
        metric:
            mean_valid(
                [
                    r[metric]
                    for r in rows
                ]
            )
        for metric in METRICS
    }


def method_summary(
    rows,
    methods,
):
    out = {}

    for method in methods:
        subset = [
            r
            for r in rows
            if r["method"] == method
        ]

        if not subset:
            raise RuntimeError(
                f"No rows for method {method}"
            )

        out[method] = (
            summarize_method_rows(
                subset
            )
        )

    return out


fresh_summary = method_summary(
    fresh_rows,
    [
        "Original",
        "Vanilla",
        "CoPA-3B",
        "DIPPER",
        "HPCD",
    ],
)

confirm_summary = method_summary(
    confirm_rows,
    [
        "Original",
        "Vanilla",
        "DIPPER",
        "HPCD",
    ],
)


# Original NLI is structurally undefined in publication table.
fresh_summary[
    "Original"
][
    "nli_min"
] = None

confirm_summary[
    "Original"
][
    "nli_min"
] = None


print()
print("===== QUALITY MEANS =====")

for split_name, summary in [
    (
        "Fresh95",
        fresh_summary,
    ),
    (
        "CONFIRM48",
        confirm_summary,
    ),
]:
    print()
    print(split_name)

    for method, metrics in summary.items():
        print(
            method,
            {
                k:
                    (
                        round(v, 6)
                        if v is not None
                        else None
                    )
                for k, v
                in metrics.items()
            },
        )


# =============================================================================
# PAIRED BASE-ID EFFECTS + DOMAIN-STRATIFIED CLUSTER BOOTSTRAP
# =============================================================================

def paired_bootstrap(
    rows,
    *,
    method_a,
    method_b,
    metric,
):
    """
    Effect = A - B.

    1. Pair source-generator realizations.
    2. Average paired deltas inside RAID base ID.
    3. Domain-stratified bootstrap of base IDs.
    """
    a = {
        r["instance_id"]: r
        for r in rows
        if r["method"] == method_a
    }

    b = {
        r["instance_id"]: r
        for r in rows
        if r["method"] == method_b
    }

    shared = sorted(
        set(a)
        & set(b)
    )

    instance_effects = []

    for iid in shared:

        va = finite(
            a[iid].get(metric)
        )

        vb = finite(
            b[iid].get(metric)
        )

        if (
            va is None
            or vb is None
        ):
            continue

        if (
            a[iid]["base_id"]
            !=
            b[iid]["base_id"]
        ):
            raise RuntimeError(
                f"Base mismatch: {iid}"
            )

        instance_effects.append(
            {
                "instance_id":
                    iid,

                "base_id":
                    a[iid]["base_id"],

                "domain":
                    a[iid]["domain"],

                "delta":
                    float(
                        va - vb
                    ),
            }
        )

    if not instance_effects:
        raise RuntimeError(
            f"No paired values for "
            f"{method_a}-{method_b} {metric}"
        )

    by_base = defaultdict(list)
    domain_by_base = {}

    for r in instance_effects:

        base = r["base_id"]

        by_base[base].append(
            r["delta"]
        )

        if (
            base in domain_by_base
            and
            domain_by_base[base]
            != r["domain"]
        ):
            raise RuntimeError(
                f"Domain mismatch within base: {base}"
            )

        domain_by_base[
            base
        ] = r["domain"]

    cluster_effects = []

    for base in sorted(by_base):

        cluster_effects.append(
            {
                "base_id":
                    base,

                "domain":
                    domain_by_base[
                        base
                    ],

                "delta":
                    float(
                        np.mean(
                            np.asarray(
                                by_base[
                                    base
                                ],
                                dtype=np.float64,
                            )
                        )
                    ),
            }
        )

    point = float(
        np.mean(
            np.asarray(
                [
                    r["delta"]
                    for r in cluster_effects
                ],
                dtype=np.float64,
            )
        )
    )

    domain_arrays = {}

    for domain in DOMAINS:

        vals = [
            r["delta"]
            for r in cluster_effects
            if r["domain"] == domain
        ]

        if vals:
            domain_arrays[
                domain
            ] = np.asarray(
                vals,
                dtype=np.float64,
            )

    rng = np.random.default_rng(
        BOOT_SEED
    )

    boot = np.zeros(
        B,
        dtype=np.float64,
    )

    total_cluster_count = 0

    for domain in DOMAINS:

        if domain not in domain_arrays:
            continue

        values = domain_arrays[
            domain
        ]

        n = len(values)

        idx = rng.integers(
            0,
            n,
            size=(
                B,
                n,
            ),
        )

        boot += (
            values[idx]
            .sum(
                axis=1
            )
        )

        total_cluster_count += n

    if total_cluster_count == 0:
        raise RuntimeError(
            "No clusters in bootstrap"
        )

    boot /= total_cluster_count

    ci_low, ci_high = np.quantile(
        boot,
        [
            0.025,
            0.975,
        ],
    )

    return {
        "method_a":
            method_a,

        "method_b":
            method_b,

        "contrast":
            f"{method_a} - {method_b}",

        "metric":
            metric,

        "n_instances":
            len(instance_effects),

        "n_base_ids":
            len(cluster_effects),

        "mean_delta":
            point,

        "ci_low":
            float(ci_low),

        "ci_high":
            float(ci_high),

        "p_delta_lt_0":
            float(
                np.mean(
                    boot < 0
                )
            ),

        "p_delta_gt_0":
            float(
                np.mean(
                    boot > 0
                )
            ),

        "bootstrap_B":
            B,

        "bootstrap_seed":
            BOOT_SEED,

        "unit":
            "RAID base ID; generator deltas averaged within base first",

        "bootstrap":
            "domain-stratified percentile cluster bootstrap",
    }


pairwise_rows = []

for comparator in [
    "Vanilla",
    "CoPA-3B",
    "HPCD",
]:
    for metric in METRICS:

        result = paired_bootstrap(
            fresh_rows,
            method_a="DIPPER",
            method_b=comparator,
            metric=metric,
        )

        result["split"] = (
            "Fresh95"
        )

        pairwise_rows.append(
            result
        )

for comparator in [
    "Vanilla",
    "HPCD",
]:
    for metric in METRICS:

        result = paired_bootstrap(
            confirm_rows,
            method_a="DIPPER",
            method_b=comparator,
            metric=metric,
        )

        result["split"] = (
            "CONFIRM48"
        )

        pairwise_rows.append(
            result
        )


write_csv(
    PAIRWISE_CSV,
    pairwise_rows,
)


pair_md_lines = [
    "# Paired DIPPER Quality Comparisons",
    "",
    (
        "All deltas are **DIPPER − comparator**. "
        "Generator-specific paired effects are averaged within RAID base ID first; "
        "95% intervals use the frozen domain-stratified cluster bootstrap "
        f"(B={B:,}, seed={BOOT_SEED})."
    ),
    "",
    "| Split | Contrast | Metric | Base IDs | Delta | 95% CI | P(Δ<0) | P(Δ>0) |",
    "|---|---|---|---:|---:|---:|---:|---:|",
]

for r in pairwise_rows:
    pair_md_lines.append(
        "| "
        f"{r['split']} | "
        f"{r['contrast']} | "
        f"{r['metric']} | "
        f"{r['n_base_ids']} | "
        f"{r['mean_delta']:+.4f} | "
        f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | "
        f"{r['p_delta_lt_0']:.4f} | "
        f"{r['p_delta_gt_0']:.4f} |"
    )

PAIRWISE_MD.write_text(
    "\n".join(
        pair_md_lines
    )
    + "\n",
    encoding="utf-8",
)


# =============================================================================
# DOMAIN / GENERATOR DESCRIPTIVES
# =============================================================================

def descriptive_breakdown(
    all_rows,
    group_field,
):
    output = []

    for split in [
        "fresh95",
        "confirm48",
    ]:

        split_rows = [
            r
            for r in all_rows
            if r["split"] == split
        ]

        methods = sorted(
            {
                r["method"]
                for r in split_rows
            }
        )

        groups = sorted(
            {
                r[group_field]
                for r in split_rows
            }
        )

        for group in groups:
            for method in methods:

                subset = [
                    r
                    for r in split_rows
                    if r[group_field]
                    == group
                    and
                    r["method"]
                    == method
                ]

                if not subset:
                    continue

                for metric in METRICS:

                    values = [
                        r[metric]
                        for r in subset
                        if finite(
                            r[metric]
                        )
                        is not None
                    ]

                    if not values:
                        continue

                    output.append(
                        {
                            "split":
                                split,

                            group_field:
                                group,

                            "method":
                                method,

                            "metric":
                                metric,

                            "n_instances":
                                len(values),

                            "n_base_ids":
                                len(
                                    {
                                        r[
                                            "base_id"
                                        ]
                                        for r
                                        in subset
                                        if finite(
                                            r[
                                                metric
                                            ]
                                        )
                                        is not None
                                    }
                                ),

                            "mean":
                                mean_valid(
                                    values
                                ),

                            "median":
                                median_valid(
                                    values
                                ),

                            "std":
                                std_valid(
                                    values
                                ),
                        }
                    )

    return output


write_csv(
    DOMAIN_CSV,
    descriptive_breakdown(
        all_quality_rows,
        "domain",
    ),
)

write_csv(
    GENERATOR_CSV,
    descriptive_breakdown(
        all_quality_rows,
        "generator_key",
    ),
)


# =============================================================================
# PUBLICATION TABLES
# =============================================================================

def build_table_rows(
    summary,
    t0_map,
    method_order,
):
    rows = []

    for method in method_order:

        q = summary[method]

        rows.append(
            {
                "Method":
                    method,

                "T0":
                    t0_map[method],

                "MPNet":
                    q["mpnet"],

                "Bi-NLI":
                    q["nli_min"],

                "Entity":
                    q[
                        "entity_preservation"
                    ],

                "Num/Date":
                    q[
                        "numdate_preservation"
                    ],

                "Edit":
                    q[
                        "edit_fraction"
                    ],

                "Length Dev.":
                    q[
                        "length_deviation"
                    ],
            }
        )

    return rows


fresh_table = build_table_rows(
    fresh_summary,
    FRESH_T0,
    [
        "Original",
        "Vanilla",
        "CoPA-3B",
        "DIPPER",
        "HPCD",
    ],
)

confirm_table = build_table_rows(
    confirm_summary,
    CONFIRM_T0,
    [
        "Original",
        "Vanilla",
        "DIPPER",
        "HPCD",
    ],
)

write_csv(
    MAIN_TABLE_CSV,
    fresh_table,
)

write_csv(
    CONFIRM_TABLE_CSV,
    confirm_table,
)

table_columns = [
    (
        "Method",
        "Method",
    ),
    (
        "T0",
        "T0↓",
    ),
    (
        "MPNet",
        "MPNet↑",
    ),
    (
        "Bi-NLI",
        "Bi-NLI↑",
    ),
    (
        "Entity",
        "Entity↑",
    ),
    (
        "Num/Date",
        "Num/Date↑",
    ),
    (
        "Edit",
        "Edit",
    ),
    (
        "Length Dev.",
        "Length Dev.↓",
    ),
]

MAIN_TABLE_MD.write_text(
    "# Fresh95 Process/Fidelity Table\n\n"
    + markdown_table(
        fresh_table,
        table_columns,
    )
    + "\n",
    encoding="utf-8",
)

CONFIRM_TABLE_MD.write_text(
    "# CONFIRM48 Process/Fidelity Table\n\n"
    + markdown_table(
        confirm_table,
        table_columns,
    )
    + "\n",
    encoding="utf-8",
)


# Cross-split T0 replication.
cross_rows = [
    {
        "Split":
            "Fresh95",

        "Vanilla T0":
            1.110840,

        "DIPPER T0":
            1.017539,

        "HPCD T0":
            0.730614,

        "HPCD-Vanilla delta":
            -0.3762651213,

        "HPCD-Vanilla CI low":
            -0.5278136362,

        "HPCD-Vanilla CI high":
            -0.2290116830,

        "DIPPER-HPCD delta":
            0.28316413283471287,

        "DIPPER-HPCD CI low":
            0.11258575445584257,

        "DIPPER-HPCD CI high":
            0.46502754166693505,
    },

    {
        "Split":
            "CONFIRM48",

        "Vanilla T0":
            1.2194043123127005,

        "DIPPER T0":
            1.039241,

        "HPCD T0":
            0.77451596574869,

        "HPCD-Vanilla delta":
            -0.4441082439930872,

        "HPCD-Vanilla CI low":
            -0.5911256659667654,

        "HPCD-Vanilla CI high":
            -0.3040240408077499,

        "DIPPER-HPCD delta":
            0.25383748487978447,

        "DIPPER-HPCD CI low":
            -0.016273100936980173,

        "DIPPER-HPCD CI high":
            0.5305281998871768,
    },
]

write_csv(
    CROSS_SPLIT_CSV,
    cross_rows,
)

CROSS_SPLIT_MD.write_text(
    "# Cross-Split Replication\n\n"
    "| Split | Vanilla T0 | DIPPER T0 | HPCD T0 | "
    "HPCD−Vanilla Δ [95% CI] | DIPPER−HPCD Δ [95% CI] |\n"
    "|---|---:|---:|---:|---:|---:|\n"
    + "\n".join(
        (
            f"| {r['Split']} | "
            f"{r['Vanilla T0']:.4f} | "
            f"{r['DIPPER T0']:.4f} | "
            f"{r['HPCD T0']:.4f} | "
            f"{r['HPCD-Vanilla delta']:+.4f} "
            f"[{r['HPCD-Vanilla CI low']:+.4f}, "
            f"{r['HPCD-Vanilla CI high']:+.4f}] | "
            f"{r['DIPPER-HPCD delta']:+.4f} "
            f"[{r['DIPPER-HPCD CI low']:+.4f}, "
            f"{r['DIPPER-HPCD CI high']:+.4f}] |"
        )
        for r in cross_rows
    )
    + "\n",
    encoding="utf-8",
)


# =============================================================================
# PROCESS-FIDELITY FIGURE DATA
# =============================================================================

vanilla_t0 = FRESH_T0[
    "Vanilla"
]

vanilla_nli = fresh_summary[
    "Vanilla"
][
    "nli_min"
]

vanilla_mpnet = fresh_summary[
    "Vanilla"
][
    "mpnet"
]

figure_rows = []

for method in [
    "Vanilla",
    "CoPA-3B",
    "DIPPER",
    "HPCD",
]:

    t0 = FRESH_T0[
        method
    ]

    nli = fresh_summary[
        method
    ][
        "nli_min"
    ]

    mpnet = fresh_summary[
        method
    ][
        "mpnet"
    ]

    figure_rows.append(
        {
            "method":
                method,

            "T0_absolute":
                t0,

            "NLI_min_absolute":
                nli,

            "MPNet_absolute":
                mpnet,

            # Raw method - Vanilla contrasts.
            "T0_delta_vs_Vanilla":
                t0
                - vanilla_t0,

            "NLI_delta_vs_Vanilla":
                nli
                - vanilla_nli,

            "MPNet_delta_vs_Vanilla":
                mpnet
                - vanilla_mpnet,

            # Positive means process improvement over Vanilla.
            "T0_process_gain_vs_Vanilla":
                vanilla_t0
                - t0,
        }
    )

write_csv(
    FIGURE_DATA,
    figure_rows,
)


# =============================================================================
# FINAL PAPER NUMBER SOURCE OF TRUTH
# =============================================================================

paper_numbers = {
    "status":
        "FINAL_FROZEN_PAPER_NUMBERS",

    "definitions": {
        "T0":
            "lower is more human-process-typical",

        "quality_higher_better":
            [
                "MPNet",
                "Bi-NLI",
                "Entity",
                "Num/Date",
            ],

        "length_deviation":
            "mean instance-level abs(word_length_ratio - 1)",

        "quality_inference_unit":
            "RAID base ID",

        "quality_bootstrap":
            {
                "type":
                    "domain-stratified cluster bootstrap",

                "B":
                    B,

                "seed":
                    BOOT_SEED,

                "within_base":
                    "average available generator-specific paired deltas first",
            },
    },

    "primary_fresh95": {
        "vanilla_T0":
            1.110840,

        "hpcd_T0":
            0.730614,

        "hpcd_minus_vanilla_cluster_delta":
            -0.3762651213,

        "ci95":
            [
                -0.5278136362,
                -0.2290116830,
            ],

        "directional_probability_hpcd_lower":
            1.0,
    },

    "confirm48": {
        "original_T0":
            1.401584923221229,

        "vanilla_T0":
            1.2194043123127005,

        "dipper_T0":
            1.039241,

        "hpcd_T0":
            0.77451596574869,

        "hpcd_minus_vanilla_instance_delta":
            -0.4448883465640107,

        "hpcd_minus_vanilla_cluster_delta":
            -0.4441082439930872,

        "hpcd_minus_vanilla_ci95":
            [
                -0.5911256659667654,
                -0.3040240408077499,
            ],

        "hpcd_minus_vanilla_p_lower":
            1.0,

        "hpcd_minus_vanilla_domain_descriptive": {
            "news":
                -0.5552027570445893,
            "reddit":
                -0.5513477554753637,
            "reviews":
                -0.43838744091875564,
            "wiki":
                -0.23149502253363993,
        },

        "hpcd_minus_vanilla_generator_descriptive": {
            "DeepSeek":
                -0.5346343292050885,
            "Qwen":
                -0.3532328749305699,
        },
    },

    "quality": {
        "fresh95":
            fresh_table,

        "confirm48":
            confirm_table,

        "dipper_pairwise":
            pairwise_rows,
    },

    "dipper": {
        "fresh95": {
            "T0":
                1.017539,

            "minus_original_cluster_delta":
                -0.32287237432161003,

            "minus_original_ci95":
                [
                    -0.5691396531486004,
                    -0.08025341177205093,
                ],

            "minus_original_p_lt0":
                0.9955,

            "minus_vanilla_cluster_delta":
                -0.0931009884490145,

            "minus_vanilla_ci95":
                [
                    -0.2721211149449428,
                    0.10541330801229326,
                ],

            "minus_vanilla_p_lt0":
                0.8338,

            "minus_copa_cluster_delta":
                -0.016685125117045285,

            "minus_copa_ci95":
                [
                    -0.21743819499484862,
                    0.18875977527008006,
                ],

            "minus_hpcd_cluster_delta":
                0.28316413283471287,

            "minus_hpcd_ci95":
                [
                    0.11258575445584257,
                    0.46502754166693505,
                ],

            "minus_hpcd_p_lt0":
                0.00024,

            "minus_hpcd_by_generator_descriptive": {
                "DeepSeek":
                    0.4264275872925397,

                "Qwen":
                    0.1503275592058312,
            },

            "minus_hpcd_by_domain_descriptive": {
                "news":
                    0.4557057079165481,

                "reddit":
                    0.5437358718113209,

                "reviews":
                    0.08549056091537827,

                "wiki":
                    0.047724390695604406,
            },
        },

        "confirm48": {
            "T0":
                1.039241,

            "minus_original_cluster_delta":
                -0.3666995042857484,

            "minus_original_ci95":
                [
                    -0.614156251690363,
                    -0.11651357351827103,
                ],

            "minus_vanilla_cluster_delta":
                -0.19027075911330268,

            "minus_vanilla_ci95":
                [
                    -0.45625210591786647,
                    0.06747454946391972,
                ],

            "minus_hpcd_cluster_delta":
                0.25383748487978447,

            "minus_hpcd_ci95":
                [
                    -0.016273100936980173,
                    0.5305281998871768,
                ],

            "minus_hpcd_p_lt0":
                0.03256,

            "minus_hpcd_by_domain_descriptive": {
                "news":
                    0.38266237060954184,

                "reddit":
                    0.6867365201418196,

                "reviews":
                    -0.18962449936792838,

                "wiki":
                    0.135575548135705,
            },
        },
    },

    "copa": {
        "fresh95_T0":
            1.028039,

        "matched_backbone":
            "Qwen2.5-3B",

        "lambda":
            0.5,

        "alpha":
            1e-5,
    },

    "temperature_control": {
        "temperature":
            1.1,

        "fresh95_T0":
            1.030768,

        "fresh95_MPNet":
            0.9725,

        "fresh95_NLI_min":
            0.9765,

        "fresh95_edit":
            0.4417,
    },

    "detectors": {
        "note":
            "secondary transfer evidence; no detector feedback was used by HPCD",

        "Fast_DetectGPT": {
            "Original":
                {
                    "AUROC": 0.9689,
                    "TPR5": 0.8632,
                },

            "Vanilla":
                {
                    "AUROC": 0.8164,
                    "TPR5": 0.5474,
                    "ASR5": 0.3659,
                },

            "Temperature_Control":
                {
                    "AUROC": 0.7325,
                    "TPR5": 0.4105,
                    "ASR5": 0.5244,
                },

            "CoPA_3B":
                {
                    "AUROC": 0.7059,
                    "TPR5": 0.2000,
                    "ASR5": 0.7805,
                },

            "HPCD":
                {
                    "AUROC": 0.8416,
                    "TPR5": 0.6000,
                    "ASR5": 0.3171,
                },
        },

        "RADAR": {
            "Original":
                {
                    "AUROC": 0.9934,
                    "TPR5": 0.8105,
                },

            "Vanilla":
                {
                    "AUROC": 0.9862,
                    "TPR5": 0.7474,
                    "ASR5": 0.0779,
                },

            "Temperature_Control":
                {
                    "AUROC": 0.9820,
                    "TPR5": 0.6842,
                    "ASR5": 0.1558,
                },

            "CoPA_3B":
                {
                    "AUROC": 0.9822,
                    "TPR5": 0.7158,
                    "ASR5": 0.1948,
                },

            "HPCD":
                {
                    "AUROC": 0.9877,
                    "TPR5": 0.7158,
                    "ASR5": 0.1299,
                },
        },

        "SurpMark": {
            "Original":
                {
                    "AUROC": 0.9292,
                    "TPR5": 0.6421,
                },

            "Vanilla":
                {
                    "AUROC": 0.7842,
                    "TPR5": 0.2526,
                    "ASR5": 0.6230,
                },

            "Temperature_Control":
                {
                    "AUROC": 0.6849,
                    "TPR5": 0.2211,
                    "ASR5": 0.6557,
                },

            "CoPA_3B":
                {
                    "AUROC": 0.7116,
                    "TPR5": 0.0526,
                    "ASR5": 0.9180,
                },

            "HPCD":
                {
                    "AUROC": 0.7599,
                    "TPR5": 0.2211,
                    "ASR5": 0.6557,
                },
        },

        "thresholds_global_5pct_FPR": {
            "Fast_DetectGPT":
                1.79150391,

            "RADAR":
                0.88504705,

            "SurpMark":
                0.00000143,

            "calibration_humans":
                280,

            "false_positives_each":
                14,
        },

        "hpcd_minus_original_continuous_score_shift": {
            "Fast_DetectGPT":
                -1.951,

            "RADAR":
                -0.0448,

            "SurpMark":
                -0.000427,

            "LogLikelihood":
                -0.425,

            "LogRank":
                -0.297,

            "DetectLLM_LRR":
                -0.165,
        },
    },

    "mage": {
        "status":
            "FROZEN_COMPLETE",

        "hpcd_minus_original_machine_score_delta":
            -0.1824,

        "ci95":
            [
                -0.4163,
                -0.0106,
            ],

        "hpcd_TPR5":
            0.9684,

        "hpcd_ASR5":
            0.0316,

        "interpretation":
            (
                "HPCD produces a humanward continuous-score shift, "
                "but MAGE remains highly effective at the fixed 5% FPR threshold."
            ),
    },

    "independent_evaluator": {
        "model":
            "Qwen2.5-1.5B",

        "DEV_hpcd_minus_vanilla":
            -0.129,

        "DEV_ci95":
            [
                -0.199,
                -0.059,
            ],

        "Fresh95_hpcd_minus_vanilla":
            -0.1064267253,

        "Fresh95_ci95":
            [
                -0.2706330091,
                0.0541069480,
            ],

        "Fresh95_directional_probability":
            0.90134,
    },

    "coverage": {
        "Fresh95": {
            "n_instances":
                95,

            "Vanilla_rewrite":
                64,

            "Vanilla_fallback":
                31,

            "HPCD_rewrite":
                63,

            "HPCD_fallback":
                32,
        },

        "CONFIRM48": {
            "n_instances":
                95,

            "Vanilla_rewrite":
                67,

            "Vanilla_fallback":
                28,

            "HPCD_rewrite":
                69,

            "HPCD_fallback":
                26,
        },
    },
}

PAPER_NUMBERS_JSON.write_text(
    json.dumps(
        paper_numbers,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)


# Flatten to CSV.
flat_rows = []


def flatten_json(
    value,
    prefix="",
):
    if isinstance(
        value,
        dict,
    ):
        for k, v in value.items():

            child = (
                f"{prefix}.{k}"
                if prefix
                else str(k)
            )

            flatten_json(
                v,
                child,
            )

    elif isinstance(
        value,
        list,
    ):
        flat_rows.append(
            {
                "key":
                    prefix,

                "value":
                    json.dumps(
                        value,
                        ensure_ascii=False,
                    ),
            }
        )

    else:
        flat_rows.append(
            {
                "key":
                    prefix,

                "value":
                    value,
            }
        )


flatten_json(
    paper_numbers
)

write_csv(
    PAPER_NUMBERS_CSV,
    flat_rows,
)


# =============================================================================
# PAPER NUMBERS MARKDOWN
# =============================================================================

paper_md = f"""# HPCD Final Paper Numbers

**Status:** FINAL_FROZEN_PAPER_NUMBERS

## Primary Fresh95

- Vanilla T0: {paper_numbers["primary_fresh95"]["vanilla_T0"]:.6f}
- HPCD T0: {paper_numbers["primary_fresh95"]["hpcd_T0"]:.6f}
- HPCD − Vanilla cluster Δ: {paper_numbers["primary_fresh95"]["hpcd_minus_vanilla_cluster_delta"]:+.6f}
- 95% CI: [{paper_numbers["primary_fresh95"]["ci95"][0]:+.6f}, {paper_numbers["primary_fresh95"]["ci95"][1]:+.6f}]

## Independent CONFIRM48

- Vanilla T0: {paper_numbers["confirm48"]["vanilla_T0"]:.6f}
- HPCD T0: {paper_numbers["confirm48"]["hpcd_T0"]:.6f}
- HPCD − Vanilla cluster Δ: {paper_numbers["confirm48"]["hpcd_minus_vanilla_cluster_delta"]:+.6f}
- 95% CI: [{paper_numbers["confirm48"]["hpcd_minus_vanilla_ci95"][0]:+.6f}, {paper_numbers["confirm48"]["hpcd_minus_vanilla_ci95"][1]:+.6f}]

## Fresh95 Quality

{markdown_table(fresh_table, table_columns)}

## CONFIRM48 Quality

{markdown_table(confirm_table, table_columns)}

## DIPPER vs HPCD T0

- Fresh95 DIPPER − HPCD: **+0.283164**, 95% CI **[+0.112586, +0.465028]**.
- CONFIRM48 DIPPER − HPCD: **+0.253837**, 95% CI **[-0.016273, +0.530528]**.
- Positive DIPPER − HPCD means lower T0 for HPCD.

## Detector Transfer

### TPR at global 5% FPR

| Method | Fast-DGPT↓ | RADAR↓ | SurpMark↓ |
|---|---:|---:|---:|
| Original | 0.8632 | 0.8105 | 0.6421 |
| Vanilla | 0.5474 | 0.7474 | 0.2526 |
| Temperature Control | 0.4105 | 0.6842 | 0.2211 |
| CoPA-3B | 0.2000 | 0.7158 | 0.0526 |
| HPCD | 0.6000 | 0.7158 | 0.2211 |

MAGE: HPCD − Original continuous machine-score Δ = **−0.1824**
(95% CI **[−0.4163, −0.0106]**); HPCD TPR@5 = **0.9684**,
ASR@5 = **0.0316**.

## Coverage

- Fresh95: Vanilla 64 rewrite / 31 fallback; HPCD 63 / 32.
- CONFIRM48: Vanilla 67 rewrite / 28 fallback; HPCD 69 / 26.

## Files

- Paired DIPPER quality inference: `{PAIRWISE_CSV.relative_to(ROOT)}`
- Fresh95 table: `{MAIN_TABLE_CSV.relative_to(ROOT)}`
- CONFIRM48 table: `{CONFIRM_TABLE_CSV.relative_to(ROOT)}`
- Process/fidelity figure data: `{FIGURE_DATA.relative_to(ROOT)}`

All manuscript numbers must be taken from this frozen package.
"""

PAPER_NUMBERS_MD.write_text(
    paper_md,
    encoding="utf-8",
)


# =============================================================================
# CONSISTENCY AUDIT
# =============================================================================

# Fatal scientific-sign checks.
assert (
    paper_numbers[
        "primary_fresh95"
    ][
        "hpcd_minus_vanilla_cluster_delta"
    ]
    < 0
)

assert (
    paper_numbers[
        "confirm48"
    ][
        "hpcd_minus_vanilla_cluster_delta"
    ]
    < 0
)

assert (
    paper_numbers[
        "dipper"
    ][
        "fresh95"
    ][
        "minus_hpcd_cluster_delta"
    ]
    > 0
)

assert (
    paper_numbers[
        "dipper"
    ][
        "confirm48"
    ][
        "minus_hpcd_cluster_delta"
    ]
    > 0
)

# Fresh DIPPER-HPCD CI excludes zero.
assert (
    paper_numbers[
        "dipper"
    ][
        "fresh95"
    ][
        "minus_hpcd_ci95"
    ][0]
    > 0
)

# Confirm DIPPER-HPCD CI MUST cross zero.
confirm_dh_ci = (
    paper_numbers[
        "dipper"
    ][
        "confirm48"
    ][
        "minus_hpcd_ci95"
    ]
)

assert (
    confirm_dh_ci[0]
    < 0
    < confirm_dh_ci[1]
)

# DIPPER-Vanilla CIs cross zero on both splits.
for split in [
    "fresh95",
    "confirm48",
]:
    ci = (
        paper_numbers[
            "dipper"
        ][
            split
        ][
            "minus_vanilla_ci95"
        ]
    )

    assert (
        ci[0]
        < 0
        < ci[1]
    )

consistency_text = """# Final Consistency Audit

## PASS

- Fresh95 HPCD − Vanilla has the frozen negative sign; lower T0 favors HPCD.
- CONFIRM48 HPCD − Vanilla has the frozen negative sign; lower T0 favors HPCD.
- Fresh95 DIPPER − HPCD is positive and its 95% CI excludes zero; this favors HPCD.
- CONFIRM48 DIPPER − HPCD is positive but its 95% CI crosses zero.
- DIPPER − Vanilla confidence intervals cross zero on both splits.
- Lower T0 is interpreted as more human-process-typical.
- Positive DIPPER − HPCD is interpreted as favoring HPCD.
- Negative HPCD − Vanilla is interpreted as favoring HPCD.
- Domain and source-generator subgroup results are descriptive unless explicitly supported by their own interval.
- No detector metric is treated as an HPCD optimization objective.
- No result claims that HPCD is the strongest detector-evasion attack.
- No claim says HPCD beats DIPPER in every CONFIRM48 domain.
- CoPA is the matched-backbone Qwen2.5-3B implementation, not the 72B system.
- Fresh95 contains two source-generator families, with one frozen missing DeepSeek realization in Wiki.
- Length >= 0.8 is not treated as an HPCD selection gate in the manuscript.
- Original Bi-NLI remains structurally reported as “--”.
- BERTScore was skipped because it is optional and not a blocker.

## FINAL STATUS

CONSISTENCY_AUDIT=PASS
"""

CONSISTENCY_AUDIT.write_text(
    consistency_text,
    encoding="utf-8",
)


# =============================================================================
# CLAIM AUDIT
# =============================================================================

claim_text = """# Final Claim Audit

## SUPPORTED

- HPCD clearly lowers T0 relative to Vanilla on Fresh95.
- The preregistered HPCD-vs-Vanilla direction independently reproduces on CONFIRM48, with the pooled cluster-bootstrap CI excluding zero.
- HPCD has the lowest mean T0 among the evaluated rewriting methods on Fresh95.
- HPCD has the lowest mean T0 among the evaluated rewriting methods on CONFIRM48.
- DIPPER lowers T0 relative to Original on both Fresh95 and CONFIRM48.
- On Fresh95, HPCD has clearly lower T0 than DIPPER.
- Detector feedback is never used by HPCD for generation, selection, beta choice, or method development.
- Detector-score reductions are secondary transfer evidence rather than the optimization objective.
- HPCD retains approximately Vanilla-level fidelity on Fresh95, while CoPA incurs substantially larger fidelity loss.

## QUALIFIED

- On CONFIRM48, the DIPPER-HPCD point estimate favors HPCD, but the 95% cluster-bootstrap CI narrowly includes zero.
- The independent Qwen2.5-1.5B evaluator is favorable on Fresh95, but its confidence interval crosses zero.
- Domain and source-generator results are supporting/descriptive unless an explicitly reported interval establishes otherwise.
- The human-derived occupancy controller reliably shifts generation toward the human-typical region relative to Vanilla, but a shuffled-prior ablation did not establish that the exact human occupancy mapping is uniquely responsible for this effect.
- MAGE shows a humanward continuous-score shift, while remaining highly effective at the fixed 5% FPR threshold.

## FORBIDDEN

- HPCD significantly beats DIPPER on both splits.
- HPCD beats DIPPER in every domain.
- DIPPER significantly beats Vanilla.
- DIPPER is significantly better than CoPA.
- HPCD is the strongest detector-evasion attack.
- HPCD achieves universal detector evasion or “defeats MAGE.”
- The exact human occupancy mapping is uniquely causal.
- Process alignment proves human cognition.
- CoPA is a 72B baseline.
- Qwen3 was evaluated in the final Fresh95/CONFIRM48 experiment.
- Fresh95 uses three source generators.
- Length >= 0.8 is an HPCD quality-selection gate.
"""

CLAIM_AUDIT.write_text(
    claim_text,
    encoding="utf-8",
)


# =============================================================================
# FINAL PAPER HANDOFF
# =============================================================================

handoff = f"""============================================================
HPCD FINAL PAPER HANDOFF
============================================================

## 1. PRIMARY FRESH95 RESULT

Vanilla T0: 1.110840
HPCD T0: 0.730614
HPCD − Vanilla cluster delta: -0.376265
95% CI: [-0.527814, -0.229012]

Interpretation:
HPCD produces a clear pooled shift toward the human-calibrated
token-choice occupancy region relative to Vanilla.

## 2. INDEPENDENT CONFIRM48 REPLICATION

Vanilla T0: 1.219404
HPCD T0: 0.774516
HPCD − Vanilla cluster delta: -0.444108
95% CI: [-0.591126, -0.304024]

Interpretation:
CONFIRM48 independently reproduces the preregistered direction:
HPCD has lower T0 than Vanilla, and the 95% domain-stratified
cluster-bootstrap CI excludes zero.

## 3. EXTERNAL BASELINES

### Fresh95

Original: 1.343030
Vanilla: 1.110840
CoPA-3B: 1.028039
DIPPER: 1.017539
HPCD: 0.730614

### CONFIRM48

Original: 1.401585
Vanilla: 1.219404
DIPPER: 1.039241
HPCD: 0.774516

Across both splits, HPCD has the lowest mean T0 among the
evaluated rewriting methods.

## 4. DIPPER VS HPCD

Fresh95 DIPPER − HPCD:
+0.283164
95% CI: [+0.112586, +0.465028]

Interpretation:
Positive DIPPER − HPCD favors HPCD. The Fresh95 external-baseline
advantage is clearly resolved.

CONFIRM48 DIPPER − HPCD:
+0.253837
95% CI: [-0.016273, +0.530528]

Interpretation:
The independent split is directionally consistent with Fresh95,
but the cluster-bootstrap confidence interval narrowly includes zero.

Safe cross-split statement:

“Across both Fresh95 and the independently constructed CONFIRM48 split,
HPCD achieves the lowest mean T0 among the evaluated rewriting methods.
Relative to DIPPER, HPCD shows a clear advantage on Fresh95 and a
directionally consistent advantage on CONFIRM48, although the CONFIRM48
cluster-bootstrap confidence interval narrowly includes zero.”

## 5. QUALITY

### Fresh95

{markdown_table(fresh_table, table_columns)}

### CONFIRM48

{markdown_table(confirm_table, table_columns)}

DIPPER fidelity is evaluated on every frozen DIPPER output.
No output was filtered, regenerated, or selected using quality metrics.

Full paired inference:
`{PAIRWISE_CSV.relative_to(ROOT)}`

## 6. DETECTOR TRANSFER

Detector work is frozen and closed.

### TPR at global 5% FPR

| Method | Fast-DGPT↓ | RADAR↓ | SurpMark↓ |
|---|---:|---:|---:|
| Original | 0.8632 | 0.8105 | 0.6421 |
| Vanilla | 0.5474 | 0.7474 | 0.2526 |
| Temperature Control | 0.4105 | 0.6842 | 0.2211 |
| CoPA-3B | 0.2000 | 0.7158 | 0.0526 |
| HPCD | 0.6000 | 0.7158 | 0.2211 |

MAGE:
HPCD − Original continuous machine-score delta = -0.1824
95% CI = [-0.4163, -0.0106]
HPCD TPR@5 = 0.9684
HPCD ASR@5 = 0.0316

Correct interpretation:
HPCD shifts detector evidence toward the human side relative to Original,
but it is not the strongest detector-evasion attack. MAGE remains highly
effective at the fixed 5% FPR threshold.

DIPPER was not added to the frozen detector suite under the final
closed-detector directive.

## 7. COVERAGE

Fresh95:
Vanilla = 64 rewrite / 31 fallback
HPCD = 63 rewrite / 32 fallback

CONFIRM48:
Vanilla = 67 rewrite / 28 fallback
HPCD = 69 rewrite / 26 fallback

## 8. DOMAIN GENERALIZATION

Fresh95 HPCD − Vanilla T0 point estimates:
News: -0.483639
Reddit: -0.401711
Reviews: -0.356135
Wiki: -0.263575

All four point estimates favor HPCD; the Reddit interval crosses zero.
Do not claim four separately significant domain effects.

Fresh95 DIPPER − HPCD descriptive T0:
News: +0.455706
Reddit: +0.543736
Reviews: +0.085491
Wiki: +0.047724

CONFIRM48 HPCD − Vanilla descriptive T0:
News: -0.555203
Reddit: -0.551348
Reviews: -0.438387
Wiki: -0.231495

CONFIRM48 DIPPER − HPCD descriptive T0:
News: +0.382662
Reddit: +0.686737
Reviews: -0.189624
Wiki: +0.135576

Subgroup results are supporting/descriptive.

## 9. SOURCE-GENERATOR GENERALIZATION

Fresh95 DIPPER − HPCD:
DeepSeek: +0.426428
Qwen: +0.150328

Both Fresh95 source-generator point estimates favor HPCD.

CONFIRM48 HPCD − Vanilla:
DeepSeek: -0.534634
Qwen: -0.353233

Do not elevate source-generator breakdowns to primary inference.

## 10. INDEPENDENT PROCESS EVALUATOR

Qwen2.5-1.5B evaluator:

DEV:
HPCD − Vanilla approximately -0.129
95% CI [-0.199, -0.059]

Fresh95:
HPCD − Vanilla -0.106427
95% CI [-0.270633, +0.054107]

Interpretation:
Independent evaluation is supportive; Fresh95 cross-model evidence is
directionally favorable but not conclusive.

## 11. NEGATIVE CONTROL

The human-derived occupancy controller reliably shifts generation toward
the human-typical region relative to Vanilla, but a shuffled-prior
ablation did not establish that the exact human occupancy mapping is
uniquely responsible for this effect.

## 12. CLAIM BOUNDARIES

- Do not frame HPCD as the strongest detector attack.
- Do not claim HPCD significantly beats DIPPER on CONFIRM48.
- Do not claim every domain or source-generator subgroup is significant.
- Do not claim the exact human occupancy map is uniquely causal.
- Detector feedback was never used for HPCD generation or selection.

## 13. REMAINING EXPERIMENT STATUS

Required remaining:
NONE

Optional human evaluation:
Not required for experimental freeze; do not delay manuscript finalization.

BERTScore:
SKIPPED. It is appendix-only and not a blocker.

Detector work:
CLOSED.

Model generation:
CLOSED.

## 14. RECOMMENDED PAPER STORY

PROCESS ALIGNMENT
        ↓
FIDELITY PRESERVATION
        ↓
DETECTOR-INDEPENDENT TRANSFER

The contribution is human-process control, not detector optimization.

============================================================
FINAL EXPERIMENTAL STATUS
============================================================

REQUIRED EXPERIMENTS: COMPLETE
NEXT STAGE: MANUSCRIPT FINALIZATION
"""

FINAL_HANDOFF.write_text(
    handoff,
    encoding="utf-8",
)


# =============================================================================
# FINAL FREEZE MANIFEST
# =============================================================================

def git_commit():
    try:
        return (
            subprocess.check_output(
                [
                    "git",
                    "rev-parse",
                    "HEAD",
                ],
                cwd=ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
        )
    except Exception:
        return None


def git_dirty():
    try:
        value = subprocess.check_output(
            [
                "git",
                "status",
                "--porcelain",
            ],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )

        return bool(
            value.strip()
        )

    except Exception:
        return None


source_manifest = {
    str(p.relative_to(ROOT)):
        sha256_file(p)
    for p in EXPECTED_SHA
}

# Add other important frozen sources when present.
optional_frozen_sources = [
    ROOT
    / "results/detectors/fresh95_final_v1/inputs/"
      "fresh95_detector_input_v1.jsonl",

    ROOT
    / "results/final_analysis_v1/asr/"
      "detector_asr_by_dataset_v1.csv",

    ROOT
    / "results/extension_oct2026_v1/mage/"
      "mage_protocol_freeze_v1.json",
]

for p in optional_frozen_sources:
    if p.is_file():
        source_manifest[
            str(
                p.relative_to(ROOT)
            )
        ] = sha256_file(p)


# Output hashes excluding manifest itself.
output_artifacts = {}

for p in ALL_TARGETS:

    if p == FREEZE_MANIFEST:
        continue

    if not p.is_file():
        raise RuntimeError(
            f"Final output missing before freeze: {p}"
        )

    output_artifacts[
        str(
            p.relative_to(ROOT)
        )
    ] = sha256_file(p)


# Include final integration script.
integration_script = (
    ROOT
    / "code/final_analysis_v2/"
      "finalize_experimental_package_v1.py"
)

if integration_script.is_file():
    output_artifacts[
        str(
            integration_script.relative_to(
                ROOT
            )
        )
    ] = sha256_file(
        integration_script
    )


freeze = {
    "status":
        "EXPERIMENTAL_PACKAGE_FROZEN_COMPLETE",

    "analysis_timestamp_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "repository": {
        "git_commit":
            git_commit(),

        "git_dirty":
            git_dirty(),
    },

    "environment": {
        "python":
            sys.version,

        "platform":
            platform.platform(),

        "numpy":
            np.__version__,

        "spacy":
            spacy.__version__,

        "spacy_model":
            nlp.meta.get(
                "name"
            ),

        "spacy_model_version":
            nlp.meta.get(
                "version"
            ),
    },

    "frozen_parameters": {
        "HPCD": {
            "backbone":
                "Qwen2.5-3B-Instruct",

            "controller_order":
                0,

            "beta":
                0.22,

            "temperature":
                1.0,

            "top_p":
                0.95,

            "seed":
                2027,
        },

        "DIPPER": {
            "checkpoint":
                "kalpeshk2011/dipper-paraphraser-xxl",

            "lex_diversity":
                60,

            "order_diversity":
                60,

            "sent_interval":
                3,

            "do_sample":
                True,

            "top_p":
                0.75,

            "max_length":
                512,

            "global_seed":
                15485863,
        },

        "quality_bootstrap": {
            "B":
                B,

            "seed":
                BOOT_SEED,

            "cluster_unit":
                "RAID base ID",

            "stratification":
                "domain",

            "within_base":
                "average available generator-specific paired deltas first",
        },

        "quality_metrics": {
            "MPNet":
                "frozen exact historical CUDA regression; batch size 16",

            "NLI":
                (
                    "sentence-aware <=220-token chunks; "
                    "bidirectional max-support weighted document score"
                ),

            "entity":
                "source-only spaCy entity surfaces; normalized exact occurrence",

            "number_date":
                "exact case-insensitive source-string preservation",

            "edit":
                "case-folded token Levenshtein / max token count",

            "length_deviation":
                "abs(whitespace-word length ratio - 1)",
        },
    },

    "source_artifacts":
        source_manifest,

    "output_artifacts":
        output_artifacts,

    "stop_condition": {
        "required_remaining":
            "NONE",

        "detector_work":
            "CLOSED",

        "generation":
            "CLOSED",

        "new_models":
            "CLOSED",

        "next_stage":
            "MANUSCRIPT_FINALIZATION",
    },
}

FREEZE_MANIFEST.write_text(
    json.dumps(
        freeze,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)


# =============================================================================
# FINAL PRINT
# =============================================================================

print()
print("=" * 110)
print("FINAL PUBLICATION TABLE — FRESH95")
print("=" * 110)
print(
    markdown_table(
        fresh_table,
        table_columns,
    )
)

print()
print("=" * 110)
print("FINAL PUBLICATION TABLE — CONFIRM48")
print("=" * 110)
print(
    markdown_table(
        confirm_table,
        table_columns,
    )
)

print()
print("=" * 110)
print("DIPPER PAIRED QUALITY COMPARISONS")
print("=" * 110)

for r in pairwise_rows:
    print(
        f"{r['split']:9s} "
        f"{r['contrast']:23s} "
        f"{r['metric']:24s} "
        f"Δ={r['mean_delta']:+.6f} "
        f"CI=[{r['ci_low']:+.6f}, {r['ci_high']:+.6f}] "
        f"bases={r['n_base_ids']}"
    )

print()
print("=" * 110)
print("FINAL FILE HASHES")
print("=" * 110)

for p in ALL_TARGETS:

    if not p.is_file():
        raise RuntimeError(
            f"Missing final artifact: {p}"
        )

    print(
        sha256_file(p),
        p.relative_to(ROOT),
    )

print()
print(
    "INTEGRATION SCRIPT SHA:",
    sha256_file(
        integration_script
    ),
)

print()
print("=" * 110)
print("EXPERIMENTAL_PACKAGE_FROZEN_COMPLETE")
print("REQUIRED REMAINING EXPERIMENTS: NONE")
print("NEXT STAGE: MANUSCRIPT FINALIZATION")
print("=" * 110)
