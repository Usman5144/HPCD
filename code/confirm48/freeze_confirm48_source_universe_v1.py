#!/usr/bin/env python3

from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import os


ROOT = Path(os.environ.get("HPCD_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))

BASE_MANIFEST = (
    ROOT
    / "results/confirm48_v1/manifests/"
      "confirm48_base_ids_v1.json"
)

PREP_MANIFEST = (
    ROOT
    / "results/confirm48_v1/manifests/"
      "confirm48_source_generation_prepare_v1.json"
)

QWEN_SELECTED = (
    ROOT
    / "results/confirm48_v1/source_generations/"
      "qwen25_3b_v1/"
      "qwen25_3b_confirm48_selected_v1.jsonl"
)

DEEP_SELECTED = (
    ROOT
    / "results/confirm48_v1/source_generations/"
      "deepseek_prefill_v1/"
      "deepseek_prefill_confirm48_selected_v1.jsonl"
)

QWEN_DIR = (
    ROOT
    / "results/confirm48_v1/source_generations/"
      "qwen25_3b_v1"
)

DEEP_DIR = (
    ROOT
    / "results/confirm48_v1/source_generations/"
      "deepseek_prefill_v1"
)

OUT_SOURCE = (
    ROOT
    / "results/confirm48_v1/source_generations/"
      "confirm48_machine_sources_v1.jsonl"
)

OUT_FAILURES = (
    ROOT
    / "results/confirm48_v1/source_generations/"
      "confirm48_source_failures_v1.jsonl"
)

OUT_COVERAGE = (
    ROOT
    / "results/confirm48_v1/manifests/"
      "confirm48_source_coverage_v1.json"
)

OUT_FREEZE = (
    ROOT
    / "results/confirm48_v1/manifests/"
      "confirm48_source_freeze_v1.json"
)


EXPECTED = {
    BASE_MANIFEST:
        "897463403597724237e1b1f88a01e05e"
        "580994d645574be4294f4f13a2b3066a",

    PREP_MANIFEST:
        "11f54779c76dfefda31b7ca86d36f8f5"
        "81293be09fbadd2c9de612afd3ae289d",

    QWEN_SELECTED:
        "d0ccb3ea21bd283701113e873ab8a0914"
        "38c2093c3ad7bf3208b36bf3fff5dd7",

    DEEP_SELECTED:
        "012d066eb2348f75a649b22e36f00d5e7"
        "7d948477dbe56b5543ea0feec7c20b5",
}


QWEN_NAME = "Qwen2.5-3B-Instruct"
DEEP_NAME = "DeepSeek-R1-Distill-Qwen-7B"

GENERATOR_KEY = {
    QWEN_NAME:
        "qwen25_3b",

    DEEP_NAME:
        "deepseek_r1_qwen7b_prefill",
}


# =====================================================================
# Helpers
# =====================================================================

def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def load_jsonl(path: Path):
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                rows.append(
                    json.loads(line)
                )
            except Exception as e:
                raise RuntimeError(
                    f"{path}: bad JSON record {n}: {e}"
                ) from e

    return rows


def write_jsonl_atomic(path: Path, rows):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    with tmp.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    os.replace(tmp, path)


def write_json_atomic(path: Path, obj):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(tmp, path)


def status(row):
    x = row.get(
        "validation_status"
    )

    if x is None:
        x = row.get(
            "final_validation_status"
        )

    return x


def chosen_attempt(row):
    """
    Supports both:
    - flattened selected rows;
    - parent rows containing attempts + selected_retry_index.
    """
    if (
        "attempts" not in row
        or "selected_retry_index" not in row
    ):
        return None

    idx = row[
        "selected_retry_index"
    ]

    for a in row["attempts"]:
        if a.get("retry_index") == idx:
            return a

    raise RuntimeError(
        f"Selected retry not found: "
        f"{row.get('sample_id')}"
    )


def extract_text(row):
    for key in (
        "source_text",
        "generated_text",
        "output_text",
        "selected_text",
    ):
        value = row.get(key)

        if isinstance(
            value,
            str,
        ) and value.strip():
            return value

    a = chosen_attempt(row)

    if a is not None:
        value = a.get(
            "generated_text"
        )

        if isinstance(
            value,
            str,
        ) and value.strip():
            return value

    raise RuntimeError(
        f"No generated source text found: "
        f"{row.get('sample_id')}"
    )


def validation_reasons(row):
    reasons = row.get(
        "validation_reasons"
    )

    if isinstance(reasons, list):
        return reasons

    a = chosen_attempt(row)

    if a is not None:
        reasons = a.get(
            "validation_reasons"
        )

        if isinstance(reasons, list):
            return reasons

    return []


def protocol_name(row):
    return (
        row.get("protocol_version")
        or row.get("source_generation_protocol")
        or "unknown"
    )


# =====================================================================
# Refuse overwrite
# =====================================================================

for p in (
    OUT_SOURCE,
    OUT_FAILURES,
    OUT_COVERAGE,
    OUT_FREEZE,
):
    if p.exists():
        raise RuntimeError(
            "\nCONFIRM48 source universe "
            "already frozen.\n"
            f"Refusing overwrite:\n{p}"
        )


print("=" * 104)
print("CONFIRM48 PHASE 5D — FREEZE SOURCE UNIVERSE")
print("=" * 104)


# =====================================================================
# 1. Frozen SHA verification
# =====================================================================

print("\n===== FROZEN INPUT SHA CHECK =====")

for p, expected in EXPECTED.items():

    actual = sha256(p)

    print(
        actual,
        p.relative_to(ROOT),
    )

    if actual != expected:
        raise RuntimeError(
            f"SHA mismatch:\n{p}\n"
            f"expected={expected}\n"
            f"actual={actual}"
        )


print("SOURCE GENERATION INPUTS: PASS")


# =====================================================================
# 2. Base manifest
# =====================================================================

base_obj = json.loads(
    BASE_MANIFEST.read_text(
        encoding="utf-8"
    )
)

base_docs = sorted(
    base_obj["documents"],
    key=lambda x:
        int(x["manifest_index"]),
)

if len(base_docs) != 48:
    raise RuntimeError(
        f"Expected 48 base IDs; got "
        f"{len(base_docs)}"
    )


base_ids = [
    x["base_id"]
    for x in base_docs
]

if len(set(base_ids)) != 48:
    raise RuntimeError(
        "Base manifest IDs not unique"
    )


base_by_id = {
    x["base_id"]: x
    for x in base_docs
}


# =====================================================================
# 3. Load Qwen + DeepSeek selected outputs
# =====================================================================

qwen = load_jsonl(
    QWEN_SELECTED
)

deep = load_jsonl(
    DEEP_SELECTED
)


if len(qwen) != 48:
    raise RuntimeError(
        f"Qwen selected rows !=48: {len(qwen)}"
    )

if len(deep) != 48:
    raise RuntimeError(
        f"DeepSeek selected rows !=48: {len(deep)}"
    )


def index_rows(rows, label):

    by_id = {}

    for r in rows:

        sid = (
            r.get("base_id")
            or r.get("sample_id")
        )

        if sid is None:
            raise RuntimeError(
                f"{label}: row missing base/sample ID"
            )

        if sid in by_id:
            raise RuntimeError(
                f"{label}: duplicate ID {sid}"
            )

        by_id[sid] = r

    if set(by_id) != set(base_ids):
        raise RuntimeError(
            f"{label}: base-ID mismatch\n"
            f"missing={sorted(set(base_ids)-set(by_id))}\n"
            f"extra={sorted(set(by_id)-set(base_ids))}"
        )

    return by_id


qwen_by_id = index_rows(
    qwen,
    "Qwen",
)

deep_by_id = index_rows(
    deep,
    "DeepSeek",
)


# =====================================================================
# 4. Validate metadata against frozen base manifest
# =====================================================================

for generator, rows in (
    (QWEN_NAME, qwen_by_id),
    (DEEP_NAME, deep_by_id),
):

    for bid, row in rows.items():

        b = base_by_id[bid]

        for key in (
            "domain",
            "source_id",
            "title",
        ):
            observed = row.get(key)
            expected = b.get(key)

            if observed != expected:
                raise RuntimeError(
                    f"{generator}: metadata mismatch "
                    f"{bid} {key}\n"
                    f"expected={expected!r}\n"
                    f"observed={observed!r}"
                )


print("BASE METADATA MATCH: PASS")


# =====================================================================
# 5. Build frozen valid source universe
#
# Order reproduces Fresh95-style organization:
#   base manifest order
#     Qwen
#     DeepSeek
#
# Invalid realizations are NOT included as machine-source instances.
# They are separately frozen as explicit missing realizations.
# =====================================================================

successful = []
failures = []


for b in base_docs:

    bid = b["base_id"]

    for generator, rows in (
        (QWEN_NAME, qwen_by_id),
        (DEEP_NAME, deep_by_id),
    ):

        row = rows[bid]
        vstatus = status(row)

        gkey = GENERATOR_KEY[
            generator
        ]

        machine_id = (
            f"{bid}::{gkey}"
        )

        if vstatus != "valid":

            failure = {
                "base_id":
                    bid,

                "sample_id":
                    bid,

                "machine_sample_id":
                    machine_id,

                "source_id":
                    b["source_id"],

                "domain":
                    b["domain"],

                "title":
                    b["title"],

                "source_generator":
                    generator,

                "generator_key":
                    gkey,

                "source_generation_protocol":
                    protocol_name(row),

                "source_validation_status":
                    vstatus,

                "validation_reasons":
                    validation_reasons(row),

                "failure_policy":
                    (
                        "Preserved as missing realization; "
                        "no rescue, no replacement, no resampling."
                    ),
            }

            failures.append(
                failure
            )

            continue


        text = extract_text(
            row
        )


        source_row = {
            "base_id":
                bid,

            "domain":
                b["domain"],

            "generator_key":
                gkey,

            "machine_sample_id":
                machine_id,

            "sample_id":
                bid,

            "source_generation_protocol":
                protocol_name(row),

            "source_generation_status":
                "frozen_confirmatory",

            "source_generator":
                generator,

            "source_id":
                b["source_id"],

            "source_text":
                text,

            "source_validation_status":
                "valid",

            "title":
                b["title"],
        }


        successful.append(
            source_row
        )


# =====================================================================
# 6. Scientific coverage checks
# =====================================================================

if len(successful) != 95:
    raise RuntimeError(
        f"Expected 95 successful source "
        f"realizations, got {len(successful)}"
    )

if len(failures) != 1:
    raise RuntimeError(
        f"Expected exactly 1 failed realization, "
        f"got {len(failures)}"
    )


machine_ids = [
    x["machine_sample_id"]
    for x in successful
]

if len(set(machine_ids)) != 95:
    raise RuntimeError(
        "machine_sample_id not unique"
    )


covered_base_ids = {
    x["base_id"]
    for x in successful
}

if len(covered_base_ids) != 48:
    raise RuntimeError(
        f"Expected all 48 base clusters represented; "
        f"got {len(covered_base_ids)}"
    )


generator_counts = Counter(
    x["source_generator"]
    for x in successful
)

domain_counts = Counter(
    x["domain"]
    for x in successful
)

generator_domain = defaultdict(
    Counter
)

for x in successful:
    generator_domain[
        x["source_generator"]
    ][
        x["domain"]
    ] += 1


expected_generator_counts = {
    QWEN_NAME: 47,
    DEEP_NAME: 48,
}

if dict(generator_counts) != expected_generator_counts:
    raise RuntimeError(
        f"Unexpected generator coverage: "
        f"{dict(generator_counts)}"
    )


failure = failures[0]

if failure[
    "source_generator"
] != QWEN_NAME:
    raise RuntimeError(
        "Observed failure is not the expected "
        "single Qwen realization."
    )


# =====================================================================
# 7. Write frozen source artifacts
# =====================================================================

write_jsonl_atomic(
    OUT_SOURCE,
    successful,
)

write_jsonl_atomic(
    OUT_FAILURES,
    failures,
)


source_sha = sha256(
    OUT_SOURCE
)

failure_sha = sha256(
    OUT_FAILURES
)


# =====================================================================
# 8. Hash all raw source-generation artifacts
# =====================================================================

raw_artifacts = {}

for directory in (
    QWEN_DIR,
    DEEP_DIR,
):

    for p in sorted(
        x
        for x in directory.iterdir()
        if x.is_file()
    ):
        raw_artifacts[
            str(
                p.relative_to(ROOT)
            )
        ] = sha256(p)


# =====================================================================
# 9. Coverage audit
# =====================================================================

coverage = {
    "status":
        "PASS",

    "experiment":
        "CONFIRM48_V1",

    "stage":
        "frozen source generation coverage",

    "expected_realizations":
        96,

    "successful_realizations":
        95,

    "failed_realizations":
        1,

    "underlying_base_clusters":
        48,

    "successful_by_generator":
        dict(generator_counts),

    "successful_by_domain":
        dict(domain_counts),

    "successful_by_generator_and_domain": {
        k: dict(v)
        for k, v in generator_domain.items()
    },

    "failure": failure,

    "policy": {
        "replacement_base_id":
            False,

        "failure_rescue":
            False,

        "imputation":
            False,

        "resampling":
            False,

        "scientific_handling":
            (
                "The failed Qwen source realization remains "
                "missing. All downstream conditions operate "
                "only on the 95 successfully generated "
                "machine-source realizations while inference "
                "remains clustered over all 48 frozen RAID "
                "base IDs."
            ),
    },

    "source_file":
        str(
            OUT_SOURCE.relative_to(ROOT)
        ),

    "source_sha256":
        source_sha,

    "failures_file":
        str(
            OUT_FAILURES.relative_to(ROOT)
        ),

    "failures_sha256":
        failure_sha,
}


write_json_atomic(
    OUT_COVERAGE,
    coverage,
)


coverage_sha = sha256(
    OUT_COVERAGE
)


# =====================================================================
# 10. Source freeze manifest
# =====================================================================

freeze = {
    "status":
        "FROZEN",

    "experiment":
        "CONFIRM48_V1",

    "stage":
        "source universe",

    "base_manifest": {
        "path":
            str(
                BASE_MANIFEST.relative_to(ROOT)
            ),

        "sha256":
            sha256(BASE_MANIFEST),

        "base_ids":
            48,
    },

    "pre_generation_manifest": {
        "path":
            str(
                PREP_MANIFEST.relative_to(ROOT)
            ),

        "sha256":
            sha256(PREP_MANIFEST),
    },

    "raw_source_generation_artifacts":
        raw_artifacts,

    "coverage": {
        "possible_realizations":
            96,

        "valid_realizations":
            95,

        "missing_realizations":
            1,

        "qwen_valid":
            generator_counts[
                QWEN_NAME
            ],

        "deepseek_valid":
            generator_counts[
                DEEP_NAME
            ],

        "base_clusters":
            len(
                covered_base_ids
            ),
    },

    "failure": failure,

    "canonical_machine_source_file": {
        "path":
            str(
                OUT_SOURCE.relative_to(ROOT)
            ),

        "sha256":
            source_sha,

        "rows":
            len(successful),

        "schema":
            sorted(
                successful[0].keys()
            ),
    },

    "failure_file": {
        "path":
            str(
                OUT_FAILURES.relative_to(ROOT)
            ),

        "sha256":
            failure_sha,

        "rows":
            len(failures),
    },

    "coverage_audit": {
        "path":
            str(
                OUT_COVERAGE.relative_to(ROOT)
            ),

        "sha256":
            coverage_sha,
    },

    "downstream_rule":
        (
            "Original, Vanilla, and HPCD-n0 use exactly the "
            "95 valid machine-source realizations in the "
            "canonical source file. The failed Qwen "
            "realization is not replaced."
        ),
}


write_json_atomic(
    OUT_FREEZE,
    freeze,
)


# =====================================================================
# 11. Final console report
# =====================================================================

print()
print("=" * 104)
print("CONFIRM48 SOURCE COVERAGE")
print("=" * 104)

print(
    "possible realizations : 96"
)

print(
    "successful           :",
    len(successful),
)

print(
    "failed               :",
    len(failures),
)

print(
    "base clusters        :",
    len(covered_base_ids),
)

print(
    "by generator         :",
    dict(generator_counts),
)

print(
    "by domain            :",
    dict(domain_counts),
)


print()
print("===== GENERATOR × DOMAIN =====")

for generator in (
    QWEN_NAME,
    DEEP_NAME,
):
    print(
        generator,
        dict(
            generator_domain[
                generator
            ]
        ),
    )


print()
print("===== FROZEN MISSING REALIZATION =====")

print(
    json.dumps(
        failure,
        indent=2,
        ensure_ascii=False,
    )
)


print()
print("=" * 104)
print("FROZEN SOURCE ARTIFACT HASHES")
print("=" * 104)

for p in (
    OUT_SOURCE,
    OUT_FAILURES,
    OUT_COVERAGE,
    OUT_FREEZE,
):
    print(
        sha256(p),
        p.relative_to(ROOT),
    )


print()
print(
    "CONFIRM48 SOURCE UNIVERSE FREEZE: PASS"
)
print(
    "DO NOT REGENERATE THE FAILED QWEN REALIZATION."
)

