#!/usr/bin/env python3

from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
import re
import unicodedata


ROOT = Path(os.environ.get("HPCD_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))

CANDIDATE = ROOT / "data/processed/raid_fresh_candidate160_v1.jsonl"
CANDIDATE_CFG = ROOT / "configs/raid_fresh_candidate160_acquisition_v1.json"

OLD400 = ROOT / "data/processed/raid_pilot_human.jsonl"
OLD400_IDS = ROOT / "data/splits/raid_pilot_ids.json"
OLD400_SPLIT = ROOT / "data/splits/raid_pilot_split_seed2027.json"

FRESH48_MANIFEST = ROOT / "data/splits/raid_fresh48_manifest_v1.json"
FRESH48_CFG = ROOT / "configs/raid_fresh48_selection_v1.json"

DEV111 = ROOT / "results/hpcd/dev111_recovery_v1.json"

# Frozen human trajectory material.
HJTM_MANIFEST = ROOT / "results/hjtm/t1b_full_mac_v1/manifest.jsonl"

HPCD_ROOT = ROOT / "results/hpcd"

OUTDIR = ROOT / "results/confirm48_v1/manifests"

OUT_JSON = OUTDIR / "confirm48_base_ids_v1.json"
OUT_CSV = OUTDIR / "confirm48_base_ids_v1.csv"
OUT_AUDIT = OUTDIR / "confirm48_overlap_audit_v1.json"

EXPECTED_CANDIDATE_SHA = (
    "b21d22edd876a2f89daf8664818a6062"
    "89b05221ec5bd5517cf560918fcf4899"
)

EXPECTED_FRESH48_MANIFEST_SHA = (
    "375cd0387597b8b4b1a1ff004b01e762"
    "4df540246f678ddc7066168876027bc7"
)

DOMAINS = ("news", "reddit", "reviews", "wiki")
SELECTION_SEED = 43
N_PER_DOMAIN = 12

ID_RE = re.compile(
    r"raid_train_(?:news|reddit|reviews|wiki)_[0-9a-f]{16}"
)


# =====================================================================
# HELPERS
# =====================================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def load_jsonl(path: Path):
    """
    IMPORTANT:
    Iterate on actual '\\n' record delimiters.
    Do NOT use str.splitlines(), because RAID review text can contain
    Unicode line-separator characters inside valid JSON strings.
    """
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        for line_no, line in enumerate(f, 1):

            if not line.strip():
                continue

            try:
                rows.append(
                    json.loads(line)
                )
            except Exception as exc:
                raise RuntimeError(
                    f"{path}: malformed JSON record "
                    f"at physical newline record {line_no}: {exc}"
                ) from exc

    return rows


def normalize_overlap_text(text: str) -> str:
    """
    Project canonical surface normalization:
      NFKC
      normalize curly quotes / long dashes
      casefold
      collapse Unicode whitespace
    """
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


def normalized_hash(text: str) -> str:
    return sha256_text(
        normalize_overlap_text(text)
    )


def rank64(seed, domain, source_id):
    payload = (
        f"{seed}|{domain}|{source_id}"
    ).encode("utf-8")

    # Existing Fresh48 rule:
    # first 64 bits of SHA256(seed | domain | source_id)
    return int(
        hashlib.sha256(
            payload
        ).hexdigest()[:16],
        16,
    )


def recursive_raid_ids(path: Path):
    """
    Conservatively recover any prior RAID base/sample IDs mentioned
    anywhere in frozen HPCD results.
    """
    ids = set()
    evidence = {}

    suffixes = {
        ".json",
        ".jsonl",
        ".csv",
        ".txt",
        ".md",
    }

    for p in sorted(
        x for x in path.rglob("*")
        if x.is_file()
        and x.suffix.lower() in suffixes
    ):
        try:
            text = p.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except Exception:
            continue

        found = set(
            ID_RE.findall(text)
        )

        if found:
            ids.update(found)
            evidence[
                str(p.relative_to(ROOT))
            ] = len(found)

    return ids, evidence


def atomic_json(path: Path, obj):
    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    with tmp.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            obj,
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.write("\n")

    os.replace(tmp, path)


# =====================================================================
# REFUSE ACCIDENTAL RESAMPLING
# =====================================================================

for p in (
    OUT_JSON,
    OUT_CSV,
    OUT_AUDIT,
):
    if p.exists():
        raise RuntimeError(
            "\nCONFIRM48 split already exists.\n"
            f"Refusing to resample or overwrite:\n{p}"
        )


print("=" * 104)
print("CONFIRM48 PHASE 2/3 — FROZEN SPLIT CONSTRUCTION")
print("=" * 104)


# =====================================================================
# 1. VERIFY CANDIDATE160
# =====================================================================

candidate_sha = sha256_file(
    CANDIDATE
)

print("\n===== CANDIDATE160 =====")
print("SHA256 :", candidate_sha)

if candidate_sha != EXPECTED_CANDIDATE_SHA:
    raise RuntimeError(
        "candidate160 SHA mismatch"
    )


candidate = load_jsonl(
    CANDIDATE
)

candidate_counts = Counter(
    r["domain"]
    for r in candidate
)

print("Rows   :", len(candidate))
print("Domains:", dict(candidate_counts))


if len(candidate) != 640:
    raise RuntimeError(
        f"Expected 640 candidate rows, got {len(candidate)}"
    )

if candidate_counts != Counter({
    "news": 160,
    "reddit": 160,
    "reviews": 160,
    "wiki": 160,
}):
    raise RuntimeError(
        f"candidate160 domain mismatch: {candidate_counts}"
    )


sample_ids = [
    r["sample_id"]
    for r in candidate
]

source_ids = [
    r["source_id"]
    for r in candidate
]

if len(set(sample_ids)) != 640:
    raise RuntimeError(
        "candidate160 sample IDs not unique"
    )

if len(set(source_ids)) != 640:
    raise RuntimeError(
        "candidate160 source IDs not unique"
    )


# =====================================================================
# 2. VERIFY STORED FRESH48 RANK FUNCTION
# =====================================================================

rank_mismatches = []

for r in candidate:

    expected = rank64(
        SELECTION_SEED,
        r["domain"],
        r["source_id"],
    )

    observed = int(
        r["selection_hash64"]
    )

    if expected != observed:
        rank_mismatches.append({
            "sample_id":
                r["sample_id"],
            "expected":
                expected,
            "observed":
                observed,
        })


if rank_mismatches:
    raise RuntimeError(
        "Fresh48 deterministic rank function "
        "did not reproduce candidate160.\n"
        + json.dumps(
            rank_mismatches[:5],
            indent=2,
        )
    )


print(
    "Rank function reproduction : PASS "
    "(640/640)"
)


# =====================================================================
# 3. LOAD OLD400 + VERIFY ITS FROZEN HASH
# =====================================================================

old_meta = json.loads(
    OLD400_IDS.read_text(
        encoding="utf-8"
    )
)

expected_old_sha = old_meta[
    "jsonl_sha256"
]

actual_old_sha = sha256_file(
    OLD400
)

if actual_old_sha != expected_old_sha:
    raise RuntimeError(
        "\nold400 SHA mismatch\n"
        f"expected={expected_old_sha}\n"
        f"actual={actual_old_sha}"
    )


old_rows = load_jsonl(
    OLD400
)

if len(old_rows) != 400:
    raise RuntimeError(
        f"Expected old400=400, got {len(old_rows)}"
    )


old_sample_ids = {
    r["sample_id"]
    for r in old_rows
}

old_source_ids = {
    r["source_id"]
    for r in old_rows
}

old_raw_text_hashes = {
    sha256_text(
        r["human_reference"]
    )
    for r in old_rows
}

old_norm_text_hashes = {
    normalized_hash(
        r["human_reference"]
    )
    for r in old_rows
}


print("\n===== OLD400 =====")
print("Rows:", len(old_rows))
print("SHA :", actual_old_sha)


# =====================================================================
# 4. LOAD FRESH48
# =====================================================================

fresh_cfg = json.loads(
    FRESH48_CFG.read_text(
        encoding="utf-8"
    )
)

fresh_manifest_sha = sha256_file(
    FRESH48_MANIFEST
)

if (
    fresh_manifest_sha
    !=
    EXPECTED_FRESH48_MANIFEST_SHA
):
    raise RuntimeError(
        "Fresh48 manifest SHA mismatch"
    )

if (
    fresh_cfg["manifest_sha256"]
    !=
    fresh_manifest_sha
):
    raise RuntimeError(
        "Fresh48 config/manifest SHA disagreement"
    )


fresh_manifest = json.loads(
    FRESH48_MANIFEST.read_text(
        encoding="utf-8"
    )
)

fresh_docs = fresh_manifest[
    "documents"
]

if len(fresh_docs) != 48:
    raise RuntimeError(
        f"Expected 48 Fresh48 documents, got {len(fresh_docs)}"
    )


fresh_sample_ids = {
    d["sample_id"]
    for d in fresh_docs
}

fresh_source_ids = {
    d["source_id"]
    for d in fresh_docs
}

fresh_norm_text_hashes = {
    normalized_hash(
        d["human_text"]
    )
    for d in fresh_docs
}


# =====================================================================
# 5. REPRODUCE HISTORICAL FRESH48 EXACTLY
#
# Historical rule:
# - candidate160 source universe
# - exclude original400 sample/source/exact raw human-text hash
# - seed 43
# - SHA256(seed | domain | source_id)
# - take first 12/domain
# =====================================================================

historical_eligible = defaultdict(
    list
)

for r in candidate:

    raw_h = sha256_text(
        r["human_reference"]
    )

    if (
        r["sample_id"]
        in old_sample_ids
    ):
        continue

    if (
        r["source_id"]
        in old_source_ids
    ):
        continue

    if (
        raw_h
        in old_raw_text_hashes
    ):
        continue

    historical_eligible[
        r["domain"]
    ].append(r)


historical_selected = {}

for domain in DOMAINS:

    ranked = sorted(
        historical_eligible[domain],
        key=lambda r: (
            int(
                r["selection_hash64"]
            ),
            r["sample_id"],
        ),
    )

    historical_selected[
        domain
    ] = [
        r["sample_id"]
        for r in ranked[:12]
    ]


frozen_fresh_by_domain = defaultdict(
    list
)

for d in fresh_docs:
    frozen_fresh_by_domain[
        d["domain"]
    ].append(d)


for domain in DOMAINS:

    frozen_order = [
        d["sample_id"]
        for d in sorted(
            frozen_fresh_by_domain[
                domain
            ],
            key=lambda x:
                int(
                    x[
                        "selection_rank_after_original400_exclusion"
                    ]
                ),
        )
    ]

    if (
        historical_selected[
            domain
        ]
        !=
        frozen_order
    ):
        raise RuntimeError(
            "\nHISTORICAL FRESH48 REPRODUCTION FAILED\n"
            f"domain={domain}\n"
            f"reproduced={historical_selected[domain]}\n"
            f"frozen={frozen_order}"
        )


print()
print("=" * 104)
print("HISTORICAL FRESH48 SELECTION REPRODUCTION: PASS")
print("=" * 104)


# =====================================================================
# 6. HUMAN SIGNATURE SET
# =====================================================================

split_obj = json.loads(
    OLD400_SPLIT.read_text(
        encoding="utf-8"
    )
)

signature_ids = {
    x["sample_id"]
    for x in split_obj[
        "assignments"
    ]
    if (
        x.get("partition")
        ==
        "signature_train"
    )
}

if len(signature_ids) != 280:
    raise RuntimeError(
        f"Expected 280 signature IDs, got {len(signature_ids)}"
    )


# =====================================================================
# 7. DEV111
# =====================================================================

dev_obj = json.loads(
    DEV111.read_text(
        encoding="utf-8"
    )
)

dev111_ids = set(
    dev_obj["ids"]
)

if len(dev111_ids) != 111:
    raise RuntimeError(
        f"Expected DEV111=111, got {len(dev111_ids)}"
    )


# =====================================================================
# 8. HUMAN TRAJECTORY IDs
# =====================================================================

hjtm_rows = load_jsonl(
    HJTM_MANIFEST
)

trajectory_ids = {
    r["sample_id"]
    for r in hjtm_rows
    if r.get(
        "source_type"
    ) == "human"
}


# =====================================================================
# 9. RECOVER ALL PRIOR HPCD IDS CONSERVATIVELY
# =====================================================================

prior_hpcd_ids, hpcd_evidence = (
    recursive_raid_ids(
        HPCD_ROOT
    )
)


print("\n===== EXCLUSION SET COUNTS =====")
print("old400                 :", len(old_sample_ids))
print("human signature        :", len(signature_ids))
print("human trajectory       :", len(trajectory_ids))
print("DEV111                 :", len(dev111_ids))
print("Fresh48/Fresh95        :", len(fresh_sample_ids))
print("all IDs in HPCD results:", len(prior_hpcd_ids))


# =====================================================================
# 10. COMBINED PRIOR-ID SET
# =====================================================================

prior_ids = (
    old_sample_ids
    |
    signature_ids
    |
    trajectory_ids
    |
    dev111_ids
    |
    fresh_sample_ids
    |
    prior_hpcd_ids
)


candidate_by_sample = {
    r["sample_id"]: r
    for r in candidate
}


prior_source_ids = (
    set(old_source_ids)
    |
    set(fresh_source_ids)
)


# If a prior HPCD/HJTM ID is present in the candidate160 universe,
# also exclude its source UUID.
for sid in prior_ids:

    row = candidate_by_sample.get(
        sid
    )

    if row is not None:
        prior_source_ids.add(
            row["source_id"]
        )


# =====================================================================
# 11. EXACT NORMALIZED-TEXT EXCLUSION SET
# =====================================================================

prior_norm_hashes = (
    set(old_norm_text_hashes)
    |
    set(fresh_norm_text_hashes)
)


# Any previously used base ID that is represented inside candidate160
# contributes its clean-human text to the normalized overlap exclusion.
for sid in prior_ids:

    row = candidate_by_sample.get(
        sid
    )

    if row is not None:
        prior_norm_hashes.add(
            normalized_hash(
                row[
                    "human_reference"
                ]
            )
        )


# =====================================================================
# 12. BUILD CONFIRM48 ELIGIBLE POOL
# =====================================================================

eligible_by_domain = defaultdict(
    list
)

exclusion_reasons = Counter()


for r in candidate:

    sid = r["sample_id"]
    source_id = r["source_id"]

    nh = normalized_hash(
        r["human_reference"]
    )

    reasons = []

    if sid in prior_ids:
        reasons.append(
            "prior_base_id"
        )

    if source_id in prior_source_ids:
        reasons.append(
            "prior_source_id"
        )

    if nh in prior_norm_hashes:
        reasons.append(
            "prior_normalized_text"
        )

    if reasons:
        for reason in set(reasons):
            exclusion_reasons[
                reason
            ] += 1
        continue

    eligible_by_domain[
        r["domain"]
    ].append(r)


print("\n===== CONFIRM48 ELIGIBLE POOL =====")

for domain in DOMAINS:
    print(
        f"{domain:<8}: "
        f"{len(eligible_by_domain[domain])}"
    )

    if (
        len(
            eligible_by_domain[
                domain
            ]
        )
        <
        N_PER_DOMAIN
    ):
        raise RuntimeError(
            f"Insufficient eligible "
            f"{domain} rows"
        )


# =====================================================================
# 13. DETERMINISTIC CONFIRM48 SELECTION
#
# Reuse exact Fresh48 rank:
# SHA256(43 | domain | source_id)
# =====================================================================

selected = []


for domain in DOMAINS:

    ranked = sorted(
        eligible_by_domain[
            domain
        ],
        key=lambda r: (
            int(
                r["selection_hash64"]
            ),
            r["sample_id"],
        ),
    )

    for rank, r in enumerate(
        ranked,
        1,
    ):

        if rank > N_PER_DOMAIN:
            break

        human_text = (
            r["human_reference"]
        )

        selected.append({
            "manifest_index":
                len(selected),

            "sample_id":
                r["sample_id"],

            "base_id":
                r["sample_id"],

            "domain":
                r["domain"],

            "dataset":
                r.get(
                    "dataset",
                    "RAID",
                ),

            "split":
                r.get(
                    "split",
                    "train",
                ),

            "source_id":
                r["source_id"],

            "raid_id":
                r.get(
                    "raid_id"
                ),

            "adv_source_id":
                r.get(
                    "adv_source_id"
                ),

            "title":
                r.get(
                    "title"
                ),

            "human_text":
                human_text,

            "human_text_sha256":
                sha256_text(
                    human_text
                ),

            "human_text_normalized_sha256":
                normalized_hash(
                    human_text
                ),

            "selection_seed":
                SELECTION_SEED,

            "selection_hash64":
                int(
                    r[
                        "selection_hash64"
                    ]
                ),

            "selection_rank_after_all_exclusions":
                rank,

            "source_record_byte_offset":
                r.get(
                    "source_record_byte_offset"
                ),
        })


if len(selected) != 48:
    raise RuntimeError(
        f"Expected 48 selected, got {len(selected)}"
    )


selected_counts = Counter(
    r["domain"]
    for r in selected
)

expected_counts = Counter({
    "news": 12,
    "reddit": 12,
    "reviews": 12,
    "wiki": 12,
})

if selected_counts != expected_counts:
    raise RuntimeError(
        f"Selected domain counts mismatch: "
        f"{selected_counts}"
    )


# =====================================================================
# 14. FINAL ZERO-OVERLAP AUDIT
# =====================================================================

selected_ids = {
    r["base_id"]
    for r in selected
}

selected_source_ids = {
    r["source_id"]
    for r in selected
}

selected_norm_hashes = {
    r[
        "human_text_normalized_sha256"
    ]
    for r in selected
}

selected_raw_hashes = {
    r[
        "human_text_sha256"
    ]
    for r in selected
}


checks = {
    "selected_base_ids":
        len(selected_ids),

    "selected_source_ids":
        len(selected_source_ids),

    "selected_raw_text_hashes":
        len(selected_raw_hashes),

    "selected_normalized_text_hashes":
        len(selected_norm_hashes),

    "news":
        selected_counts["news"],

    "reddit":
        selected_counts["reddit"],

    "reviews":
        selected_counts["reviews"],

    "wiki":
        selected_counts["wiki"],

    "old400_id_overlap":
        len(
            selected_ids
            &
            old_sample_ids
        ),

    "human_signature_id_overlap":
        len(
            selected_ids
            &
            signature_ids
        ),

    "human_trajectory_id_overlap":
        len(
            selected_ids
            &
            trajectory_ids
        ),

    "dev111_id_overlap":
        len(
            selected_ids
            &
            dev111_ids
        ),

    "fresh95_fresh48_id_overlap":
        len(
            selected_ids
            &
            fresh_sample_ids
        ),

    "prior_hpcd_id_overlap":
        len(
            selected_ids
            &
            prior_hpcd_ids
        ),

    "prior_source_id_overlap":
        len(
            selected_source_ids
            &
            prior_source_ids
        ),

    "normalized_text_overlap_all_prior":
        len(
            selected_norm_hashes
            &
            prior_norm_hashes
        ),
}


required_zero = [
    "old400_id_overlap",
    "human_signature_id_overlap",
    "human_trajectory_id_overlap",
    "dev111_id_overlap",
    "fresh95_fresh48_id_overlap",
    "prior_hpcd_id_overlap",
    "prior_source_id_overlap",
    "normalized_text_overlap_all_prior",
]


for key in required_zero:
    if checks[key] != 0:
        raise RuntimeError(
            f"Overlap audit failed: "
            f"{key}={checks[key]}"
        )


if len(selected_ids) != 48:
    raise RuntimeError(
        "Selected base IDs not unique"
    )

if len(selected_source_ids) != 48:
    raise RuntimeError(
        "Selected source IDs not unique"
    )

if len(selected_raw_hashes) != 48:
    raise RuntimeError(
        "Selected raw human texts not unique"
    )

if len(selected_norm_hashes) != 48:
    raise RuntimeError(
        "Selected normalized human texts not unique"
    )


# =====================================================================
# 15. ONLY NOW CREATE FROZEN OUTPUTS
# =====================================================================

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


manifest = {
    "status":
        "FROZEN",

    "experiment":
        "CONFIRM48_V1",

    "dataset":
        "liamdugan/raid",

    "split":
        "train",

    "source_universe": {
        "file":
            str(
                CANDIDATE.relative_to(
                    ROOT
                )
            ),

        "sha256":
            candidate_sha,

        "rows":
            len(candidate),

        "domain_counts":
            dict(
                candidate_counts
            ),
    },

    "selection": {
        "seed":
            SELECTION_SEED,

        "rule":
            "ascending SHA256(seed | domain | source_id), first 64 bits",

        "n_per_domain":
            N_PER_DOMAIN,

        "fresh48_historical_selector_reproduced":
            True,

        "fresh48_manifest_sha256":
            fresh_manifest_sha,
    },

    "exclusions": {
        "old400_count":
            len(old_sample_ids),

        "human_signature_count":
            len(signature_ids),

        "human_trajectory_count":
            len(trajectory_ids),

        "dev111_count":
            len(dev111_ids),

        "fresh95_fresh48_count":
            len(fresh_sample_ids),

        "prior_hpcd_recovered_count":
            len(prior_hpcd_ids),

        "combined_prior_base_id_count":
            len(prior_ids),

        "combined_prior_source_id_count":
            len(prior_source_ids),

        "combined_prior_normalized_text_hash_count":
            len(prior_norm_hashes),

        "hpcd_id_evidence_files":
            hpcd_evidence,
    },

    "canonical_overlap_normalization": {
        "unicode":
            "NFKC",

        "punctuation":
            "curly quotes to ASCII; en/em dash to hyphen",

        "case":
            "casefold",

        "whitespace":
            "Unicode whitespace collapsed to single ASCII space",
    },

    "eligible_pool_after_all_exclusions": {
        d:
            len(
                eligible_by_domain[d]
            )
        for d in DOMAINS
    },

    "counts": {
        "total":
            48,

        "by_domain":
            dict(
                selected_counts
            ),
    },

    "documents":
        selected,
}


audit = {
    "status":
        "PASS",

    "experiment":
        "CONFIRM48_V1",

    "candidate_source_sha256":
        candidate_sha,

    "historical_fresh48_selection_reproduction":
        "PASS",

    "selection_seed":
        SELECTION_SEED,

    "selection_rule":
        "SHA256(seed | domain | source_id)",

    "selected_counts":
        dict(
            selected_counts
        ),

    "checks":
        checks,

    "exclusion_reason_counts_within_candidate160":
        dict(
            exclusion_reasons
        ),

    "source_files": {
        "candidate160": {
            "path":
                str(
                    CANDIDATE.relative_to(
                        ROOT
                    )
                ),

            "sha256":
                candidate_sha,
        },

        "old400": {
            "path":
                str(
                    OLD400.relative_to(
                        ROOT
                    )
                ),

            "sha256":
                actual_old_sha,
        },

        "old400_split": {
            "path":
                str(
                    OLD400_SPLIT.relative_to(
                        ROOT
                    )
                ),

            "sha256":
                sha256_file(
                    OLD400_SPLIT
                ),
        },

        "fresh48_manifest": {
            "path":
                str(
                    FRESH48_MANIFEST.relative_to(
                        ROOT
                    )
                ),

            "sha256":
                fresh_manifest_sha,
        },

        "dev111": {
            "path":
                str(
                    DEV111.relative_to(
                        ROOT
                    )
                ),

            "sha256":
                sha256_file(
                    DEV111
                ),
        },

        "human_trajectory_manifest": {
            "path":
                str(
                    HJTM_MANIFEST.relative_to(
                        ROOT
                    )
                ),

            "sha256":
                sha256_file(
                    HJTM_MANIFEST
                ),
        },
    },

    "conclusion":
        (
            "48 new RAID base IDs were deterministically selected "
            "with zero recovered prior-ID, source-ID, or exact "
            "normalized-human-text overlap."
        ),
}


atomic_json(
    OUT_JSON,
    manifest,
)

atomic_json(
    OUT_AUDIT,
    audit,
)


csv_fields = [
    "manifest_index",
    "base_id",
    "sample_id",
    "domain",
    "dataset",
    "split",
    "source_id",
    "raid_id",
    "adv_source_id",
    "title",
    "human_text_sha256",
    "human_text_normalized_sha256",
    "selection_seed",
    "selection_hash64",
    "selection_rank_after_all_exclusions",
    "source_record_byte_offset",
    "human_text",
]


tmp_csv = OUT_CSV.with_suffix(
    ".csv.tmp"
)

with tmp_csv.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=csv_fields,
    )

    writer.writeheader()

    for row in selected:
        writer.writerow(
            {
                k: row.get(k)
                for k in csv_fields
            }
        )


os.replace(
    tmp_csv,
    OUT_CSV,
)


# =====================================================================
# 16. PUBLIC FREEZE REPORT
# =====================================================================

print()
print("=" * 104)
print("CONFIRM48 SELECTED BASE IDS — NOW FROZEN")
print("=" * 104)


for domain in DOMAINS:

    print(
        f"\n[{domain.upper()}]"
    )

    domain_rows = [
        r
        for r in selected
        if r["domain"] == domain
    ]

    for r in domain_rows:
        print(
            f"{r['selection_rank_after_all_exclusions']:02d}  "
            f"{r['base_id']}  "
            f"source={r['source_id']}  "
            f"text_sha={r['human_text_sha256']}"
        )


print()
print("=" * 104)
print("OVERLAP AUDIT")
print("=" * 104)

for key, value in checks.items():
    print(
        f"{key:<40}: {value}"
    )


print()
print("=" * 104)
print("FROZEN ARTIFACT HASHES")
print("=" * 104)

for p in (
    OUT_JSON,
    OUT_CSV,
    OUT_AUDIT,
):
    print(
        sha256_file(p),
        p.relative_to(ROOT),
    )


print()
print(
    "CONFIRM48 SPLIT FREEZE: PASS"
)
print(
    "DO NOT RESAMPLE THESE 48 IDS."
)

