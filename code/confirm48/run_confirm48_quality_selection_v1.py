#!/usr/bin/env python3
from __future__ import annotations

import os

import hashlib
import importlib.util
import json
import platform
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import spacy


ROOT = Path(os.environ.get("HPCD_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))

CANDIDATES = (
    ROOT
    / "results/confirm48_v1/generations/"
      "confirm48_final_candidates_v1/candidates.jsonl"
)

GEN_FREEZE = (
    ROOT
    / "results/confirm48_v1/manifests/"
      "confirm48_candidate_generation_freeze_v1.json"
)

REWRITE_MANIFEST = (
    ROOT
    / "results/confirm48_v1/manifests/"
      "confirm48_rewrite_manifest_v1.json"
)

FRESH_QUALITY = (
    ROOT
    / "code/hpcd/"
      "analyze_fresh95_quality_selection_v1.py"
)

FRESH_ENTITY = (
    ROOT
    / "code/hpcd/"
      "finalize_fresh95_surface_entity_selection_v1.py"
)

OUT_DIR = (
    ROOT
    / "results/confirm48_v1/quality/"
      "confirm48_quality_selection_v1"
)

RESCORED = OUT_DIR / "rescored_candidates.jsonl"
SELECTED = OUT_DIR / "selected_outputs.jsonl"
SUMMARY = OUT_DIR / "summary.json"

EXPECTED_CANDIDATE_SHA = (
    "ac46b7a1df18db1b49317aae71f6835b"
    "9e249e62e2aecaefdb0cfb5d593e76d3"
)

EXPECTED_GENERATION_FREEZE_SHA = (
    "6de84fb9328e0a26666e0be35035057f"
    "a080263c6d1347420baade107dad52d0"
)

EXPECTED_FRESH_QUALITY_SHA = (
    "a1ba438278898dd7982380e77803fbcae"
    "3c9bea9baeda6a8911e89bedda9ff24"
)

EXPECTED_FRESH_ENTITY_SHA = (
    "319fc639a7c705c4eeaf779e4ae50740f"
    "bd919b98bdfc0390c45ffebe9e78b0f"
)

CONDITIONS = (
    "vanilla_t10",
    "hpcd_n0",
)

K = 4
ATTEMPTS = 2
SEMANTIC_THRESHOLD = 0.92


def sha256(path: Path) -> str:
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
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                rows.append(
                    json.loads(line)
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Invalid JSON at record {n}: {path}"
                ) from exc

    return rows


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


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    module = importlib.util.module_from_spec(
        spec
    )

    assert spec.loader is not None
    spec.loader.exec_module(module)

    return module


def main():

    print("=" * 104)
    print(
        "CONFIRM48 PHASE 7A — "
        "FROZEN QUALITY-FIRST SELECTION"
    )
    print("=" * 104)

    # -----------------------------------------------------------------
    # Immutable provenance checks
    # -----------------------------------------------------------------

    checks = {
        CANDIDATES:
            EXPECTED_CANDIDATE_SHA,

        GEN_FREEZE:
            EXPECTED_GENERATION_FREEZE_SHA,

        FRESH_QUALITY:
            EXPECTED_FRESH_QUALITY_SHA,

        FRESH_ENTITY:
            EXPECTED_FRESH_ENTITY_SHA,
    }

    print()
    print("===== PROVENANCE SHA CHECK =====")

    for path, expected in checks.items():

        actual = sha256(path)

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

    print("PROVENANCE: PASS")

    if OUT_DIR.exists():
        raise RuntimeError(
            f"Refusing overwrite: {OUT_DIR}"
        )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    # -----------------------------------------------------------------
    # Reuse EXACT frozen Fresh95 helper implementations
    # -----------------------------------------------------------------

    q = load_module(
        "fresh95_quality_frozen",
        FRESH_QUALITY,
    )

    ent = load_module(
        "fresh95_entity_frozen",
        FRESH_ENTITY,
    )

    rows = load_jsonl(
        CANDIDATES
    )

    if len(rows) != 1520:
        raise RuntimeError(
            f"Expected 1520 candidates, got {len(rows)}"
        )

    condition_counts = Counter(
        r["condition"]
        for r in rows
    )

    if condition_counts != Counter({
        "vanilla_t10": 760,
        "hpcd_n0": 760,
    }):
        raise RuntimeError(
            f"Bad condition counts: {condition_counts}"
        )

    instances = sorted({
        r["instance_id"]
        for r in rows
    })

    if len(instances) != 95:
        raise RuntimeError(
            f"Expected 95 instances, got {len(instances)}"
        )

    keys = Counter(
        (
            r["instance_id"],
            r["condition"],
            int(r["slot"]),
            int(r["attempt"]),
        )
        for r in rows
    )

    expected_keys = {
        (
            iid,
            condition,
            slot,
            attempt,
        )
        for iid in instances
        for condition in CONDITIONS
        for slot in range(K)
        for attempt in range(ATTEMPTS)
    }

    if set(keys) != expected_keys:
        raise RuntimeError(
            "Candidate key grid is incomplete"
        )

    if any(v != 1 for v in keys.values()):
        raise RuntimeError(
            "Duplicate candidate key detected"
        )

    # -----------------------------------------------------------------
    # Base-ID mapping
    # -----------------------------------------------------------------

    manifest = json.loads(
        REWRITE_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    docs = manifest["documents"]

    doc_by_instance = {
        d["instance_id"]: d
        for d in docs
    }

    if set(doc_by_instance) != set(instances):
        raise RuntimeError(
            "Rewrite manifest membership mismatch"
        )

    base_ids = {
        d["base_id"]
        for d in docs
    }

    if len(base_ids) != 48:
        raise RuntimeError(
            f"Expected 48 base clusters, got {len(base_ids)}"
        )

    # -----------------------------------------------------------------
    # Source consistency
    # -----------------------------------------------------------------

    source_by_instance = {}

    for row in rows:
        iid = row["instance_id"]
        source = row["source_text"]

        if (
            iid in source_by_instance
            and source_by_instance[iid] != source
        ):
            raise RuntimeError(
                f"Inconsistent source_text for {iid}"
            )

        source_by_instance[iid] = source

    # -----------------------------------------------------------------
    # Exact MPNet scoring from frozen Fresh95 implementation
    # -----------------------------------------------------------------

    print()
    print("===== MPNET =====")
    print(
        "threshold:",
        SEMANTIC_THRESHOLD,
    )

    encoder = q.MPNetEncoder()

    source_order = instances

    source_texts = [
        source_by_instance[iid]
        for iid in source_order
    ]

    candidate_texts = [
        r["text"]
        for r in rows
    ]

    print(
        "Encoding",
        len(source_texts),
        "sources +",
        len(candidate_texts),
        "candidates",
    )

    embeddings = encoder.encode(
        source_texts + candidate_texts,
        batch_size=16,
    )

    source_embeddings = {
        iid: embeddings[i]
        for i, iid in enumerate(source_order)
    }

    candidate_embeddings = (
        embeddings[len(source_order):]
    )

    # -----------------------------------------------------------------
    # Frozen exact number/date checks
    # -----------------------------------------------------------------

    fact_cache = {}

    for iid in instances:
        source = source_by_instance[iid]

        fact_cache[iid] = {
            "numbers":
                q.extract_numbers(source),

            "dates":
                q.extract_dates(source),
        }

    # -----------------------------------------------------------------
    # Corrected frozen source-only entity-surface rule
    # -----------------------------------------------------------------

    print()
    print("===== SOURCE-ONLY ENTITY SURFACE GATE =====")

    nlp = spacy.load(
        "en_core_web_sm"
    )

    source_entity_cache = {
        iid:
            ent.source_entities(
                nlp,
                source_by_instance[iid],
            )
        for iid in instances
    }

    # -----------------------------------------------------------------
    # Candidate scoring
    # -----------------------------------------------------------------

    rescored = []

    for idx, row in enumerate(rows):

        iid = row["instance_id"]

        source = row["source_text"]
        text = row["text"]

        similarity = float(
            np.dot(
                source_embeddings[iid],
                candidate_embeddings[idx],
            )
        )

        semantic_ok = bool(
            similarity >= SEMANTIC_THRESHOLD
        )

        number_ok = bool(
            q.preserves(
                fact_cache[iid]["numbers"],
                q.extract_numbers(text),
            )
        )

        date_ok = bool(
            q.preserves(
                fact_cache[iid]["dates"],
                q.extract_dates(text),
            )
        )

        surface_entity_ok, missing_entities = (
            ent.surface_entity_check(
                source_entity_cache[iid],
                text,
            )
        )

        quality_valid = bool(
            text.strip()
            and semantic_ok
            and number_ok
            and date_ok
            and surface_entity_ok
        )

        doc = doc_by_instance[iid]

        rec = dict(row)

        rec.update({
            "base_id":
                doc["base_id"],

            "source_id":
                doc["source_id"],

            "source_generator":
                doc["source_generator"],

            "semantic_similarity":
                similarity,

            "semantic_ok":
                semantic_ok,

            "number_ok":
                number_ok,

            "date_ok":
                date_ok,

            "source_entity_surfaces":
                source_entity_cache[iid],

            "missing_source_entity_surfaces":
                missing_entities,

            "surface_entity_ok":
                bool(surface_entity_ok),

            "quality_valid":
                quality_valid,

            "edit_fraction":
                q.lexical_edit_fraction(
                    source,
                    text,
                ),

            "source_char_length":
                len(source),

            "output_char_length":
                len(text),

            "length_ratio":
                len(text)
                / max(1, len(source)),
        })

        rescored.append(rec)

    write_jsonl(
        RESCORED,
        rescored,
    )

    # -----------------------------------------------------------------
    # Candidate-level gate diagnostics
    # -----------------------------------------------------------------

    print()
    print("=" * 104)
    print("CORRECTED CANDIDATE VALIDITY")
    print("=" * 104)

    candidate_summary = {}

    for condition in CONDITIONS:

        subset = [
            r for r in rescored
            if r["condition"] == condition
        ]

        valid = [
            r for r in subset
            if r["quality_valid"]
        ]

        info = {
            "valid_candidates":
                len(valid),

            "total_candidates":
                len(subset),

            "candidate_valid_rate":
                len(valid) / len(subset),

            "semantic_pass_rate":
                sum(
                    r["semantic_ok"]
                    for r in subset
                )
                / len(subset),

            "number_pass_rate":
                sum(
                    r["number_ok"]
                    for r in subset
                )
                / len(subset),

            "date_pass_rate":
                sum(
                    r["date_ok"]
                    for r in subset
                )
                / len(subset),

            "source_entity_surface_pass_rate":
                sum(
                    r["surface_entity_ok"]
                    for r in subset
                )
                / len(subset),
        }

        candidate_summary[condition] = info

        print()
        print(condition)
        print(
            "  valid candidates :",
            f"{len(valid)}/{len(subset)}",
            f"({len(valid)/len(subset):.1%})",
        )
        print(
            "  semantic pass    :",
            f"{info['semantic_pass_rate']:.1%}",
        )
        print(
            "  number pass      :",
            f"{info['number_pass_rate']:.1%}",
        )
        print(
            "  date pass        :",
            f"{info['date_pass_rate']:.1%}",
        )
        print(
            "  entity pass      :",
            f"{info['source_entity_surface_pass_rate']:.1%}",
        )

    # -----------------------------------------------------------------
    # First quality-valid attempt in each slot
    # -----------------------------------------------------------------

    grouped = defaultdict(list)

    for row in rescored:

        grouped[
            (
                row["instance_id"],
                row["condition"],
                int(row["slot"]),
            )
        ].append(row)

    valid_slots = defaultdict(list)

    for iid in instances:
        for condition in CONDITIONS:
            for slot in range(K):

                attempts = sorted(
                    grouped[
                        (
                            iid,
                            condition,
                            slot,
                        )
                    ],
                    key=lambda r:
                        int(r["attempt"]),
                )

                if len(attempts) != ATTEMPTS:
                    raise RuntimeError(
                        f"Bad attempts: "
                        f"{iid}/{condition}/{slot}"
                    )

                chosen = next(
                    (
                        r for r in attempts
                        if r["quality_valid"]
                    ),
                    None,
                )

                if chosen is not None:
                    valid_slots[
                        (
                            iid,
                            condition,
                        )
                    ].append(chosen)

    # -----------------------------------------------------------------
    # Frozen document-level selection
    # -----------------------------------------------------------------

    selected = []

    for iid in instances:

        metadata = doc_by_instance[iid]
        source = source_by_instance[iid]

        for condition in CONDITIONS:

            valid = sorted(
                valid_slots[
                    (
                        iid,
                        condition,
                    )
                ],
                key=lambda r:
                    int(r["slot"]),
            )

            if not valid:

                selected.append({
                    "instance_id":
                        iid,

                    "base_id":
                        metadata["base_id"],

                    "manifest_index":
                        metadata["manifest_index"],

                    "generator_key":
                        metadata["generator_key"],

                    "source_generator":
                        metadata["source_generator"],

                    "sample_id":
                        metadata["sample_id"],

                    "source_id":
                        metadata["source_id"],

                    "domain":
                        metadata["domain"],

                    "condition":
                        condition,

                    "status":
                        "ABSTAIN",

                    "reason":
                        "no_quality_valid_candidate",

                    "selection_rule":
                        "return_source_unchanged",

                    "quality_valid_slots":
                        0,

                    "output_text":
                        source,

                    "selected_slot":
                        None,

                    "selected_attempt":
                        None,

                    "selected_seed":
                        None,

                    "semantic_similarity":
                        1.0,

                    "edit_fraction":
                        0.0,

                    "length_ratio":
                        1.0,

                    "trajectory_typicality":
                        None,

                    "mean_control_tv":
                        0.0,

                    "mean_support_size":
                        None,

                    "controllable_fraction":
                        None,
                })

                continue

            if condition == "vanilla_t10":

                chosen = min(
                    valid,
                    key=lambda r:
                        int(r["slot"]),
                )

                selection_rule = (
                    "first_quality_valid_slot"
                )

            elif condition == "hpcd_n0":

                chosen = min(
                    valid,
                    key=lambda r: (
                        float(
                            r[
                                "trajectory_typicality"
                            ]
                        ),
                        int(r["slot"]),
                    ),
                )

                selection_rule = (
                    "minimum_generation_time_"
                    "n0_trajectory_typicality"
                )

            else:
                raise RuntimeError(
                    f"Unexpected condition: {condition}"
                )

            selected.append({
                "instance_id":
                    iid,

                "base_id":
                    metadata["base_id"],

                "manifest_index":
                    chosen["manifest_index"],

                "generator_key":
                    chosen["generator_key"],

                "source_generator":
                    metadata["source_generator"],

                "sample_id":
                    chosen["sample_id"],

                "source_id":
                    metadata["source_id"],

                "domain":
                    chosen["domain"],

                "condition":
                    condition,

                "status":
                    "REWRITE",

                "reason":
                    "quality_valid_candidate_available",

                "selection_rule":
                    selection_rule,

                "quality_valid_slots":
                    len(valid),

                "output_text":
                    chosen["text"],

                "selected_slot":
                    int(chosen["slot"]),

                "selected_attempt":
                    int(chosen["attempt"]),

                "selected_seed":
                    int(chosen["seed"]),

                "semantic_similarity":
                    float(
                        chosen[
                            "semantic_similarity"
                        ]
                    ),

                "edit_fraction":
                    float(
                        chosen["edit_fraction"]
                    ),

                "length_ratio":
                    float(
                        chosen["length_ratio"]
                    ),

                "trajectory_typicality":
                    float(
                        chosen[
                            "trajectory_typicality"
                        ]
                    ),

                "mean_control_tv":
                    float(
                        chosen[
                            "mean_control_tv"
                        ]
                    ),

                "mean_support_size":
                    float(
                        chosen[
                            "mean_support_size"
                        ]
                    ),

                "controllable_fraction":
                    float(
                        chosen[
                            "controllable_fraction"
                        ]
                    ),
            })

    if len(selected) != 190:
        raise RuntimeError(
            f"Expected 190 selected outputs, "
            f"got {len(selected)}"
        )

    selection_keys = {
        (
            r["instance_id"],
            r["condition"],
        )
        for r in selected
    }

    expected_selection_keys = {
        (
            iid,
            condition,
        )
        for iid in instances
        for condition in CONDITIONS
    }

    if selection_keys != expected_selection_keys:
        raise RuntimeError(
            "Selected-output grid incomplete"
        )

    write_jsonl(
        SELECTED,
        selected,
    )

    # -----------------------------------------------------------------
    # Selection summary
    # -----------------------------------------------------------------

    print()
    print("=" * 104)
    print("DOCUMENT-LEVEL ITT SELECTION")
    print("=" * 104)

    condition_selection = {}

    for condition in CONDITIONS:

        subset = [
            r for r in selected
            if r["condition"] == condition
        ]

        rewritten = [
            r for r in subset
            if r["status"] == "REWRITE"
        ]

        abstained = [
            r for r in subset
            if r["status"] == "ABSTAIN"
        ]

        by_domain = Counter(
            r["domain"]
            for r in rewritten
        )

        by_generator = Counter(
            r["generator_key"]
            for r in rewritten
        )

        ratios = [
            float(r["length_ratio"])
            for r in rewritten
        ]

        info = {
            "rewrites":
                len(rewritten),

            "abstentions":
                len(abstained),

            "coverage":
                len(rewritten) / 95,

            "domain_rewrites":
                dict(by_domain),

            "generator_rewrites":
                dict(by_generator),

            "mean_selected_semantic_similarity":
                (
                    float(
                        np.mean([
                            r["semantic_similarity"]
                            for r in rewritten
                        ])
                    )
                    if rewritten
                    else None
                ),

            "mean_edit_fraction":
                (
                    float(
                        np.mean([
                            r["edit_fraction"]
                            for r in rewritten
                        ])
                    )
                    if rewritten
                    else None
                ),

            "mean_length_ratio":
                (
                    float(np.mean(ratios))
                    if ratios
                    else None
                ),

            "median_length_ratio":
                (
                    float(np.median(ratios))
                    if ratios
                    else None
                ),

            "selected_below_0_80":
                sum(
                    x < 0.80
                    for x in ratios
                ),
        }

        condition_selection[condition] = info

        print()
        print(condition)
        print(
            "  rewrites   :",
            f"{len(rewritten)}/95",
            f"({len(rewritten)/95:.1%})",
        )
        print(
            "  abstentions:",
            f"{len(abstained)}/95",
        )
        print(
            "  domains    :",
            dict(by_domain),
        )
        print(
            "  generators :",
            dict(by_generator),
        )

    matched_rewrite = [
        iid
        for iid in instances
        if all(
            next(
                r
                for r in selected
                if (
                    r["instance_id"] == iid
                    and r["condition"] == condition
                )
            )["status"]
            == "REWRITE"
            for condition in CONDITIONS
        )
    ]

    summary = {
        "status":
            "FROZEN_COMPLETE",

        "experiment":
            "CONFIRM48_V1",

        "stage":
            "Frozen quality-first ITT selection",

        "input_candidates": {
            "path":
                str(
                    CANDIDATES.relative_to(ROOT)
                ),

            "sha256":
                sha256(CANDIDATES),

            "rows":
                1520,
        },

        "generation_freeze": {
            "path":
                str(
                    GEN_FREEZE.relative_to(ROOT)
                ),

            "sha256":
                sha256(GEN_FREEZE),
        },

        "helper_provenance": {
            "fresh95_quality": {
                "path":
                    str(
                        FRESH_QUALITY.relative_to(ROOT)
                    ),

                "sha256":
                    sha256(FRESH_QUALITY),
            },

            "fresh95_corrected_entity": {
                "path":
                    str(
                        FRESH_ENTITY.relative_to(ROOT)
                    ),

                "sha256":
                    sha256(FRESH_ENTITY),
            },
        },

        "quality_gate": {
            "mpnet_similarity_ge":
                SEMANTIC_THRESHOLD,

            "numbers_preserved":
                True,

            "dates_preserved":
                True,

            "entity_rule":
                (
                    "source-only spaCy entities "
                    "must remain present as "
                    "normalized surface strings"
                ),

            "spacy_model":
                "en_core_web_sm",

            "nonempty":
                True,

            "length_ratio_threshold":
                None,

            "length_ratio_use":
                "diagnostic only",
        },

        "selection_policy": {
            "within_slot":
                "first quality-valid attempt",

            "vanilla_t10":
                "first quality-valid slot",

            "hpcd_n0":
                (
                    "minimum generation-time n0 "
                    "trajectory_typicality among "
                    "quality-valid slots"
                ),

            "tie_break":
                "lowest slot index",

            "abstention":
                "return source unchanged",

            "detector_feedback":
                False,
        },

        "candidate_validity":
            candidate_summary,

        "selection":
            condition_selection,

        "matched_both_rewrites":
            len(matched_rewrite),

        "matched_both_rewrite_ids":
            matched_rewrite,

        "instances":
            95,

        "base_clusters":
            48,

        "rescored_candidates_sha256":
            sha256(RESCORED),

        "selected_outputs_sha256":
            sha256(SELECTED),

        "environment": {
            "python":
                platform.python_version(),

            "spacy":
                spacy.__version__,
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

    print()
    print("=" * 104)
    print("ARTIFACT HASHES")
    print("=" * 104)

    print(
        "rescored candidates:",
        sha256(RESCORED),
    )

    print(
        "selected outputs    :",
        sha256(SELECTED),
    )

    print(
        "summary             :",
        sha256(SUMMARY),
    )

    print()
    print(
        "CONFIRM48 QUALITY-FIRST SELECTION: PASS"
    )
    print(
        "NEXT: FREEZE SELECTED 95 VANILLA + "
        "95 HPCD ITT OUTPUTS"
    )


if __name__ == "__main__":
    main()
