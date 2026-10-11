import os
#!/usr/bin/env python3

from pathlib import Path
from collections import Counter
import hashlib
import importlib.metadata as md
import json
import math
import sys
import time

import torch
from transformers import (
    LongformerConfig,
    LongformerForSequenceClassification,
    LongformerTokenizer,
)


ROOT = Path(os.environ.get("HPCD_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))

REPO = ROOT / "external/extension_oct2026_v1/MAGE-main"
MODEL_DIR = ROOT / "models/MAGE"

INPUT = (
    ROOT
    / "results/detectors/fresh95_final_v1/inputs/"
      "fresh95_detector_input_v1.jsonl"
)

OUT_DIR = ROOT / "results/extension_oct2026_v1/mage"

OUT = OUT_DIR / "mage_fresh95_frozen523_v1.jsonl"
META = OUT_DIR / "mage_fresh95_frozen523_v1_meta.json"
FREEZE = OUT_DIR / "mage_protocol_freeze_v1.json"

EXPECTED_INPUT_SHA = (
    "5e5bc0f60923b43675be252f6c3651e7"
    "0338c46ac4ee1a6ba3d211e8f0cdfcb5"
)

EXPECTED_COUNTS = {
    "human": 48,
    "original": 95,
    "vanilla_t10": 95,
    "vanilla_t11": 95,
    "copa3b": 95,
    "hpcd_n0": 95,
}

DEVICE = "cpu"

# Exact released deployment threshold.
OFFICIAL_THRESHOLD = -3.08583984375

# Longformer config: max_position_embeddings = 4098.
# Safe usable input length including special tokens.
MAX_SAFE_INPUT_TOKENS = 4096


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def text_sha256(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


# ============================================================
# VERIFY ENVIRONMENT
# ============================================================

versions = {
    "clean-text": md.version("clean-text"),
    "ftfy": md.version("ftfy"),
    "Unidecode": md.version("Unidecode"),
    "torch": torch.__version__,
}

try:
    versions["transformers"] = md.version("transformers")
except Exception:
    versions["transformers"] = "unknown"

if versions["clean-text"] != "0.6.0":
    raise RuntimeError(
        f"Expected clean-text==0.6.0, found {versions['clean-text']}"
    )

if versions["ftfy"] != "6.1.3":
    raise RuntimeError(
        f"Expected ftfy==6.1.3, found {versions['ftfy']}"
    )


# ============================================================
# OFFICIAL PREPROCESSING
# ============================================================

sys.path.insert(0, str(REPO))
from deployment.utils import preprocess  # noqa: E402


# ============================================================
# VERIFY PATHS / SOURCE
# ============================================================

for p in [REPO, MODEL_DIR, INPUT]:
    if not p.exists():
        raise FileNotFoundError(p)

input_sha = sha256(INPUT)

if input_sha != EXPECTED_INPUT_SHA:
    raise RuntimeError(
        "\nFROZEN INPUT SHA MISMATCH\n"
        f"Expected: {EXPECTED_INPUT_SHA}\n"
        f"Actual:   {input_sha}"
    )

with open(INPUT, "r", encoding="utf-8") as f:
    rows = [
        json.loads(line)
        for line in f
        if line.strip()
    ]

if len(rows) != 523:
    raise RuntimeError(
        f"Expected 523 rows, found {len(rows)}"
    )

required_fields = {
    "base_id",
    "sample_id",
    "domain",
    "source_generator",
    "condition",
    "text",
    "text_sha256",
    "is_human",
    "source_sample_id",
}

missing = required_fields - set(rows[0])

if missing:
    raise RuntimeError(
        f"Missing required fields: {sorted(missing)}"
    )

counts = Counter(
    str(r["condition"]).strip()
    for r in rows
)

if dict(counts) != EXPECTED_COUNTS:
    raise RuntimeError(
        "\nCONDITION COUNT MISMATCH\n"
        f"Expected: {EXPECTED_COUNTS}\n"
        f"Actual:   {dict(counts)}"
    )


# ============================================================
# VERIFY RAW TEXT HASHES
# ============================================================

for i, row in enumerate(rows):
    actual = text_sha256(row["text"])
    expected = row["text_sha256"]

    if actual != expected:
        raise RuntimeError(
            f"Text SHA mismatch at row {i}: "
            f"{row['sample_id']}"
        )


print("=" * 100)
print("MAGE FULL FROZEN DETECTOR RUN — 523 ROWS")
print("=" * 100)
print(f"Input            : {INPUT.relative_to(ROOT)}")
print(f"Input SHA256     : {input_sha}")
print(f"Rows             : {len(rows)}")
print(f"Conditions       : {dict(counts)}")
print(f"Device           : {DEVICE}")
print(f"Official th      : {OFFICIAL_THRESHOLD}")
print(f"clean-text       : {versions['clean-text']}")
print(f"ftfy             : {versions['ftfy']}")
print(f"Unidecode        : {versions['Unidecode']}")
print(f"transformers     : {versions['transformers']}")
print(f"torch            : {versions['torch']}")


# ============================================================
# LOAD MODEL — COMPATIBILITY NORMALIZATION IN MEMORY ONLY
# ============================================================

with open(
    MODEL_DIR / "config.json",
    "r",
    encoding="utf-8",
) as f:
    config_dict = json.load(f)

released_id2label = config_dict.get("id2label")
released_label2id = config_dict.get("label2id")

config_dict["id2label"] = {
    int(k): str(v)
    for k, v in config_dict["id2label"].items()
}

config_dict["label2id"] = {
    str(k): int(v)
    for k, v in config_dict["label2id"].items()
}

config = LongformerConfig.from_dict(
    config_dict
)

tokenizer = LongformerTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
)

model = LongformerForSequenceClassification.from_pretrained(
    MODEL_DIR,
    config=config,
    local_files_only=True,
)

model.to(DEVICE)
model.eval()

if model.config.num_labels != 2:
    raise RuntimeError(
        f"Expected 2 labels, got {model.config.num_labels}"
    )

print()
print("Model class       :", model.__class__.__name__)
print("Tokenizer class   :", tokenizer.__class__.__name__)
print(
    "Max positions     :",
    model.config.max_position_embeddings,
)


# ============================================================
# PREPROCESS + TOKEN-LENGTH VALIDATION FIRST
# ============================================================

print()
print("=" * 100)
print("PREPROCESSING / LENGTH VALIDATION")
print("=" * 100)

prepared = []
token_lengths = []

for i, row in enumerate(rows, start=1):

    raw_text = row["text"]
    processed = preprocess(raw_text)

    encoded = tokenizer(
        processed,
        truncation=False,
    )

    input_ids = encoded["input_ids"]
    n_tokens = len(input_ids)

    if n_tokens > MAX_SAFE_INPUT_TOKENS:
        raise RuntimeError(
            "\nTEXT EXCEEDS SAFE MAGE INPUT LENGTH\n"
            f"Row       : {i}\n"
            f"sample_id : {row['sample_id']}\n"
            f"condition : {row['condition']}\n"
            f"tokens    : {n_tokens}\n"
            f"safe max  : {MAX_SAFE_INPUT_TOKENS}\n"
            "\nSTOPPING BEFORE FULL INFERENCE."
        )

    prepared.append(
        {
            "row": row,
            "processed_text": processed,
            "input_ids": input_ids,
            "token_count": n_tokens,
        }
    )

    token_lengths.append(
        n_tokens
    )

print(
    f"Token length min/max: "
    f"{min(token_lengths)} / {max(token_lengths)}"
)

print(
    f"Mean token length   : "
    f"{sum(token_lengths) / len(token_lengths):.2f}"
)

print("LENGTH_VALIDATION=PASS")


# ============================================================
# FULL OFFICIAL-STYLE INFERENCE
# ============================================================

print()
print("=" * 100)
print("SCORING 523 ROWS")
print("=" * 100)

results = []

start = time.perf_counter()

with torch.no_grad():

    for i, item in enumerate(
        prepared,
        start=1,
    ):

        row = item["row"]

        tensor_input = torch.tensor(
            [item["input_ids"]],
            dtype=torch.long,
            device=DEVICE,
        )

        outputs = model(
            tensor_input
        )

        logits = outputs.logits[0].detach().cpu()

        logit0 = float(
            logits[0].item()
        )

        logit1 = float(
            logits[1].item()
        )

        probs = torch.softmax(
            logits,
            dim=-1,
        )

        prob0 = float(
            probs[0].item()
        )

        prob1 = float(
            probs[1].item()
        )

        # Exact released deployment quantity.
        official_score = -logit0

        if official_score < OFFICIAL_THRESHOLD:
            official_label = 0
            official_decision = "machine-generated"
        else:
            official_label = 1
            official_decision = "human-written"

        # Frozen ROC convention:
        # higher value = more machine-like.
        machine_score = logit0

        result = {
            "row_index": i - 1,
            "base_id": row["base_id"],
            "sample_id": row["sample_id"],
            "domain": row["domain"],
            "source_generator": row["source_generator"],
            "condition": row["condition"],
            "is_human": row["is_human"],
            "source_sample_id": row["source_sample_id"],
            "text_sha256": row["text_sha256"],
            "preprocessed_text_sha256": text_sha256(
                item["processed_text"]
            ),
            "raw_char_count": len(
                row["text"]
            ),
            "preprocessed_char_count": len(
                item["processed_text"]
            ),
            "token_count": item["token_count"],
            "logit0": logit0,
            "logit1": logit1,
            "prob_label0": prob0,
            "prob_label1": prob1,
            "official_score_neg_logit0": official_score,
            "machine_score_logit0": machine_score,
            "official_threshold": OFFICIAL_THRESHOLD,
            "official_label": official_label,
            "official_decision": official_decision,
        }

        results.append(
            result
        )

        if (
            i == 1
            or i % 25 == 0
            or i == len(prepared)
        ):
            elapsed = time.perf_counter() - start
            rate = i / elapsed
            remain = (
                (len(prepared) - i) / rate
                if rate > 0 else math.nan
            )

            print(
                f"[{i:03d}/{len(prepared)}] "
                f"{row['condition']:<12} "
                f"tok={item['token_count']:<4d} "
                f"score={machine_score:+.5f} "
                f"decision={official_decision:<17} "
                f"elapsed={elapsed/60:.1f}m "
                f"ETA={remain/60:.1f}m",
                flush=True,
            )


# ============================================================
# WRITE FINAL RESULTS
# ============================================================

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

with open(
    OUT,
    "w",
    encoding="utf-8",
) as f:
    for row in results:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )


# ============================================================
# FREEZE PROTOCOL
# ============================================================

protocol = {
    "name": "MAGE frozen extension protocol v1",
    "model": "local released MAGE Longformer checkpoint",
    "model_dir": str(
        MODEL_DIR.relative_to(ROOT)
    ),
    "official_repo": str(
        REPO.relative_to(ROOT)
    ),
    "input": str(
        INPUT.relative_to(ROOT)
    ),
    "input_sha256": input_sha,
    "n_rows": 523,
    "preprocessing": (
        "official deployment.utils.preprocess"
    ),
    "clean_text_version": versions["clean-text"],
    "ftfy_version": versions["ftfy"],
    "unidecode_version": versions["Unidecode"],
    "device": DEVICE,
    "tokenization": (
        "tokenizer(preprocessed_text, truncation=False)"
    ),
    "max_safe_input_tokens": MAX_SAFE_INPUT_TOKENS,
    "official_label_semantics": {
        "0": "machine-generated",
        "1": "human-written",
    },
    "released_id2label": released_id2label,
    "released_label2id": released_label2id,
    "compatibility_normalization": (
        "id2label/label2id normalized in memory only "
        "for current Transformers; model files unchanged"
    ),
    "official_detection_score": "-logit0",
    "official_threshold": OFFICIAL_THRESHOLD,
    "official_machine_rule": (
        "-logit0 < -3.08583984375"
    ),
    "evaluation_machine_score": "logit0",
    "evaluation_score_orientation": (
        "higher = more machine-like"
    ),
    "truncation": False,
    "selection_or_tuning_on_results": False,
}

with open(
    FREEZE,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        protocol,
        f,
        indent=2,
        ensure_ascii=False,
    )
    f.write("\n")


# ============================================================
# METADATA / SANITY SUMMARY
# ============================================================

decision_counts = {}

for condition in EXPECTED_COUNTS:

    subset = [
        r for r in results
        if r["condition"] == condition
    ]

    machine_count = sum(
        r["official_label"] == 0
        for r in subset
    )

    mean_score = sum(
        r["machine_score_logit0"]
        for r in subset
    ) / len(subset)

    decision_counts[condition] = {
        "n": len(subset),
        "official_machine_calls": machine_count,
        "official_machine_rate": (
            machine_count / len(subset)
        ),
        "mean_machine_score_logit0": mean_score,
    }


meta = {
    "input_sha256": input_sha,
    "output_sha256": sha256(OUT),
    "protocol_sha256": sha256(FREEZE),
    "n_rows": len(results),
    "token_count_min": min(token_lengths),
    "token_count_max": max(token_lengths),
    "token_count_mean": (
        sum(token_lengths) / len(token_lengths)
    ),
    "condition_summary": decision_counts,
    "versions": versions,
}

with open(
    META,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        meta,
        f,
        indent=2,
        ensure_ascii=False,
    )
    f.write("\n")


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 100)
print("MAGE FULL RUN COMPLETE")
print("=" * 100)

for condition in EXPECTED_COUNTS:
    s = decision_counts[condition]

    print(
        f"{condition:<12} "
        f"n={s['n']:<3d} "
        f"machine={s['official_machine_calls']:<3d} "
        f"rate={s['official_machine_rate']:.4f} "
        f"mean_score={s['mean_machine_score_logit0']:+.6f}"
    )

print()
print("RESULT:")
print(OUT.relative_to(ROOT))
print("SHA256:", sha256(OUT))

print()
print("PROTOCOL FREEZE:")
print(FREEZE.relative_to(ROOT))
print("SHA256:", sha256(FREEZE))

print()
print("METADATA:")
print(META.relative_to(ROOT))
print("SHA256:", sha256(META))

print()
print("MAGE_FROZEN523_COMPLETE")

