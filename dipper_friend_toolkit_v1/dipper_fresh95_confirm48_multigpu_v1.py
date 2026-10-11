#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portable multi-GPU DIPPER runner for the frozen HPCD/Fresh95 project.

Scientific condition (frozen before evaluation):
  DIPPER full contextual model
  paper diversity L60/O60 -> internal similarity control codes 40/40
  sent_interval = 3
  do_sample = True
  top_p = 0.75
  top_k = None
  max_length = 512
  initial prefix = ""
  global seed = 15485863
  per-instance seed = uint32(first 4 bytes SHA256(global_seed|machine_sample_id))

The generation semantics follow the official DIPPER paraphrasing code.  The
multi-GPU device sharding, JSONL I/O, hashing, resume support, and provenance
logging are execution/provenance additions only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# IMPORTANT: set CUDA_VISIBLE_DEVICES in the shell BEFORE this program starts.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import nltk
import torch
import transformers
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer


# =============================================================================
# FROZEN SCIENTIFIC SETTINGS -- DO NOT TUNE ON RESULTS
# =============================================================================

LEX_DIVERSITY = 60
ORDER_DIVERSITY = 60
LEX_CODE = 100 - LEX_DIVERSITY
ORDER_CODE = 100 - ORDER_DIVERSITY

SENT_INTERVAL = 3

DO_SAMPLE = True
TOP_P = 0.75
TOP_K = None
MAX_LENGTH = 512

GLOBAL_SEED = 15485863
INITIAL_PREFIX = ""

EXPECTED_ROWS = 95

OFFICIAL_REPO = "martiansideofthemoon/ai-detection-paraphrases"
OFFICIAL_MODEL = "kalpeshk2011/dipper-paraphraser-xxl"
OFFICIAL_TOKENIZER = "google/t5-v1_1-xxl"


# =============================================================================
# HELPERS
# =============================================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_seed(global_seed: int, machine_sample_id: str) -> int:
    raw = f"{global_seed}|{machine_sample_id}".encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise RuntimeError(
                    f"Invalid JSONL record {n} in {path}: {exc}"
                ) from exc
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_json_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def git_commit(repo: str | None) -> str | None:
    if not repo:
        return None
    p = Path(repo)
    if not (p / ".git").exists():
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(p), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def nltk_version() -> str:
    return getattr(nltk, "__version__", "unknown")


def ensure_sentence_tokenizer(allow_download: bool) -> None:
    # NLTK <=3.8 generally needs punkt; newer versions can also request punkt_tab.
    probe = "One sentence. Another sentence."
    try:
        sent_tokenize(probe)
        return
    except LookupError as first:
        if not allow_download:
            raise RuntimeError(
                "NLTK sentence tokenizer data is missing. "
                "Install/download the required punkt resource in the DIPPER "
                "environment, or rerun with --allow_nltk_download."
            ) from first

    nltk.download("punkt", quiet=False)
    try:
        nltk.download("punkt_tab", quiet=False)
    except Exception:
        pass

    try:
        sent_tokenize(probe)
    except LookupError as exc:
        raise RuntimeError(
            "NLTK sentence tokenizer is still unavailable after download."
        ) from exc


def validate_input(rows: list[dict], expected_rows: int) -> None:
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} input rows, found {len(rows)}."
        )

    required = {
        "base_id",
        "domain",
        "generator_key",
        "machine_sample_id",
        "sample_id",
        "source_generator",
        "source_id",
        "source_text",
        "title",
    }

    for i, row in enumerate(rows):
        missing = sorted(required - set(row))
        if missing:
            raise RuntimeError(f"Input row {i} missing fields: {missing}")
        if not isinstance(row["source_text"], str) or not row["source_text"].strip():
            raise RuntimeError(f"Input row {i} has empty source_text")

    mids = [row["machine_sample_id"] for row in rows]
    if len(set(mids)) != len(mids):
        raise RuntimeError("machine_sample_id is not unique.")


# =============================================================================
# DIPPER
# =============================================================================

class DipperParaphraser:
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        gpu_memory_gib: float,
        cpu_memory_gib: float,
        verbose: bool = True,
    ):
        if "no-context" in model_path.lower() or "no_context" in model_path.lower():
            raise RuntimeError(
                "The no-context DIPPER checkpoint is not allowed for this run. "
                "Use the full contextual DIPPER model."
            )

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this DIPPER run.")

        n_gpu = torch.cuda.device_count()
        if n_gpu < 2:
            raise RuntimeError(
                f"Only {n_gpu} visible CUDA GPU(s). "
                "This package is configured for multi-card RTX 3090 execution."
            )

        t0 = time.time()

        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)

        max_memory = {
            i: f"{gpu_memory_gib:g}GiB"
            for i in range(n_gpu)
        }
        max_memory["cpu"] = f"{cpu_memory_gib:g}GiB"

        # "balanced" spreads the 11B fp16 model across visible cards.
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_path,
            device_map="balanced",
            max_memory=max_memory,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

        # Input IDs must start on the device that owns T5's shared embedding.
        self.input_device = self.model.shared.weight.device
        self.model_load_seconds = time.time() - t0
        self.max_memory = max_memory

        if verbose:
            print(f"Model loaded in {self.model_load_seconds:.2f}s")
            print("Visible GPUs:", n_gpu)
            print("Input device:", self.input_device)
            print("Device map:", getattr(self.model, "hf_device_map", None))
            print("Max memory:", max_memory)

    def paraphrase(
        self,
        input_text: str,
        prefix: str = INITIAL_PREFIX,
    ) -> tuple[str, dict]:

        input_text_normalized = " ".join(input_text.split())
        sentences = sent_tokenize(input_text_normalized)
        prefix = " ".join(prefix.replace("\n", " ").split())
        output_text = ""

        window_meta = []

        for sent_idx in range(0, len(sentences), SENT_INTERVAL):
            curr_window = " ".join(
                sentences[sent_idx : sent_idx + SENT_INTERVAL]
            )

            final_input_text = (
                f"lexical = {LEX_CODE}, order = {ORDER_CODE}"
            )
            if prefix:
                final_input_text += f" {prefix}"
            final_input_text += f" <sent> {curr_window} </sent>"

            batch = self.tokenizer(
                [final_input_text],
                return_tensors="pt",
            )
            input_tokens = int(batch["input_ids"].shape[1])
            batch = {
                k: v.to(self.input_device)
                for k, v in batch.items()
            }

            with torch.inference_mode():
                outputs = self.model.generate(
                    **batch,
                    do_sample=DO_SAMPLE,
                    top_p=TOP_P,
                    top_k=TOP_K,
                    max_length=MAX_LENGTH,
                )

            text = self.tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
            )[0].strip()

            prefix = (prefix + " " + text).strip()
            output_text = (output_text + " " + text).strip()

            window_meta.append(
                {
                    "window_index": len(window_meta),
                    "sentence_start": sent_idx,
                    "sentence_count": len(
                        sentences[sent_idx : sent_idx + SENT_INTERVAL]
                    ),
                    "model_input_tokens": input_tokens,
                    "generated_tokens": int(outputs.shape[1]),
                }
            )

        return output_text.strip(), {
            "sentence_count": len(sentences),
            "window_count": len(window_meta),
            "windows": window_meta,
        }


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--input", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--run_name", required=True)

    p.add_argument("--model_path", required=True)
    p.add_argument("--tokenizer_path", required=True)

    p.add_argument(
        "--dipper_repo",
        default=None,
        help="Optional local official DIPPER repository path for git provenance.",
    )

    p.add_argument(
        "--mode",
        choices=["smoke", "full"],
        default="full",
    )

    p.add_argument(
        "--expected_rows",
        type=int,
        default=EXPECTED_ROWS,
    )

    # Execution-only memory controls, not scientific generation parameters.
    p.add_argument(
        "--gpu_memory_gib",
        type=float,
        default=22.0,
        help="Per-visible-GPU allocation ceiling; 22 GiB is suitable for 24GB 3090s.",
    )
    p.add_argument(
        "--cpu_memory_gib",
        type=float,
        default=64.0,
    )

    p.add_argument(
        "--allow_nltk_download",
        action="store_true",
    )

    return p.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input).resolve()
    output_root = Path(args.output_dir).resolve()
    mode_dir = output_root / args.run_name / args.mode
    mode_dir.mkdir(parents=True, exist_ok=True)

    final_path = mode_dir / f"dipper_{args.run_name}_{args.mode}_final.jsonl"
    failures_path = mode_dir / f"dipper_{args.run_name}_{args.mode}_failures.jsonl"
    summary_path = mode_dir / f"dipper_{args.run_name}_{args.mode}_summary.json"
    manifest_path = mode_dir / f"dipper_{args.run_name}_{args.mode}_manifest.json"

    ensure_sentence_tokenizer(args.allow_nltk_download)

    all_rows = load_jsonl(input_path)
    validate_input(all_rows, args.expected_rows)

    # Deterministic smoke selection:
    # first Qwen row, then first DeepSeek row from a different domain if possible.
    if args.mode == "smoke":
        qwen = next(
            r for r in all_rows
            if r["generator_key"] == "qwen25_3b"
        )
        deep_candidates = [
            r for r in all_rows
            if r["generator_key"] == "deepseek_r1_qwen7b_prefill"
        ]
        deep = next(
            (r for r in deep_candidates if r["domain"] != qwen["domain"]),
            deep_candidates[0],
        )
        rows = [qwen, deep]
    else:
        rows = all_rows

    expected_this_mode = 2 if args.mode == "smoke" else args.expected_rows

    completed_rows = load_jsonl(final_path) if final_path.exists() else []
    completed = {
        r["machine_sample_id"]
        for r in completed_rows
    }

    if len(completed) != len(completed_rows):
        raise RuntimeError("Duplicate machine_sample_id in existing output.")

    if len(completed_rows) > expected_this_mode:
        raise RuntimeError("Existing output has more rows than expected.")

    print("=" * 96)
    print("DIPPER FROZEN MULTI-GPU RUN")
    print("=" * 96)
    print("run_name       :", args.run_name)
    print("mode           :", args.mode)
    print("input          :", input_path)
    print("input SHA256   :", sha256_file(input_path))
    print("input rows     :", len(all_rows))
    print("run rows       :", len(rows))
    print("already done   :", len(completed))
    print("L/O diversity  :", LEX_DIVERSITY, ORDER_DIVERSITY)
    print("control codes  :", LEX_CODE, ORDER_CODE)
    print("sent_interval  :", SENT_INTERVAL)
    print("do_sample      :", DO_SAMPLE)
    print("top_p          :", TOP_P)
    print("top_k          :", TOP_K)
    print("max_length     :", MAX_LENGTH)
    print("global seed    :", GLOBAL_SEED)
    print("initial prefix :", repr(INITIAL_PREFIX))
    print("CUDA visible   :", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("GPU count      :", torch.cuda.device_count())
    print("=" * 96)

    run_start = time.time()
    run_started_at = utc_now()

    dipper = DipperParaphraser(
        args.model_path,
        args.tokenizer_path,
        args.gpu_memory_gib,
        args.cpu_memory_gib,
    )

    runtimes = []

    for rec in tqdm(rows, desc=f"DIPPER {args.run_name} {args.mode}"):
        mid = rec["machine_sample_id"]

        if mid in completed:
            continue

        seed = stable_seed(GLOBAL_SEED, mid)
        set_all_seeds(seed)

        start = time.time()
        status = "success"
        output_text = ""
        error_type = None
        error_message = None
        meta = {}

        try:
            output_text, meta = dipper.paraphrase(
                rec["source_text"],
                prefix=INITIAL_PREFIX,
            )
            if not output_text.strip():
                raise RuntimeError("DIPPER produced empty output.")
        except Exception as exc:
            status = "failure"
            error_type = type(exc).__name__
            error_message = repr(exc)

        elapsed = time.time() - start
        runtimes.append(elapsed)

        source_token_count = len(
            dipper.tokenizer(
                rec["source_text"],
                add_special_tokens=True,
            )["input_ids"]
        )

        output_token_count = (
            len(
                dipper.tokenizer(
                    output_text,
                    add_special_tokens=True,
                )["input_ids"]
            )
            if output_text
            else 0
        )

        out = {
            # Original identifiers
            "sample_id": rec["sample_id"],
            "base_id": rec["base_id"],
            "machine_sample_id": mid,
            "source_id": rec["source_id"],
            "domain": rec["domain"],
            "source_generator": rec["source_generator"],
            "generator_key": rec["generator_key"],
            "title": rec["title"],

            # Text
            "source_text": rec["source_text"],
            "source_text_sha256": sha256_text(rec["source_text"]),
            "output_text": output_text,
            "output_text_sha256": sha256_text(output_text) if output_text else None,

            # Outcome
            "status": status,
            "error_type": error_type,
            "error_message": error_message,

            # Frozen DIPPER protocol
            "dipper_model_path": args.model_path,
            "dipper_tokenizer_path": args.tokenizer_path,
            "dipper_official_model": OFFICIAL_MODEL,
            "dipper_official_repo": OFFICIAL_REPO,
            "lex_diversity": LEX_DIVERSITY,
            "order_diversity": ORDER_DIVERSITY,
            "lex_internal_code": LEX_CODE,
            "order_internal_code": ORDER_CODE,
            "sent_interval": SENT_INTERVAL,
            "generation_kwargs": {
                "do_sample": DO_SAMPLE,
                "top_p": TOP_P,
                "top_k": TOP_K,
                "max_length": MAX_LENGTH,
            },
            "initial_prefix": INITIAL_PREFIX,
            "global_seed": GLOBAL_SEED,
            "instance_seed": seed,

            # Diagnostics
            "source_word_count": len(rec["source_text"].split()),
            "output_word_count": len(output_text.split()) if output_text else 0,
            "source_token_count": source_token_count,
            "output_token_count": output_token_count,
            "sentence_count": meta.get("sentence_count"),
            "window_count": meta.get("window_count"),
            "windows": meta.get("windows", []),
            "runtime_seconds": elapsed,
        }

        append_jsonl(final_path, out)

        if status == "failure":
            append_jsonl(failures_path, out)

        completed.add(mid)

        print(
            f"[{len(completed):03d}/{expected_this_mode}] "
            f"{rec['domain']:7s} "
            f"{rec['generator_key']:28s} "
            f"{status:7s} "
            f"{elapsed:.2f}s",
            flush=True,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_rows = load_jsonl(final_path)

    if len(final_rows) != expected_this_mode:
        raise RuntimeError(
            f"Expected {expected_this_mode} final rows, got {len(final_rows)}."
        )

    final_ids = [r["machine_sample_id"] for r in final_rows]
    if len(set(final_ids)) != len(final_ids):
        raise RuntimeError("Duplicate final machine_sample_id.")

    if set(final_ids) != {r["machine_sample_id"] for r in rows}:
        raise RuntimeError("Final output membership does not match requested input.")

    statuses = Counter(r["status"] for r in final_rows)
    domains = Counter(r["domain"] for r in final_rows)
    generators = Counter(r["generator_key"] for r in final_rows)

    all_runtime = [
        float(r["runtime_seconds"])
        for r in final_rows
    ]

    summary = {
        "status": "COMPLETE",
        "run_name": args.run_name,
        "mode": args.mode,
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "input_rows": len(all_rows),
        "processed_rows": len(final_rows),
        "status_counts": dict(statuses),
        "domain_counts": dict(domains),
        "generator_counts": dict(generators),
        "frozen_protocol": {
            "lex_diversity": LEX_DIVERSITY,
            "order_diversity": ORDER_DIVERSITY,
            "lex_internal_code": LEX_CODE,
            "order_internal_code": ORDER_CODE,
            "sent_interval": SENT_INTERVAL,
            "do_sample": DO_SAMPLE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "max_length": MAX_LENGTH,
            "global_seed": GLOBAL_SEED,
            "initial_prefix": INITIAL_PREFIX,
        },
        "runtime_seconds": {
            "total_generation_only": sum(all_runtime),
            "mean": statistics.mean(all_runtime) if all_runtime else None,
            "median": statistics.median(all_runtime) if all_runtime else None,
            "wall_since_program_generation_start": time.time() - run_start,
            "model_load": dipper.model_load_seconds,
        },
        "final_output": str(final_path),
        "final_output_sha256": sha256_file(final_path),
        "failures_output": str(failures_path),
        "failures_output_sha256": (
            sha256_file(failures_path)
            if failures_path.exists()
            else None
        ),
    }

    write_json_atomic(summary_path, summary)

    manifest = {
        "status": "FROZEN_GENERATION_OUTPUT",
        "run_started_at": run_started_at,
        "run_finished_at": utc_now(),
        "hostname": platform.node(),
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "nltk": nltk_version(),
        "cuda_runtime": torch.version.cuda,
        "gpu_names": [
            torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())
        ],
        "device_map": getattr(dipper.model, "hf_device_map", None),
        "max_memory": dipper.max_memory,
        "official_repo": OFFICIAL_REPO,
        "dipper_repo_commit": git_commit(args.dipper_repo),
        "official_model": OFFICIAL_MODEL,
        "model_path": args.model_path,
        "official_tokenizer": OFFICIAL_TOKENIZER,
        "tokenizer_path": args.tokenizer_path,
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "final_output_path": str(final_path),
        "final_output_sha256": sha256_file(final_path),
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "frozen_protocol": summary["frozen_protocol"],
        "no_detector_feedback": True,
        "no_parameter_sweep": True,
        "no_posthoc_selection": True,
    }

    write_json_atomic(manifest_path, manifest)

    print()
    print("=" * 96)
    print("DIPPER RUN COMPLETE")
    print("=" * 96)
    print("rows             :", len(final_rows))
    print("status           :", dict(statuses))
    print("domains          :", dict(domains))
    print("generators       :", dict(generators))
    print("final SHA256     :", sha256_file(final_path))
    print("summary SHA256   :", sha256_file(summary_path))
    print("manifest SHA256  :", sha256_file(manifest_path))
    print("=" * 96)


if __name__ == "__main__":
    main()
