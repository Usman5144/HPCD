#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


EXPECTED_CONFIG_SHA = (
    "7e847fd0c1a8993e2b80ced8df816944"
    "83a1351acdd7cd947818709ee690400d"
)

EXPECTED_INPUT_SHA = {
    "prompts":
        "e810757c150bd83ad8af1bfb45e53b80066f67c3c2713fe7f97c8b44d9fd639f",

    "human":
        "784b207162863e6a84ffc11043abef5bf33baf89e8d5d68f7c609cf609674c63",

    "qwen25_3b":
        "ef989cf66fadb20bf241256c4e340e27a263194cb57ffd0e841d352e4168b918",

    "phi35_mini":
        "7ca0c07859e22c7074a93ae4b0adad2dac60281f6dfdca3d23ac3339249758b0",

    "llama32_3b":
        "5e0f4c326e64110c88d43a386505903d282f3fa26c148cac8d6a7af89381343f",
}

EXPECTED_COMMON = 251

EXPECTED_DOMAINS = {
    "news": 69,
    "reddit": 46,
    "reviews": 68,
    "wiki": 68,
}

EXPECTED_MODEL_OUTPUT_VOCAB = 151936
EXPECTED_TOKENIZER_LEN = 151665

SOURCE_ORDER = (
    "human",
    "qwen25_3b",
    "phi35_mini",
    "llama32_3b",
)

TEXT_FIELDS = {
    "human": "human_reference",
    "qwen25_3b": "generated_text",
    "phi35_mini": "generated_text",
    "llama32_3b": "generated_text",
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--limit_ids",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--position_chunk",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--local_files_only",
        action="store_true",
    )

    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(block)

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
    ) as f:

        for line_no, line in enumerate(
            f,
            1,
        ):
            if not line.strip():
                continue

            try:
                rows.append(
                    json.loads(line)
                )

            except Exception as exc:
                raise RuntimeError(
                    f"{path}: malformed JSON "
                    f"line {line_no}: {exc}"
                ) from exc

    return rows


def atomic_json_write(
    path: Path,
    obj,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_name(
        path.name
        +
        ".tmp"
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

        f.write(
            "\n"
        )

        f.flush()

        os.fsync(
            f.fileno()
        )

    os.replace(
        tmp,
        path,
    )


def atomic_npz_write(
    path: Path,
    arrays: dict,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_name(
        path.stem
        +
        ".tmp.npz"
    )

    np.savez_compressed(
        tmp,
        **arrays,
    )

    os.replace(
        tmp,
        path,
    )


def choose_device():
    if torch.cuda.is_available():
        return torch.device(
            "cuda"
        )

    if torch.backends.mps.is_available():
        return torch.device(
            "mps"
        )

    return torch.device(
        "cpu"
    )


def clear_device_cache(
    device,
):
    if device.type == "cuda":
        torch.cuda.empty_cache()

    elif device.type == "mps":
        torch.mps.empty_cache()


def memory_snapshot(
    device,
):
    if device.type == "cuda":
        return {
            "allocated_bytes":
                int(
                    torch.cuda.memory_allocated()
                ),

            "max_allocated_bytes":
                int(
                    torch.cuda.max_memory_allocated()
                ),
        }

    if device.type == "mps":
        allocated = None

        if hasattr(
            torch.mps,
            "current_allocated_memory",
        ):
            allocated = int(
                torch.mps.current_allocated_memory()
            )

        return {
            "allocated_bytes":
                allocated,

            "max_allocated_bytes":
                None,
        }

    return {
        "allocated_bytes": None,
        "max_allocated_bytes": None,
    }


def extract_trajectory(
    text,
    tokenizer,
    model,
    device,
    position_chunk,
):
    encoded = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=False,
        return_token_type_ids=False,
    )

    input_ids_cpu = encoded[
        "input_ids"
    ][
        0
    ]

    total_tokens = int(
        input_ids_cpu.shape[
            0
        ]
    )

    if total_tokens < 2:
        raise RuntimeError(
            "Document has fewer than "
            "two tokenizer tokens"
        )

    model_max = getattr(
        model.config,
        "max_position_embeddings",
        None,
    )

    if (
        model_max is not None
        and
        total_tokens > model_max
    ):
        raise RuntimeError(
            f"Document token count "
            f"{total_tokens} exceeds "
            f"model max {model_max}"
        )

    encoded_device = {
        key:
            value.to(
                device
            )
        for key, value
        in encoded.items()
    }

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()

    with torch.inference_mode():
        outputs = model(
            **encoded_device,
            use_cache=False,
        )

    logits = outputs.logits[
        0,
        :-1,
        :
    ]

    target_ids_cpu = input_ids_cpu[
        1:
    ]

    target_ids_device = (
        target_ids_cpu.to(
            device
        )
    )

    scored_tokens = int(
        target_ids_cpu.shape[
            0
        ]
    )

    output_vocab = int(
        logits.shape[
            -1
        ]
    )

    if (
        output_vocab
        !=
        EXPECTED_MODEL_OUTPUT_VOCAB
    ):
        raise RuntimeError(
            f"Unexpected logits vocabulary "
            f"{output_vocab}"
        )

    selected_probability = np.empty(
        scored_tokens,
        dtype=np.float32,
    )

    surprisal = np.empty(
        scored_tokens,
        dtype=np.float32,
    )

    raw_rank = np.empty(
        scored_tokens,
        dtype=np.int32,
    )

    mass_position = np.empty(
        scored_tokens,
        dtype=np.float32,
    )

    context_entropy = np.empty(
        scored_tokens,
        dtype=np.float32,
    )

    normalized_entropy = np.empty(
        scored_tokens,
        dtype=np.float32,
    )

    for start in range(
        0,
        scored_tokens,
        position_chunk,
    ):
        end = min(
            start
            +
            position_chunk,
            scored_tokens,
        )

        chunk_logits = (
            logits[
                start:end
            ]
            .float()
        )

        target_chunk = (
            target_ids_device[
                start:end
            ]
        )

        selected_logits = (
            chunk_logits
            .gather(
                1,
                target_chunk.unsqueeze(
                    1
                ),
            )
            .squeeze(
                1
            )
        )

        log_probs = torch.log_softmax(
            chunk_logits,
            dim=-1,
        )

        probs = torch.exp(
            log_probs
        )

        selected_logp = (
            log_probs
            .gather(
                1,
                target_chunk.unsqueeze(
                    1
                ),
            )
            .squeeze(
                1
            )
        )

        selected_prob = torch.exp(
            selected_logp
        )

        rank = (
            (
                chunk_logits
                >
                selected_logits.unsqueeze(
                    1
                )
            )
            .sum(
                dim=-1
            )
            +
            1
        )

        cumulative_mass = (
            probs
            *
            (
                chunk_logits
                >=
                selected_logits.unsqueeze(
                    1
                )
            )
        ).sum(
            dim=-1
        )

        entropy = -(
            probs
            *
            log_probs
        ).sum(
            dim=-1
        )

        norm_entropy = (
            entropy
            /
            math.log(
                output_vocab
            )
        )

        sl = slice(
            start,
            end,
        )

        selected_probability[
            sl
        ] = (
            selected_prob
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        surprisal[
            sl
        ] = (
            -selected_logp
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        raw_rank[
            sl
        ] = (
            rank
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.int32
            )
        )

        mass_position[
            sl
        ] = (
            cumulative_mass
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        context_entropy[
            sl
        ] = (
            entropy
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        normalized_entropy[
            sl
        ] = (
            norm_entropy
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        del chunk_logits
        del target_chunk
        del selected_logits
        del log_probs
        del probs
        del selected_logp
        del selected_prob
        del rank
        del cumulative_mass
        del entropy
        del norm_entropy

    runtime = float(
        time.time()
        -
        start_time
    )

    arrays = {
        "token_ids":
            target_ids_cpu
            .numpy()
            .astype(
                np.int32
            ),

        "selected_probability":
            selected_probability,

        "surprisal":
            surprisal,

        "raw_rank":
            raw_rank,

        "probability_mass_position":
            mass_position,

        "context_entropy":
            context_entropy,

        "normalized_context_entropy":
            normalized_entropy,
    }

    for name, values in arrays.items():
        if name == "token_ids":
            continue

        if not np.all(
            np.isfinite(
                values
            )
        ):
            raise RuntimeError(
                f"Non-finite values in {name}"
            )

    if not np.all(
        (
            selected_probability
            >
            0
        )
        &
        (
            selected_probability
            <=
            1.00001
        )
    ):
        raise RuntimeError(
            "Invalid selected probability"
        )

    if not np.all(
        (
            mass_position
            >=
            selected_probability
            -
            1e-5
        )
        &
        (
            mass_position
            <=
            1.00002
        )
    ):
        raise RuntimeError(
            "Invalid probability-mass "
            "position"
        )

    if not np.all(
        (
            raw_rank
            >=
            1
        )
        &
        (
            raw_rank
            <=
            output_vocab
        )
    ):
        raise RuntimeError(
            "Invalid rank"
        )

    identity_error = np.max(
        np.abs(
            np.exp(
                -surprisal.astype(
                    np.float64
                )
            )
            -
            selected_probability.astype(
                np.float64
            )
        )
    )

    if (
        float(
            identity_error
        )
        >
        2e-5
    ):
        raise RuntimeError(
            f"Probability-surprisal "
            f"identity error "
            f"{identity_error}"
        )

    metadata = {
        "token_count_with_specials":
            total_tokens,

        "scored_token_count":
            scored_tokens,

        "unscored_initial_tokens":
            (
                total_tokens
                -
                scored_tokens
            ),

        "model_output_vocab_size":
            output_vocab,

        "mean_selected_probability":
            float(
                selected_probability.mean()
            ),

        "mean_surprisal":
            float(
                surprisal.mean()
            ),

        "mean_raw_rank":
            float(
                raw_rank.mean()
            ),

        "mean_probability_mass_position":
            float(
                mass_position.mean()
            ),

        "mean_context_entropy":
            float(
                context_entropy.mean()
            ),

        "runtime_seconds":
            runtime,

        "probability_surprisal_identity_max_error":
            float(
                identity_error
            ),

        "memory":
            memory_snapshot(
                device
            ),
    }

    del outputs
    del logits
    del encoded
    del encoded_device
    del target_ids_device

    clear_device_cache(
        device
    )

    return (
        arrays,
        metadata,
    )


def main():
    args = parse_args()

    root = args.root.resolve()

    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else
        root
        /
        args.output_dir
    ).resolve()

    prompts_path = (
        root
        /
        "data/processed/"
        "raid_pilot_prompts.jsonl"
    )

    human_path = (
        root
        /
        "data/processed/"
        "raid_pilot_human.jsonl"
    )

    qwen_path = (
        root
        /
        "generations/"
        "qwen25_3b_raid_lengthmatched_v13_400_seed43_valid.jsonl"
    )

    phi_path = (
        root
        /
        "generations/"
        "phi35_mini_raid_v13_400_seed43_valid.jsonl"
    )

    llama_path = (
        root
        /
        "generations/"
        "llama32_3b_raid_v13_400_seed43_valid.jsonl"
    )

    config_path = (
        root
        /
        "configs/hjtm/"
        "t1b_full_trajectory_v1.json"
    )

    source_paths = {
        "human":
            human_path,

        "qwen25_3b":
            qwen_path,

        "phi35_mini":
            phi_path,

        "llama32_3b":
            llama_path,
    }

    print(
        "=" * 100
    )

    print(
        "HJTM T1B — RESUMABLE "
        "PROBABILITY TRAJECTORY EXTRACTION"
    )

    print(
        "=" * 100
    )

    print(
        "\n===== CONFIG / INPUT VERIFICATION ====="
    )

    config_sha = sha256_file(
        config_path
    )

    print(
        "config SHA256:",
        config_sha,
    )

    if (
        config_sha
        !=
        EXPECTED_CONFIG_SHA
    ):
        raise RuntimeError(
            "Frozen T1B config "
            "checksum mismatch"
        )

    actual_input_sha = {}

    input_paths = {
        "prompts":
            prompts_path,

        **source_paths,
    }

    for name, path in (
        input_paths.items()
    ):
        digest = sha256_file(
            path
        )

        actual_input_sha[
            name
        ] = digest

        print(
            name,
            digest,
        )

        if (
            digest
            !=
            EXPECTED_INPUT_SHA[
                name
            ]
        ):
            raise RuntimeError(
                f"{name} checksum mismatch"
            )

    prompts = load_jsonl(
        prompts_path
    )

    human_rows = load_jsonl(
        human_path
    )

    qwen_rows = load_jsonl(
        qwen_path
    )

    phi_rows = load_jsonl(
        phi_path
    )

    llama_rows = load_jsonl(
        llama_path
    )

    signature_prompts = {
        row[
            "sample_id"
        ]:
            row
        for row in prompts
        if (
            row.get(
                "partition"
            )
            ==
            "signature_train"
        )
    }

    if len(
        signature_prompts
    ) != 280:
        raise RuntimeError(
            "Expected 280 signature "
            "training prompts"
        )

    human_map = {
        row[
            "sample_id"
        ]:
            row
        for row in human_rows
        if (
            row[
                "sample_id"
            ]
            in
            signature_prompts
        )
    }

    machine_maps = {
        "qwen25_3b": {
            row[
                "sample_id"
            ]:
                row
            for row in qwen_rows
            if (
                row.get(
                    "partition"
                )
                ==
                "signature_train"
            )
        },

        "phi35_mini": {
            row[
                "sample_id"
            ]:
                row
            for row in phi_rows
            if (
                row.get(
                    "partition"
                )
                ==
                "signature_train"
            )
        },

        "llama32_3b": {
            row[
                "sample_id"
            ]:
                row
            for row in llama_rows
            if (
                row.get(
                    "partition"
                )
                ==
                "signature_train"
            )
        },
    }

    source_maps = {
        "human":
            human_map,

        **machine_maps,
    }

    common_ids = (
        set(
            signature_prompts
        )
        &
        set(
            human_map
        )
        &
        set(
            machine_maps[
                "qwen25_3b"
            ]
        )
        &
        set(
            machine_maps[
                "phi35_mini"
            ]
        )
        &
        set(
            machine_maps[
                "llama32_3b"
            ]
        )
    )

    if len(
        common_ids
    ) != EXPECTED_COMMON:
        raise RuntimeError(
            f"Expected 251 common IDs, "
            f"found {len(common_ids)}"
        )

    domain_by_id = {
        sid:
            signature_prompts[
                sid
            ][
                "domain"
            ]
        for sid
        in common_ids
    }

    domain_counts = Counter(
        domain_by_id.values()
    )

    print(
        "common IDs:",
        len(
            common_ids
        ),
    )

    print(
        "domains:",
        dict(
            domain_counts
        ),
    )

    if dict(
        domain_counts
    ) != EXPECTED_DOMAINS:
        raise RuntimeError(
            "Common-domain distribution "
            "mismatch"
        )

    selected_ids = sorted(
        common_ids
    )

    if args.limit_ids is not None:
        if args.limit_ids < 1:
            raise RuntimeError(
                "--limit_ids must be >=1"
            )

        selected_ids = selected_ids[
            :args.limit_ids
        ]

    expected_documents = (
        len(
            selected_ids
        )
        *
        len(
            SOURCE_ORDER
        )
    )

    print(
        "selected IDs:",
        len(
            selected_ids
        ),
    )

    print(
        "expected documents:",
        expected_documents,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    trajectory_root = (
        output_dir
        /
        "trajectories"
    )

    failure_path = (
        output_dir
        /
        "failures.jsonl"
    )

    final_manifest_path = (
        output_dir
        /
        "manifest.jsonl"
    )

    summary_path = (
        output_dir
        /
        "summary.json"
    )

    run_metadata_path = (
        output_dir
        /
        "run_metadata.json"
    )

    device = choose_device()

    print(
        "\n===== LOAD REFERENCE LM ====="
    )

    print(
        "device:",
        device,
    )

    print(
        "model_path:",
        args.model_path,
    )

    print(
        "local_files_only:",
        args.local_files_only,
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            args.model_path,
            local_files_only=(
                args.local_files_only
            ),
        )
    )

    if (
        len(
            tokenizer
        )
        !=
        EXPECTED_TOKENIZER_LEN
    ):
        raise RuntimeError(
            f"Unexpected tokenizer length "
            f"{len(tokenizer)}"
        )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            args.model_path,
            local_files_only=(
                args.local_files_only
            ),
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    )

    model.to(
        device
    )

    model.eval()

    if (
        int(
            model.config.vocab_size
        )
        !=
        EXPECTED_MODEL_OUTPUT_VOCAB
    ):
        raise RuntimeError(
            "Unexpected model vocab size"
        )

    print(
        "dtype:",
        next(
            model.parameters()
        ).dtype,
    )

    print(
        "model_type:",
        model.config.model_type,
    )

    print(
        "config vocab_size:",
        model.config.vocab_size,
    )

    print(
        "tokenizer length:",
        len(
            tokenizer
        ),
    )

    print(
        "max_position_embeddings:",
        getattr(
            model.config,
            "max_position_embeddings",
            None,
        ),
    )

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )

        print(
            "CUDA runtime:",
            torch.version.cuda,
        )

    run_metadata = {
        "experiment_id":
            (
                "HJTM_T1B_FULL_TRAJECTORY_V1"
                if args.limit_ids is None
                else
                "HJTM_T1B_MAC_SMOKE_V1"
            ),

        "scientific_stage":
            "continuous trajectory extraction only",

        "joint_probability_fitted":
            False,

        "state_bins_fitted":
            False,

        "root":
            str(
                root
            ),

        "model_path":
            str(
                args.model_path
            ),

        "device":
            str(
                device
            ),

        "dtype":
            str(
                next(
                    model.parameters()
                ).dtype
            ),

        "torch":
            torch.__version__,

        "transformers":
            transformers.__version__,

        "numpy":
            np.__version__,

        "python":
            sys.version,

        "platform":
            platform.platform(),

        "cuda_runtime":
            torch.version.cuda,

        "position_chunk":
            args.position_chunk,

        "common_ids":
            EXPECTED_COMMON,

        "selected_ids":
            len(
                selected_ids
            ),

        "expected_documents":
            expected_documents,

        "config_sha256":
            config_sha,

        "input_sha256":
            actual_input_sha,

        "probability_support":
            "full_model_output_logits",

        "model_output_vocab_size":
            EXPECTED_MODEL_OUTPUT_VOCAB,

        "tokenizer_length":
            EXPECTED_TOKENIZER_LEN,
    }

    atomic_json_write(
        run_metadata_path,
        run_metadata,
    )

    print(
        "\n===== RESUME AUDIT ====="
    )

    completed = {}

    for source_type in SOURCE_ORDER:

        source_dir = (
            trajectory_root
            /
            source_type
        )

        source_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for sid in selected_ids:

            npz_path = (
                source_dir
                /
                f"{sid}.npz"
            )

            metadata_path = (
                source_dir
                /
                f"{sid}.json"
            )

            if (
                npz_path.exists()
                and
                not metadata_path.exists()
            ):
                print(
                    "removing orphan NPZ from "
                    "interrupted write:",
                    npz_path,
                )

                npz_path.unlink()

            if (
                metadata_path.exists()
                and
                not npz_path.exists()
            ):
                raise RuntimeError(
                    f"Metadata exists without "
                    f"trajectory: "
                    f"{metadata_path}"
                )

            if (
                npz_path.exists()
                and
                metadata_path.exists()
            ):
                meta = json.loads(
                    metadata_path.read_text(
                        encoding="utf-8"
                    )
                )

                observed_sha = sha256_file(
                    npz_path
                )

                if (
                    observed_sha
                    !=
                    meta[
                        "trajectory_sha256"
                    ]
                ):
                    raise RuntimeError(
                        "Existing trajectory "
                        "checksum mismatch: "
                        f"{npz_path}"
                    )

                completed[
                    (
                        source_type,
                        sid,
                    )
                ] = meta

    print(
        "already completed:",
        len(
            completed
        ),
        "/",
        expected_documents,
    )

    print(
        "\n===== EXTRACTION ====="
    )

    extraction_start = time.time()

    processed_this_run = 0

    for sid in selected_ids:

        domain = domain_by_id[
            sid
        ]

        for source_type in SOURCE_ORDER:

            key = (
                source_type,
                sid,
            )

            if key in completed:
                print(
                    "SKIP",
                    source_type,
                    sid,
                    "already complete",
                    flush=True,
                )

                continue

            row = source_maps[
                source_type
            ][
                sid
            ]

            row_domain = row.get(
                "domain"
            )

            if (
                row_domain is not None
                and
                row_domain != domain
            ):
                raise RuntimeError(
                    f"Domain mismatch: "
                    f"{source_type}/{sid}"
                )

            text_field = TEXT_FIELDS[
                source_type
            ]

            text = row.get(
                text_field
            )

            if (
                not isinstance(
                    text,
                    str,
                )
                or
                not text.strip()
            ):
                raise RuntimeError(
                    f"Missing text: "
                    f"{source_type}/{sid}"
                )

            source_dir = (
                trajectory_root
                /
                source_type
            )

            npz_path = (
                source_dir
                /
                f"{sid}.npz"
            )

            metadata_path = (
                source_dir
                /
                f"{sid}.json"
            )

            try:
                arrays, stats = (
                    extract_trajectory(
                        text=text,
                        tokenizer=tokenizer,
                        model=model,
                        device=device,
                        position_chunk=(
                            args.position_chunk
                        ),
                    )
                )

                atomic_npz_write(
                    npz_path,
                    arrays,
                )

                trajectory_sha = (
                    sha256_file(
                        npz_path
                    )
                )

                metadata = {
                    "sample_id":
                        sid,

                    "source_id":
                        row.get(
                            "source_id"
                        ),

                    "domain":
                        domain,

                    "source_type":
                        source_type,

                    "text_field":
                        text_field,

                    "text_sha256":
                        sha256_text(
                            text
                        ),

                    "trajectory_file":
                        str(
                            npz_path.relative_to(
                                root
                            )
                        ),

                    "trajectory_sha256":
                        trajectory_sha,

                    **stats,
                }

                atomic_json_write(
                    metadata_path,
                    metadata,
                )

                completed[
                    key
                ] = metadata

                processed_this_run += 1

                print(
                    f"DONE "
                    f"{len(completed):04d}/"
                    f"{expected_documents:04d}",
                    source_type,
                    domain,
                    sid,
                    "tokens=",
                    stats[
                        "scored_token_count"
                    ],
                    "mean_u=",
                    round(
                        stats[
                            "mean_probability_mass_position"
                        ],
                        6,
                    ),
                    "mean_s=",
                    round(
                        stats[
                            "mean_surprisal"
                        ],
                        6,
                    ),
                    "sec=",
                    round(
                        stats[
                            "runtime_seconds"
                        ],
                        3,
                    ),
                    flush=True,
                )

            except Exception as exc:

                failure = {
                    "sample_id":
                        sid,

                    "domain":
                        domain,

                    "source_type":
                        source_type,

                    "error":
                        repr(
                            exc
                        ),

                    "timestamp":
                        time.time(),
                }

                with failure_path.open(
                    "a",
                    encoding="utf-8",
                ) as f:
                    f.write(
                        json.dumps(
                            failure,
                            ensure_ascii=False,
                        )
                        +
                        "\n"
                    )

                    f.flush()

                    os.fsync(
                        f.fileno()
                    )

                raise

    print(
        "\n===== FINAL VALIDATION ====="
    )

    expected_keys = {
        (
            source_type,
            sid,
        )
        for source_type
        in SOURCE_ORDER
        for sid
        in selected_ids
    }

    if (
        set(
            completed
        )
        !=
        expected_keys
    ):
        missing = sorted(
            expected_keys
            -
            set(
                completed
            )
        )

        raise RuntimeError(
            f"Extraction incomplete. "
            f"Missing {len(missing)} "
            f"documents."
        )

    manifest_rows = [
        completed[
            (
                source_type,
                sid,
            )
        ]
        for sid in selected_ids
        for source_type
        in SOURCE_ORDER
    ]

    tmp_manifest = (
        final_manifest_path
        .with_name(
            final_manifest_path.name
            +
            ".tmp"
        )
    )

    with tmp_manifest.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in manifest_rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                +
                "\n"
            )

        f.flush()

        os.fsync(
            f.fileno()
        )

    os.replace(
        tmp_manifest,
        final_manifest_path,
    )

    source_counts = Counter(
        row[
            "source_type"
        ]
        for row in manifest_rows
    )

    domain_counts_out = Counter(
        row[
            "domain"
        ]
        for row in manifest_rows
    )

    total_scored_tokens = sum(
        int(
            row[
                "scored_token_count"
            ]
        )
        for row in manifest_rows
    )

    runtime_total = float(
        time.time()
        -
        extraction_start
    )

    summary = {
        "status":
            "PASS",

        "experiment_id":
            run_metadata[
                "experiment_id"
            ],

        "documents":
            len(
                manifest_rows
            ),

        "source_counts":
            dict(
                source_counts
            ),

        "domain_counts":
            dict(
                domain_counts_out
            ),

        "total_scored_tokens":
            total_scored_tokens,

        "processed_this_run":
            processed_this_run,

        "runtime_this_invocation_seconds":
            runtime_total,

        "manifest":
            str(
                final_manifest_path
                .relative_to(
                    root
                )
            ),

        "manifest_sha256":
            sha256_file(
                final_manifest_path
            ),

        "run_metadata":
            str(
                run_metadata_path
                .relative_to(
                    root
                )
            ),

        "run_metadata_sha256":
            sha256_file(
                run_metadata_path
            ),

        "config_sha256":
            config_sha,

        "input_sha256":
            actual_input_sha,

        "joint_probability_fitted":
            False,

        "state_bins_fitted":
            False,
    }

    atomic_json_write(
        summary_path,
        summary,
    )

    print(
        "documents:",
        len(
            manifest_rows
        ),
    )

    print(
        "source counts:",
        dict(
            source_counts
        ),
    )

    print(
        "domain counts:",
        dict(
            domain_counts_out
        ),
    )

    print(
        "total scored tokens:",
        total_scored_tokens,
    )

    print(
        "manifest SHA256:",
        summary[
            "manifest_sha256"
        ],
    )

    print(
        "run metadata SHA256:",
        summary[
            "run_metadata_sha256"
        ],
    )

    print(
        "summary SHA256:",
        sha256_file(
            summary_path
        ),
    )

    print(
        "script SHA256:",
        sha256_file(
            Path(
                __file__
            ).resolve()
        ),
    )

    print()
    print(
        "=" * 100
    )

    print(
        "HJTM T1B TRAJECTORY "
        "EXTRACTION: PASS"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()
