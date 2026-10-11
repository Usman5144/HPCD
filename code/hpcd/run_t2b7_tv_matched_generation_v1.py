#!/usr/bin/env python3
from __future__ import annotations

import os

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hpcd_controller_v1 import (
    HJTMModel,
    HPCDController,
)

from hpcd_reference_aligned_v2 import (
    ReferenceAlignedHPCDController,
)


ROOT = Path(os.environ.get("HPCD_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))

MODEL_PATH = Path(
    "/path/to/huggingface/hub/"
    "models--Qwen--Qwen2.5-3B-Instruct/"
    "snapshots/"
    "aa8e72537993ba99e69dfaafa59ed015b17504d1"
)

HJTM_PATH = (
    ROOT
    / "results/hjtm/final_hjtm_v1/"
      "hjtm_final_v1.json"
)

OUT_DIR = (
    ROOT
    / "results/hpcd/t2b7_tv_matched_generation_v1"
)

OUT_JSONL = (
    OUT_DIR
    / "generations.jsonl"
)

SUMMARY_PATH = (
    OUT_DIR
    / "summary.json"
)

SOURCE_FILES = {
    "qwen25_3b":
        ROOT
        / "generations/"
          "qwen25_3b_raid_lengthmatched_v13_400_seed43_valid.jsonl",

    "phi35_mini":
        ROOT
        / "generations/"
          "phi35_mini_raid_v13_400_seed43_valid.jsonl",

    "llama32_3b":
        ROOT
        / "generations/"
          "llama32_3b_raid_v13_400_seed43_valid.jsonl",
}

EXPECTED_SOURCE_SHA = {
    "qwen25_3b":
        "ef989cf66fadb20bf241256c4e340e27a263194cb57ffd0e841d352e4168b918",

    "phi35_mini":
        "7ca0c07859e22c7074a93ae4b0adad2dac60281f6dfdca3d23ac3339249758b0",

    "llama32_3b":
        "5e0f4c326e64110c88d43a386505903d282f3fa26c148cac8d6a7af89381343f",
}

DOMAIN_SOURCE = {
    "news":
        "qwen25_3b",

    "reddit":
        "phi35_mini",

    "reviews":
        "llama32_3b",

    "wiki":
        "qwen25_3b",
}

DOMAINS = (
    "news",
    "reddit",
    "reviews",
    "wiki",
)

SEED = 2027

TEMPERATURE = 1.0

TOP_P = 0.95

BETA_N0 = 0.20
BETA_N1 = 0.25
BETA_N2 = 0.25

# T2B uses the shortest genuine whole document
# available in each requested domain. We do not
# truncate source documents merely to make the
# smoke test cheaper.
MAX_SOURCE_TOKENS = None

MAX_NEW_TOKENS_CAP = 320


SYSTEM_PROMPT = (
    "You are a meaning-preserving paraphraser. "
    "Rewrite the passage with clearly different wording "
    "and sentence structure while preserving every factual "
    "claim exactly. Do not merely copy the source. "
    "Preserve named entities, numbers, dates, attribution, "
    "negation, modality, quotations, and factual relations. "
    "Do not add claims and do not remove claims. "
    "Output only the rewritten passage."
)


def sha256_file(path: Path) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(
                chunk
            )

    return h.hexdigest()


def stable_hash(text: str) -> str:

    return hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()


def load_jsonl(path: Path):

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            if line.strip():

                rows.append(
                    json.loads(
                        line
                    )
                )

    return rows


def get_text(row):

    if (
        "generated_text"
        not in row
    ):
        raise RuntimeError(
            "Expected generated_text key. "
            f"Keys={sorted(row.keys())}"
        )

    text = row[
        "generated_text"
    ]

    if not isinstance(
        text,
        str,
    ):
        raise RuntimeError(
            "generated_text is not string"
        )

    return text.strip()


def get_id(row):

    for key in (
        "sample_id",
        "id",
    ):

        if key in row:

            return str(
                row[
                    key
                ]
            )

    raise RuntimeError(
        "No sample_id/id key"
    )


def get_domain(row):

    if (
        "domain"
        not in row
    ):
        raise RuntimeError(
            "No domain key"
        )

    return str(
        row[
            "domain"
        ]
    )


def offline_score(
    token_ids,
    *,
    model,
    tokenizer,
    hjtm,
    device,
    domain,
    order,
):

    eos = tokenizer.eos_token_id

    clean = []

    for token_id in token_ids:

        if (
            eos is not None
            and
            int(
                token_id
            )
            ==
            int(
                eos
            )
        ):
            break

        clean.append(
            int(
                token_id
            )
        )

    if len(
        clean
    ) < 2:

        return {
            "nll":
                float("nan"),

            "typicality":
                float("nan"),

            "states":
                [],
        }

    ids = torch.tensor(
        [
            clean
        ],
        dtype=torch.long,
        device=device,
    )

    with torch.inference_mode():

        logits = model(
            input_ids=ids,
            use_cache=False,
        ).logits[
            0,
            :-1,
            :
        ].float()

    targets = ids[
        0,
        1:
    ]

    states = []

    entropy_states = []

    for i in range(
        len(
            targets
        )
    ):

        z = logits[
            i
        ]

        target = int(
            targets[
                i
            ].item()
        )

        lp = torch.log_softmax(
            z,
            dim=-1,
        )

        p = torch.exp(
            lp
        )

        selected_logit = z[
            target
        ]

        u = float(
            p[
                z
                >=
                selected_logit
            ]
            .sum()
            .item()
        )

        H = float(
            (
                -p
                *
                lp
            )
            .sum()
            .item()
        )

        b = hjtm.mass_state(
            u
        )

        e = hjtm.entropy_state(
            H
        )

        states.append(
            b
        )

        entropy_states.append(
            e
        )

    history = []

    costs = []

    for b, e in zip(
        states,
        entropy_states,
    ):

        ph = (
            hjtm
            .conditional_vector(
                history=history,
                entropy_state=e,
                order=order,
            )
        )

        cost = -math.log(
            float(
                ph[
                    b
                ]
            )
        )

        costs.append(
            cost
        )

        history.append(
            b
        )

    nll = float(
        np.mean(
            costs
        )
    )

    mu, sd = (
        hjtm
        .calibration_for(
            order=order,
            domain=domain,
        )
    )

    typicality = abs(
        (
            nll
            -
            mu
        )
        /
        sd
    )

    return {
        "nll":
            nll,

        "typicality":
            float(
                typicality
            ),

        "states":
            [
                int(x)
                for x in states
            ],
    }


def generate_one(
    *,
    source_text,
    domain,
    order,
    beta,
    label,
    mode,
    model,
    tokenizer,
    hjtm,
    controller,
    device,
):

    messages = [
        {
            "role":
                "system",

            "content":
                SYSTEM_PROMPT,
        },
        {
            "role":
                "user",

            "content":
                (
                    "Paraphrase the following passage "
                    "substantially in wording and syntax "
                    "while preserving all information "
                    "exactly:\n\n"
                    +
                    source_text
                ),
        },
    ]

    encoded = (
        tokenizer
        .apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    )

    if isinstance(
        encoded,
        torch.Tensor,
    ):

        prompt_ids = encoded

    elif hasattr(
        encoded,
        "input_ids",
    ):

        prompt_ids = encoded.input_ids

    else:

        prompt_ids = encoded[
            "input_ids"
        ]

    prompt_ids = prompt_ids.to(
        device
    )

    source_tokens = len(
        tokenizer(
            source_text,
            add_special_tokens=False,
        )[
            "input_ids"
        ]
    )

    max_new_tokens = min(
        MAX_NEW_TOKENS_CAP,
        max(
            96,
            int(
                math.ceil(
                    1.35
                    *
                    source_tokens
                )
            )
            +
            32,
        ),
    )

    rng = torch.Generator(
        device="cpu"
    )

    rng.manual_seed(
        SEED
    )

    sequence = (
        prompt_ids.clone()
    )

    generated = []

    history = []

    cumulative_cost = 0.0

    support_sizes = []

    controllable_steps = 0

    changed_distribution_steps = 0

    tv_values = []

    eos_ids = []

    if (
        tokenizer.eos_token_id
        is not None
    ):

        eos_ids.append(
            int(
                tokenizer.eos_token_id
            )
        )

    for step in range(
        max_new_tokens
    ):

        with torch.inference_mode():

            base_logits = model(
                input_ids=sequence,
                use_cache=False,
            ).logits[
                0,
                -1,
                :
            ]

        if len(
            generated
        ) == 0:

            base = (
                HPCDController
                ._base_distribution(
                    logits=base_logits,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    top_k=None,
                )
            )

            q = base[
                "base_full"
            ]

            support_size = int(
                base[
                    "support_size"
                ]
            )

            base_distribution = q

            result = None

        else:

            reference_ids = torch.tensor(
                [
                    generated
                ],
                dtype=torch.long,
                device=device,
            )

            with torch.inference_mode():

                reference_logits = model(
                    input_ids=reference_ids,
                    use_cache=False,
                ).logits[
                    0,
                    -1,
                    :
                ]

            result = (
                controller
                .control_reference_aligned(
                    base_logits=base_logits,
                    reference_logits=reference_logits,
                    history=history,
                    cumulative_cost=cumulative_cost,
                    order=order,
                    beta=beta,
                    domain=domain,
                    mode=mode,
                    terminal_token_ids=eos_ids,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    top_k=None,
                )
            )

            q = result[
                "distribution"
            ]

            base_distribution = result[
                "base_distribution"
            ]

            support_size = int(
                result[
                    "support_size"
                ]
            )

            tv = float(
                0.5
                *
                torch.sum(
                    torch.abs(
                        q
                        -
                        base_distribution
                    )
                )
                .item()
            )

            tv_values.append(
                tv
            )

            if (
                support_size
                >=
                2
            ):

                controllable_steps += 1

            if tv > 1e-8:

                changed_distribution_steps += 1

        support_sizes.append(
            support_size
        )

        token_id = int(
            torch.multinomial(
                q.detach().cpu(),
                1,
                generator=rng,
            ).item()
        )

        generated.append(
            token_id
        )

        sequence = torch.cat(
            [
                sequence,

                torch.tensor(
                    [
                        [
                            token_id
                        ]
                    ],
                    dtype=sequence.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )

        if token_id in eos_ids:
            break

        if result is not None:

            state = int(
                result[
                    "state_by_token"
                ][
                    token_id
                ]
                .item()
            )

            cost = float(
                result[
                    "cost_by_token"
                ][
                    token_id
                ]
                .item()
            )

            if (
                state < 0
                or
                not math.isfinite(
                    cost
                )
            ):

                raise RuntimeError(
                    "Invalid selected HJTM "
                    "state/cost"
                )

            history.append(
                state
            )

            cumulative_cost += (
                cost
            )

    text = tokenizer.decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    offline = offline_score(
        generated,
        model=model,
        tokenizer=tokenizer,
        hjtm=hjtm,
        device=device,
        domain=domain,
        order=order,
    )

    total_control_steps = max(
        0,
        len(
            generated
        )
        -
        1,
    )

    return {
        "label":
            label,

        "order":
            int(
                order
            ),

        "beta":
            float(
                beta
            ),

        "mode":
            mode,

        "seed":
            SEED,

        "temperature":
            TEMPERATURE,

        "top_p":
            TOP_P,

        "source_tokens":
            int(
                source_tokens
            ),

        "max_new_tokens":
            int(
                max_new_tokens
            ),

        "generated_tokens":
            int(
                len(
                    generated
                )
            ),

        "token_ids":
            [
                int(x)
                for x in generated
            ],

        "text":
            text,

        "trajectory_nll":
            float(
                offline[
                    "nll"
                ]
            ),

        "trajectory_typicality":
            float(
                offline[
                    "typicality"
                ]
            ),

        "mean_support_size":
            float(
                np.mean(
                    support_sizes
                )
            ),

        "max_support_size":
            int(
                max(
                    support_sizes
                )
            ),

        "control_steps":
            int(
                total_control_steps
            ),

        "controllable_steps":
            int(
                controllable_steps
            ),

        "controllable_fraction":
            (
                float(
                    controllable_steps
                    /
                    total_control_steps
                )
                if total_control_steps
                else 0.0
            ),

        "distribution_changed_steps":
            int(
                changed_distribution_steps
            ),

        "mean_control_tv":
            (
                float(
                    np.mean(
                        tv_values
                    )
                )
                if tv_values
                else 0.0
            ),

        "max_control_tv":
            (
                float(
                    np.max(
                        tv_values
                    )
                )
                if tv_values
                else 0.0
            ),
    }


def main():

    print(
        "=" * 100
    )

    print(
        "HPCD T2B.7 — STRONG-PROMPT MATCHED HPCD"
    )

    print(
        "=" * 100
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        SUMMARY_PATH.exists()
    ):

        raise RuntimeError(
            f"Refusing completed-run overwrite: "
            f"{SUMMARY_PATH}"
        )

    print(
        "\n===== VERIFY INPUTS ====="
    )

    source_rows = {}

    for source_name, path in (
        SOURCE_FILES.items()
    ):

        observed = sha256_file(
            path
        )

        print(
            source_name,
            observed,
        )

        if (
            observed
            !=
            EXPECTED_SOURCE_SHA[
                source_name
            ]
        ):

            raise RuntimeError(
                f"Input hash mismatch: "
                f"{source_name}"
            )

        source_rows[
            source_name
        ] = load_jsonl(
            path
        )

    if not torch.backends.mps.is_available():

        raise RuntimeError(
            "MPS unavailable"
        )

    device = torch.device(
        "mps"
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_PATH,
            local_files_only=True,
        )
    )

    print(
        "\n===== SELECT REAL SMOKE DOCUMENTS ====="
    )

    selected = []

    for domain in DOMAINS:

        source_name = (
            DOMAIN_SOURCE[
                domain
            ]
        )

        candidates = []

        for row in source_rows[
            source_name
        ]:

            if (
                get_domain(
                    row
                )
                !=
                domain
            ):

                continue

            text = get_text(
                row
            )

            token_count = len(
                tokenizer(
                    text,
                    add_special_tokens=False,
                )[
                    "input_ids"
                ]
            )

            candidates.append(
                (
                    int(
                        token_count
                    ),

                    stable_hash(
                        "T2B_REAL_SMOKE|"
                        +
                        get_id(
                            row
                        )
                    ),

                    row,
                )
            )

        if not candidates:

            raise RuntimeError(
                f"No candidate for domain {domain}"
            )

        # Primary key: shortest genuine whole document.
        # Secondary key: deterministic hash for ties.
        candidates.sort(
            key=lambda x: (
                x[0],
                x[1],
            )
        )

        token_count, _, row = (
            candidates[
                0
            ]
        )

        selected.append({
            "sample_id":
                get_id(
                    row
                ),

            "domain":
                domain,

            "source_generator":
                source_name,

            "source_text":
                get_text(
                    row
                ),

            "source_tokens":
                int(
                    token_count
                ),
        })

    expected_ids = {
        "news":
            "raid_train_news_cb2094f392792b11",

        "reddit":
            "raid_train_reddit_60d624b49800dd8a",

        "reviews":
            "raid_train_reviews_75579e48b7dbfe52",

        "wiki":
            "raid_train_wiki_7af67fee2b7efa7b",
    }

    observed_ids = {
        x["domain"]:
            x["sample_id"]
        for x in selected
    }

    if observed_ids != expected_ids:
        raise RuntimeError(
            "Frozen T2B document set changed. "
            f"observed={observed_ids}"
        )

    print(
        "Frozen document IDs verified: PASS"
    )

    for x in selected:

        print(
            x[
                "domain"
            ],
            x[
                "source_generator"
            ],
            x[
                "sample_id"
            ],
            "tokens=",
            x[
                "source_tokens"
            ],
        )

    print(
        "\n===== LOAD PARAPHRASER ====="
    )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    )

    model.to(
        device
    )

    model.eval()

    hjtm = HJTMModel(
        HJTM_PATH
    )

    controller = (
        ReferenceAlignedHPCDController(
            hjtm
        )
    )

    completed_keys = set()

    if OUT_JSONL.exists():

        for row in load_jsonl(
            OUT_JSONL
        ):

            completed_keys.add(
                (
                    row[
                        "sample_id"
                    ],

                    row[
                        "label"
                    ],
                )
            )

        print(
            "resuming completed conditions:",
            len(
                completed_keys
            ),
        )

    conditions = [
        (
            "hpcd_n0",
            0,
            BETA_N0,
        ),

        (
            "hpcd_n1",
            1,
            BETA_N1,
        ),

        (
            "hpcd_n2",
            2,
            BETA_N2,
        ),
    ]

    total = (
        len(
            selected
        )
        *
        len(
            conditions
        )
    )

    condition_index = 0

    for doc in selected:

        print()
        print(
            "#" * 100
        )

        print(
            doc[
                "domain"
            ],
            doc[
                "source_generator"
            ],
            doc[
                "sample_id"
            ],
        )

        print(
            "#" * 100
        )

        print(
            "SOURCE:"
        )

        print(
            doc[
                "source_text"
            ]
        )

        for (
            label,
            order,
            beta,
        ) in conditions:

            condition_index += 1

            key = (
                doc[
                    "sample_id"
                ],
                label,
            )

            if key in completed_keys:

                print(
                    f"\nSKIP "
                    f"{condition_index}/{total} "
                    f"{label}"
                )

                continue

            print()
            print(
                "-" * 80
            )

            print(
                f"RUN "
                f"{condition_index}/{total}: "
                f"{label}"
            )

            print(
                "-" * 80
            )

            rec = generate_one(
                source_text=doc[
                    "source_text"
                ],
                domain=doc[
                    "domain"
                ],
                order=order,
                beta=beta,
                label=label,
                mode="typicality",
                model=model,
                tokenizer=tokenizer,
                hjtm=hjtm,
                controller=controller,
                device=device,
            )

            rec.update({
                "sample_id":
                    doc[
                        "sample_id"
                    ],

                "domain":
                    doc[
                        "domain"
                    ],

                "source_generator":
                    doc[
                        "source_generator"
                    ],

                "source_text":
                    doc[
                        "source_text"
                    ],
            })

            print(
                "generated tokens:",
                rec[
                    "generated_tokens"
                ],
            )

            print(
                "trajectory typicality:",
                round(
                    rec[
                        "trajectory_typicality"
                    ],
                    6,
                ),
            )

            print(
                "mean support:",
                round(
                    rec[
                        "mean_support_size"
                    ],
                    4,
                ),
            )

            print(
                "controllable fraction:",
                round(
                    rec[
                        "controllable_fraction"
                    ],
                    4,
                ),
            )

            print(
                "mean control TV:",
                round(
                    rec[
                        "mean_control_tv"
                    ],
                    6,
                ),
            )

            print(
                "TEXT:"
            )

            print(
                rec[
                    "text"
                ]
            )

            with OUT_JSONL.open(
                "a",
                encoding="utf-8",
            ) as f:

                f.write(
                    json.dumps(
                        rec,
                        ensure_ascii=False,
                    )
                    +
                    "\n"
                )

                f.flush()

            completed_keys.add(
                key
            )

            torch.mps.empty_cache()

    rows = load_jsonl(
        OUT_JSONL
    )

    if len(rows) != total:
        raise RuntimeError(
            f"Expected {total} controlled rows, "
            f"found {len(rows)}"
        )

    summary = {
        "status":
            "COMPLETE",

        "stage":
            "T2B.7 strong-prompt matched HPCD generation",

        "scientific_status":
            (
                "CONTROLLED GENERATION ONLY. "
                "COMMON-METRIC RESCORING FOLLOWS."
            ),

        "temperature":
            TEMPERATURE,

        "top_p":
            TOP_P,

        "betas":
            {
                "hpcd_n0":
                    BETA_N0,

                "hpcd_n1":
                    BETA_N1,

                "hpcd_n2":
                    BETA_N2,
            },

        "seed":
            SEED,

        "documents":
            selected,

        "conditions":
            [
                "hpcd_n0",
                "hpcd_n1",
                "hpcd_n2",
            ],

        "generations_sha256":
            sha256_file(
                OUT_JSONL
            ),
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        +
        "\n",
        encoding="utf-8",
    )

    print()
    print(
        "generations SHA256:",
        summary[
            "generations_sha256"
        ],
    )

    print(
        "summary SHA256:",
        sha256_file(
            SUMMARY_PATH
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
        "HPCD T2B.7 STRONG-PROMPT "
        "MATCHED GENERATION: COMPLETE"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()
