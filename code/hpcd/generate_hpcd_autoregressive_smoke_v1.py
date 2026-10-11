from __future__ import annotations

import os

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hpcd_controller_v1 import HJTMModel, HPCDController


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

CONTROLLER_PATH = (
    ROOT
    / "code/hpcd/"
      "hpcd_controller_v1.py"
)

OUT = (
    ROOT
    / "results/hpcd/"
      "t2a2_autoregressive_smoke_v1.json"
)

EXPECTED_HJTM_SHA = (
    "1abd60c38a8787c4d44cf4e81ec157ea"
    "f18745ce1dec28c366417030b1a22c61"
)

EXPECTED_CONTROLLER_SHA = (
    "d9033d811267a52b3c66d0dd77be8e2"
    "c48f4270bb9d126396b7a8b5567a2ef0f"
)

SEED = 2027
TEMPERATURE = 1.0
TOP_P = 0.95
MAX_NEW_TOKENS = 64
DOMAIN = "news"


SOURCE_TEXT = (
    "The city council approved a new public transport plan on Monday. "
    "The proposal will add twelve electric buses to the network next year "
    "and increase service frequency during weekday peak hours. "
    "Officials said the plan will be funded through the existing transport budget."
)


SYSTEM_PROMPT = (
    "You rewrite text while preserving its factual meaning exactly. "
    "Preserve named entities, numbers, dates, attribution, negation, modality, "
    "and all factual claims. Do not add new claims and do not remove claims. "
    "Write naturally. Output only the rewritten passage."
)


USER_PROMPT = (
    "Rewrite the following passage while preserving all information:\n\n"
    + SOURCE_TEXT
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def run_generation(
    *,
    model,
    tokenizer,
    controller,
    hjtm,
    device,
    prompt_ids,
    label,
    order,
    beta,
    mode,
):

    cpu_generator = torch.Generator(
        device="cpu"
    )

    cpu_generator.manual_seed(
        SEED
    )

    sequence = prompt_ids.clone()

    generated_ids = []

    state_history = []

    cumulative_cost = 0.0

    step_rows = []

    print()
    print(
        "=" * 90
    )
    print(
        label
    )
    print(
        "=" * 90
    )

    eos_ids = set()

    if tokenizer.eos_token_id is not None:
        eos_ids.add(
            int(
                tokenizer.eos_token_id
            )
        )

    for step in range(
        MAX_NEW_TOKENS
    ):

        with torch.inference_mode():

            outputs = model(
                input_ids=sequence,
                use_cache=False,
            )

        logits = outputs.logits[
            0,
            -1,
            :
        ]

        result = controller.control(
            logits=logits,
            history=state_history,
            cumulative_cost=cumulative_cost,
            generated_count=len(
                state_history
            ),
            order=order,
            beta=beta,
            domain=DOMAIN,
            mode=mode,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=None,
        )

        q_cpu = (
            result[
                "distribution"
            ]
            .detach()
            .cpu()
        )

        token_id = int(
            torch.multinomial(
                q_cpu,
                num_samples=1,
                generator=cpu_generator,
            ).item()
        )

        state, cost = (
            controller.observe_selected(
                result,
                token_id,
            )
        )

        state_history.append(
            state
        )

        cumulative_cost += cost

        generated_ids.append(
            token_id
        )

        sequence = torch.cat(
            [
                sequence,
                torch.tensor(
                    [[token_id]],
                    dtype=sequence.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )

        current_nll = (
            cumulative_cost
            /
            len(
                state_history
            )
        )

        mu, sd = (
            hjtm.calibration_for(
                order=order,
                domain=DOMAIN,
            )
        )

        current_typicality = abs(
            (
                current_nll
                -
                mu
            )
            /
            sd
        )

        step_rows.append({
            "step":
                step + 1,

            "token_id":
                token_id,

            "token_text":
                tokenizer.decode(
                    [token_id],
                    clean_up_tokenization_spaces=False,
                ),

            "mass_state":
                state,

            "entropy_state":
                int(
                    result[
                        "entropy_state"
                    ]
                ),

            "support_size":
                int(
                    result[
                        "support_size"
                    ]
                ),

            "hjtm_cost":
                float(
                    cost
                ),

            "running_nll":
                float(
                    current_nll
                ),

            "running_typicality":
                float(
                    current_typicality
                ),

            "expected_cost_base":
                float(
                    result[
                        "expected_cost_base"
                    ]
                ),

            "expected_cost_controlled":
                float(
                    result[
                        "expected_cost_controlled"
                    ]
                ),

            "expected_penalty_base":
                float(
                    result[
                        "expected_penalty_base"
                    ]
                ),

            "expected_penalty_controlled":
                float(
                    result[
                        "expected_penalty_controlled"
                    ]
                ),
        })

        del outputs
        del logits
        del q_cpu

        if token_id in eos_ids:
            break

    text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    final_nll = (
        cumulative_cost
        /
        len(
            state_history
        )
    )

    mu, sd = hjtm.calibration_for(
        order=order,
        domain=DOMAIN,
    )

    final_typicality = abs(
        (
            final_nll
            -
            mu
        )
        /
        sd
    )

    support_sizes = [
        x[
            "support_size"
        ]
        for x in step_rows
    ]

    states = [
        x[
            "mass_state"
        ]
        for x in step_rows
    ]

    print(
        "tokens:",
        len(
            generated_ids
        )
    )

    print(
        "final trajectory NLL:",
        round(
            final_nll,
            6,
        )
    )

    print(
        "human target mu:",
        round(
            mu,
            6,
        )
    )

    print(
        "human target sd:",
        round(
            sd,
            6,
        )
    )

    print(
        "final typicality:",
        round(
            final_typicality,
            6,
        )
    )

    print(
        "mean support size:",
        round(
            float(
                np.mean(
                    support_sizes
                )
            ),
            2,
        )
    )

    print(
        "states:",
        states
    )

    print()
    print(
        "TEXT:"
    )

    print(
        text
    )

    torch.mps.empty_cache()

    return {
        "label":
            label,

        "mode":
            mode,

        "order":
            order,

        "beta":
            beta,

        "seed":
            SEED,

        "temperature":
            TEMPERATURE,

        "top_p":
            TOP_P,

        "generated_tokens":
            len(
                generated_ids
            ),

        "generated_token_ids":
            generated_ids,

        "text":
            text,

        "state_history":
            states,

        "cumulative_hjtm_cost":
            float(
                cumulative_cost
            ),

        "trajectory_nll":
            float(
                final_nll
            ),

        "human_target_mu":
            float(
                mu
            ),

        "human_target_sd":
            float(
                sd
            ),

        "trajectory_typicality":
            float(
                final_typicality
            ),

        "mean_support_size":
            float(
                np.mean(
                    support_sizes
                )
            ),

        "min_support_size":
            int(
                min(
                    support_sizes
                )
            ),

        "max_support_size":
            int(
                max(
                    support_sizes
                )
            ),

        "steps":
            step_rows,
    }


def main():

    print(
        "=" * 100
    )

    print(
        "HPCD T2A-2 — AUTOREGRESSIVE GENERATION SMOKE"
    )

    print(
        "=" * 100
    )

    if OUT.exists():
        raise RuntimeError(
            f"Refusing overwrite: {OUT}"
        )

    hjtm_sha = sha256_file(
        HJTM_PATH
    )

    controller_sha = sha256_file(
        CONTROLLER_PATH
    )

    print(
        "HJTM SHA256:",
        hjtm_sha
    )

    print(
        "controller SHA256:",
        controller_sha
    )

    if hjtm_sha != EXPECTED_HJTM_SHA:
        raise RuntimeError(
            "HJTM hash mismatch"
        )

    if (
        controller_sha
        !=
        EXPECTED_CONTROLLER_SHA
    ):
        raise RuntimeError(
            "Controller hash mismatch"
        )

    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS unavailable"
        )

    device = torch.device(
        "mps"
    )

    hjtm = HJTMModel(
        HJTM_PATH
    )

    controller = HPCDController(
        hjtm
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_PATH,
            local_files_only=True,
        )
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

    print(
        "device:",
        device
    )

    print(
        "dtype:",
        next(
            model.parameters()
        ).dtype
    )

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
                USER_PROMPT,
        },
    ]

    chat_encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    # Transformers-version compatibility:
    # apply_chat_template may return a Tensor,
    # BatchEncoding, dict-like object, or list.
    if isinstance(
        chat_encoded,
        torch.Tensor,
    ):
        prompt_ids = chat_encoded

    elif hasattr(
        chat_encoded,
        "input_ids",
    ):
        prompt_ids = chat_encoded.input_ids

    elif isinstance(
        chat_encoded,
        dict,
    ):
        prompt_ids = chat_encoded[
            "input_ids"
        ]

    else:
        prompt_ids = torch.tensor(
            chat_encoded,
            dtype=torch.long,
        )

        if prompt_ids.ndim == 1:
            prompt_ids = (
                prompt_ids
                .unsqueeze(
                    0
                )
            )

    prompt_ids = prompt_ids.to(
        device
    )

    if prompt_ids.ndim != 2:
        raise RuntimeError(
            f"Unexpected prompt_ids shape: "
            f"{tuple(prompt_ids.shape)}"
        )

    print(
        "chat-template return type:",
        type(
            chat_encoded
        ).__name__
    )

    print(
        "prompt_ids shape:",
        tuple(
            prompt_ids.shape
        )
    )

    print(
        "prompt tokens:",
        int(
            prompt_ids.shape[
                1
            ]
        )
    )

    print(
        "source:",
        SOURCE_TEXT
    )

    # --------------------------------------------------
    # VANILLA
    #
    # We intentionally use the same controller call with
    # beta=0. This gives the exact frozen temperature /
    # top-p implementation that the controlled decoder
    # will use.
    # --------------------------------------------------

    vanilla = run_generation(
        model=model,
        tokenizer=tokenizer,
        controller=controller,
        hjtm=hjtm,
        device=device,
        prompt_ids=prompt_ids,
        label="VANILLA",
        order=2,
        beta=0.0,
        mode="typicality",
    )

    # Independent repeat with identical seed.
    beta0 = run_generation(
        model=model,
        tokenizer=tokenizer,
        controller=controller,
        hjtm=hjtm,
        device=device,
        prompt_ids=prompt_ids,
        label="HPCD BETA=0 REGRESSION",
        order=2,
        beta=0.0,
        mode="typicality",
    )

    print()
    print(
        "===== BETA=0 SEQUENCE REGRESSION ====="
    )

    token_match = (
        vanilla[
            "generated_token_ids"
        ]
        ==
        beta0[
            "generated_token_ids"
        ]
    )

    text_match = (
        vanilla[
            "text"
        ]
        ==
        beta0[
            "text"
        ]
    )

    print(
        "token IDs identical:",
        token_match
    )

    print(
        "decoded text identical:",
        text_match
    )

    if not token_match:
        raise RuntimeError(
            "beta=0 token sequence "
            "does not reproduce Vanilla"
        )

    if not text_match:
        raise RuntimeError(
            "beta=0 decoded text "
            "does not reproduce Vanilla"
        )

    likelihood = run_generation(
        model=model,
        tokenizer=tokenizer,
        controller=controller,
        hjtm=hjtm,
        device=device,
        prompt_ids=prompt_ids,
        label="LIKELIHOOD HPCD n=2 beta=0.25",
        order=2,
        beta=0.25,
        mode="likelihood",
    )

    typicality = run_generation(
        model=model,
        tokenizer=tokenizer,
        controller=controller,
        hjtm=hjtm,
        device=device,
        prompt_ids=prompt_ids,
        label="TYPICALITY HPCD n=2 beta=0.25",
        order=2,
        beta=0.25,
        mode="typicality",
    )

    for rec in (
        vanilla,
        beta0,
        likelihood,
        typicality,
    ):

        if not rec[
            "generated_token_ids"
        ]:
            raise RuntimeError(
                "Empty generation"
            )

        if not math.isfinite(
            rec[
                "trajectory_nll"
            ]
        ):
            raise RuntimeError(
                "Non-finite trajectory NLL"
            )

        if not math.isfinite(
            rec[
                "trajectory_typicality"
            ]
        ):
            raise RuntimeError(
                "Non-finite typicality"
            )

    result = {
        "status":
            "PASS",

        "stage":
            "T2A-2 autoregressive generation smoke",

        "scientific_status":
            (
                "IMPLEMENTATION TEST ONLY; "
                "NO ATTACK CLAIMS ALLOWED"
            ),

        "source_text":
            SOURCE_TEXT,

        "system_prompt":
            SYSTEM_PROMPT,

        "user_prompt":
            USER_PROMPT,

        "domain":
            DOMAIN,

        "seed":
            SEED,

        "temperature":
            TEMPERATURE,

        "top_p":
            TOP_P,

        "max_new_tokens":
            MAX_NEW_TOKENS,

        "HJTM_SHA256":
            hjtm_sha,

        "controller_SHA256":
            controller_sha,

        "beta0_token_exact_match":
            token_match,

        "beta0_text_exact_match":
            text_match,

        "generations": [
            vanilla,
            beta0,
            likelihood,
            typicality,
        ],
    }

    OUT.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
        +
        "\n",
        encoding="utf-8",
    )

    print()
    print(
        "=" * 100
    )

    print(
        "TRAJECTORY SUMMARY"
    )

    print(
        "=" * 100
    )

    for rec in (
        vanilla,
        likelihood,
        typicality,
    ):

        print(
            rec[
                "label"
            ]
        )

        print(
            "  tokens:",
            rec[
                "generated_tokens"
            ]
        )

        print(
            "  NLL:",
            round(
                rec[
                    "trajectory_nll"
                ],
                6,
            )
        )

        print(
            "  typicality:",
            round(
                rec[
                    "trajectory_typicality"
                ],
                6,
            )
        )

    print()
    print(
        "===== OUTPUT HASHES ====="
    )

    print(
        "script SHA256:",
        sha256_file(
            Path(
                __file__
            ).resolve()
        )
    )

    print(
        "result SHA256:",
        sha256_file(
            OUT
        )
    )

    print()
    print(
        "=" * 100
    )

    print(
        "HPCD T2A-2 AUTOREGRESSIVE SMOKE: PASS"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()
