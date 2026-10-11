from __future__ import annotations

import os

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

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
    /
    "results/hjtm/final_hjtm_v1/"
    "hjtm_final_v1.json"
)

OUT = (
    ROOT
    /
    "results/hpcd/"
    "t2a3_reference_aligned_v2.json"
)

SOURCE = (
    "The city council approved a new public transport plan on Monday. "
    "The proposal will add twelve electric buses to the network next year "
    "and increase service frequency during weekday peak hours. "
    "Officials said the plan will be funded through the existing transport budget."
)

SYSTEM = (
    "You rewrite text while preserving its factual meaning exactly. "
    "Preserve named entities, numbers, dates, attribution, negation, modality, "
    "and all factual claims. Do not add claims and do not remove claims. "
    "Write naturally. Output only the rewritten passage."
)

USER = (
    "Rewrite the following passage while preserving all information:\n\n"
    +
    SOURCE
)

SEED = 2027
TEMP = 1.0
TOP_P = 0.95
MAX_NEW = 64
DOMAIN = "news"
ORDER = 2

# Numerical alignment tolerances.
#
# These affect ONLY the Mac FP16/MPS regression test.
# They do NOT change HJTM boundaries, probabilities,
# generation states, or scientific evaluation.
BOUNDARY_AMBIGUITY_TOL = 1e-4
MAX_BOUNDARY_MISMATCH_FRACTION = 0.03
MAX_ONLINE_OFFLINE_NLL_ERROR = 0.01


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def offline_reference_score(
    token_ids,
    *,
    model,
    hjtm,
    device,
):

    eos = (
        tokenizer.eos_token_id
    )

    cleaned = []

    for x in token_ids:

        if (
            eos is not None
            and
            int(x)
            ==
            int(eos)
        ):
            break

        cleaned.append(
            int(x)
        )

    if len(cleaned) < 2:
        raise RuntimeError(
            "Need >=2 nonterminal tokens"
        )

    ids = torch.tensor(
        [
            cleaned
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

    mass = []
    entropy = []

    CHUNK = 16

    for start in range(
        0,
        len(
            targets
        ),
        CHUNK,
    ):

        end = min(
            start
            +
            CHUNK,
            len(
                targets
            ),
        )

        z = logits[
            start:end
        ]

        y = targets[
            start:end
        ]

        selected = (
            z.gather(
                1,
                y.unsqueeze(
                    1
                ),
            )
            .squeeze(
                1
            )
        )

        lp = torch.log_softmax(
            z,
            dim=-1,
        )

        p = torch.exp(
            lp
        )

        u = (
            p
            *
            (
                z
                >=
                selected.unsqueeze(
                    1
                )
            )
        ).sum(
            dim=-1
        )

        H = -(
            p
            *
            lp
        ).sum(
            dim=-1
        )

        mass.extend(
            u.detach()
            .cpu()
            .tolist()
        )

        entropy.extend(
            H.detach()
            .cpu()
            .tolist()
        )

    states = np.searchsorted(
        hjtm.mass_boundaries,
        np.asarray(
            mass
        ),
        side="right",
    ).astype(
        int
    )

    entropy_states = np.searchsorted(
        hjtm.entropy_boundaries,
        np.asarray(
            entropy
        ),
        side="right",
    ).astype(
        int
    )

    history = []

    costs = []

    for b, e in zip(
        states.tolist(),
        entropy_states.tolist(),
    ):

        ph = (
            hjtm
            .conditional_vector(
                history=history,
                entropy_state=int(
                    e
                ),
                order=ORDER,
            )
        )

        c = -math.log(
            float(
                ph[
                    int(
                        b
                    )
                ]
            )
        )

        costs.append(
            c
        )

        history.append(
            int(
                b
            )
        )

    nll = float(
        np.mean(
            costs
        )
    )

    mu, sd = (
        hjtm
        .calibration_for(
            order=ORDER,
            domain=DOMAIN,
        )
    )

    typ = abs(
        (
            nll
            -
            mu
        )
        /
        sd
    )

    return {
        "mass_positions":
            [
                float(x)
                for x in mass
            ],

        "states":
            [
                int(x)
                for x in states
            ],

        "entropy_states":
            [
                int(x)
                for x in entropy_states
            ],

        "costs":
            [
                float(x)
                for x in costs
            ],

        "nll":
            nll,

        "typicality":
            float(
                typ
            ),
    }


def generate(
    label,
    *,
    beta,
    mode,
):

    gen = torch.Generator(
        device="cpu"
    )

    gen.manual_seed(
        SEED
    )

    seq = prompt_ids.clone()

    generated = []

    history = []

    cumulative_cost = 0.0

    scored_states = []

    scored_costs = []

    terminal_ids = []

    if tokenizer.eos_token_id is not None:
        terminal_ids.append(
            int(
                tokenizer.eos_token_id
            )
        )

    support_sizes = []

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

    for step in range(
        MAX_NEW
    ):

        with torch.inference_mode():

            base_logits = model(
                input_ids=seq,
                use_cache=False,
            ).logits[
                0,
                -1,
                :
            ]

        # First generated token is unscored and
        # uncontrolled, matching T1 standalone
        # teacher-forcing where token 1 has no
        # preceding document token.
        if len(
            generated
        ) == 0:

            base = (
                HPCDController
                ._base_distribution(
                    logits=base_logits,
                    temperature=TEMP,
                    top_p=TOP_P,
                    top_k=None,
                )
            )

            q = base[
                "base_full"
            ]

            support_sizes.append(
                int(
                    base[
                        "support_size"
                    ]
                )
            )

            result = None

        else:

            ref_ids = torch.tensor(
                [
                    generated
                ],
                dtype=torch.long,
                device=device,
            )

            with torch.inference_mode():

                reference_logits = model(
                    input_ids=ref_ids,
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
                    order=ORDER,
                    beta=beta,
                    domain=DOMAIN,
                    mode=mode,
                    terminal_token_ids=terminal_ids,
                    temperature=TEMP,
                    top_p=TOP_P,
                    top_k=None,
                )
            )

            q = result[
                "distribution"
            ]

            support_sizes.append(
                int(
                    result[
                        "support_size"
                    ]
                )
            )

        token_id = int(
            torch.multinomial(
                q.detach().cpu(),
                1,
                generator=gen,
            ).item()
        )

        generated.append(
            token_id
        )

        seq = torch.cat(
            [
                seq,
                torch.tensor(
                    [
                        [
                            token_id
                        ]
                    ],
                    dtype=seq.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )

        is_terminal = (
            token_id
            in
            terminal_ids
        )

        if is_terminal:
            break

        if result is not None:

            state = int(
                result[
                    "state_by_token"
                ][
                    token_id
                ].item()
            )

            cost = float(
                result[
                    "cost_by_token"
                ][
                    token_id
                ].item()
            )

            if (
                state < 0
                or
                not math.isfinite(
                    cost
                )
            ):
                raise RuntimeError(
                    "Invalid selected "
                    "reference state/cost"
                )

            history.append(
                state
            )

            cumulative_cost += (
                cost
            )

            scored_states.append(
                state
            )

            scored_costs.append(
                cost
            )

    text = tokenizer.decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    offline = (
        offline_reference_score(
            generated,
            model=model,
            hjtm=hjtm,
            device=device,
        )
    )

    online_nll = float(
        np.mean(
            scored_costs
        )
    )

    mismatch_indices = [
        i
        for i, (
            online_state,
            offline_state,
        )
        in enumerate(
            zip(
                scored_states,
                offline[
                    "states"
                ],
            )
        )
        if (
            online_state
            !=
            offline_state
        )
    ]

    mismatch_fraction = (
        len(
            mismatch_indices
        )
        /
        len(
            scored_states
        )
    )

    boundary_ambiguities = []

    for i in mismatch_indices:

        online_state = int(
            scored_states[
                i
            ]
        )

        offline_state = int(
            offline[
                "states"
            ][
                i
            ]
        )

        offline_u = float(
            offline[
                "mass_positions"
            ][
                i
            ]
        )

        nearest_boundary = min(
            hjtm.mass_boundaries,
            key=lambda x:
                abs(
                    float(x)
                    -
                    offline_u
                ),
        )

        boundary_distance = abs(
            offline_u
            -
            float(
                nearest_boundary
            )
        )

        ambiguity = {
            "index":
                int(
                    i
                ),

            "online_state":
                online_state,

            "offline_state":
                offline_state,

            "offline_u":
                offline_u,

            "nearest_boundary":
                float(
                    nearest_boundary
                ),

            "boundary_distance":
                float(
                    boundary_distance
                ),

            "adjacent":
                (
                    abs(
                        online_state
                        -
                        offline_state
                    )
                    ==
                    1
                ),
        }

        boundary_ambiguities.append(
            ambiguity
        )

        if not ambiguity[
            "adjacent"
        ]:
            raise RuntimeError(
                "Non-adjacent online/offline "
                f"state mismatch: {ambiguity}"
            )

        if (
            boundary_distance
            >
            BOUNDARY_AMBIGUITY_TOL
        ):
            raise RuntimeError(
                "State mismatch not local "
                "to a frozen HJTM boundary: "
                f"{ambiguity}"
            )

    if (
        mismatch_fraction
        >
        MAX_BOUNDARY_MISMATCH_FRACTION
    ):
        raise RuntimeError(
            "Too many online/offline "
            "state mismatches: "
            f"{len(mismatch_indices)}/"
            f"{len(scored_states)} "
            f"({mismatch_fraction:.6f})"
        )

    cost_errors = np.abs(
        np.asarray(
            scored_costs,
            dtype=np.float64,
        )
        -
        np.asarray(
            offline[
                "costs"
            ],
            dtype=np.float64,
        )
    )

    max_cost_error = float(
        np.max(
            cost_errors
        )
    )

    nll_error = abs(
        online_nll
        -
        offline[
            "nll"
        ]
    )

    # With zero discrete-state mismatches, costs
    # should still agree essentially exactly.
    if (
        len(
            mismatch_indices
        )
        ==
        0
        and
        max_cost_error
        >
        1e-6
    ):
        raise RuntimeError(
            "Cost mismatch without any "
            "state mismatch: "
            f"{max_cost_error}"
        )

    # When an FP16 boundary ambiguity occurs,
    # histories can differ from that point onward.
    # Require the resulting document-level HJTM
    # NLL to remain tightly aligned.
    if (
        nll_error
        >
        MAX_ONLINE_OFFLINE_NLL_ERROR
    ):
        raise RuntimeError(
            "Online/offline trajectory NLL "
            "diverged too far: "
            f"{nll_error}"
        )

    print(
        "generated tokens:",
        len(
            generated
        )
    )

    print(
        "scored trajectory tokens:",
        len(
            scored_states
        )
    )

    print(
        "mean prompt support:",
        round(
            float(
                np.mean(
                    support_sizes
                )
            ),
            4,
        )
    )

    print(
        "online NLL:",
        round(
            online_nll,
            6,
        )
    )

    print(
        "offline NLL:",
        round(
            offline[
                "nll"
            ],
            6,
        )
    )

    print(
        "offline typicality:",
        round(
            offline[
                "typicality"
            ],
            6,
        )
    )

    print(
        "exact state matches:",
        (
            len(
                scored_states
            )
            -
            len(
                mismatch_indices
            )
        ),
        "/",
        len(
            scored_states
        ),
    )

    print(
        "boundary-ambiguous states:",
        len(
            mismatch_indices
        )
    )

    print(
        "boundary mismatch fraction:",
        round(
            mismatch_fraction,
            6,
        )
    )

    if boundary_ambiguities:

        print(
            "boundary ambiguities:"
        )

        for ambiguity in (
            boundary_ambiguities
        ):
            print(
                " ",
                ambiguity
            )

    print(
        "max cost error:",
        max_cost_error
    )

    print(
        "online/offline NLL error:",
        nll_error
    )

    print(
        "numerical alignment accepted:",
        True
    )

    print()
    print(
        "TEXT:"
    )

    print(
        text
    )

    return {
        "label":
            label,

        "beta":
            beta,

        "mode":
            mode,

        "token_ids":
            generated,

        "text":
            text,

        "online_states":
            scored_states,

        "online_costs":
            scored_costs,

        "online_nll":
            online_nll,

        "offline_nll":
            offline[
                "nll"
            ],

        "offline_typicality":
            offline[
                "typicality"
            ],

        "exact_state_matches":
            int(
                len(
                    scored_states
                )
                -
                len(
                    mismatch_indices
                )
            ),

        "state_mismatch_count":
            int(
                len(
                    mismatch_indices
                )
            ),

        "state_mismatch_fraction":
            float(
                mismatch_fraction
            ),

        "boundary_ambiguities":
            boundary_ambiguities,

        "max_online_offline_cost_error":
            max_cost_error,

        "online_offline_nll_error":
            nll_error,

        "numerical_alignment_accepted":
            True,

        "mean_prompt_support":
            float(
                np.mean(
                    support_sizes
                )
            ),
    }


print(
    "=" * 100
)

print(
    "HPCD T2A-3 — REFERENCE-ALIGNED "
    "AUTOREGRESSIVE DECODER"
)

print(
    "=" * 100
)

if OUT.exists():
    raise RuntimeError(
        f"Refusing overwrite: {OUT}"
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

controller = (
    ReferenceAlignedHPCDController(
        hjtm
    )
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

messages = [
    {
        "role":
            "system",

        "content":
            SYSTEM,
    },
    {
        "role":
            "user",

        "content":
            USER,
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

print(
    "prompt tokens:",
    prompt_ids.shape[
        1
    ]
)

vanilla = generate(
    "VANILLA",
    beta=0.0,
    mode="typicality",
)

beta0 = generate(
    "HPCD BETA=0 REGRESSION",
    beta=0.0,
    mode="typicality",
)

print()
print(
    "===== BETA=0 HARD REGRESSION ====="
)

token_match = (
    vanilla[
        "token_ids"
    ]
    ==
    beta0[
        "token_ids"
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
        "beta=0 token mismatch"
    )

if not text_match:
    raise RuntimeError(
        "beta=0 text mismatch"
    )

likelihood = generate(
    "REFERENCE-ALIGNED LIKELIHOOD HPCD",
    beta=0.25,
    mode="likelihood",
)

typicality = generate(
    "REFERENCE-ALIGNED TYPICALITY HPCD",
    beta=0.25,
    mode="typicality",
)

result = {
    "status":
        "PASS",

    "stage":
        "T2A-3 reference-aligned decoder",

    "definition":
        {
            "base_landscape":
                (
                    "prompt-conditioned "
                    "paraphraser distribution"
                ),

            "HJTM_landscape":
                (
                    "standalone generated-prefix "
                    "reference distribution"
                ),

            "first_generated_token_controlled":
                False,

            "reference_temperature":
                1.0,

            "terminal_probability_preserved":
                True,
        },

    "seed":
        SEED,

    "temperature":
        TEMP,

    "top_p":
        TOP_P,

    "order":
        ORDER,

    "generations":
        [
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
    "===== HASHES ====="
)

print(
    "reference controller SHA256:",
    sha256_file(
        ROOT
        /
        "code/hpcd/"
        "hpcd_reference_aligned_v2.py"
    )
)

print(
    "test SHA256:",
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
    "HPCD T2A-3 REFERENCE-ALIGNED DECODER: PASS"
)

print(
    "=" * 100
)
