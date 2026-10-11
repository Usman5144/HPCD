from __future__ import annotations

import os

import hashlib
import json
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


ROOT = Path(os.environ.get("HPCD_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))

HJTM_PATH = (
    ROOT
    /
    "results/hjtm/final_hjtm_v1/"
    "hjtm_final_v1.json"
)

MODEL_PATH = Path(
    "/path/to/huggingface/hub/"
    "models--Qwen--Qwen2.5-3B-Instruct/"
    "snapshots/"
    "aa8e72537993ba99e69dfaafa59ed015b17504d1"
)

OUT = (
    ROOT
    /
    "results/hpcd/"
    "t2a_controller_unit_v1.json"
)

EXPECTED_HJTM_SHA = (
    "1abd60c38a8787c4d44cf4e81ec157ea"
    "f18745ce1dec28c366417030b1a22c61"
)


def sha256_file(path):
    h = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:
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


def main():

    print(
        "=" * 100
    )

    print(
        "HPCD T2A-1 — CONTROLLER UNIT TEST"
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

    print(
        "HJTM SHA256:",
        hjtm_sha,
    )

    if (
        hjtm_sha
        !=
        EXPECTED_HJTM_SHA
    ):
        raise RuntimeError(
            "HJTM checksum mismatch"
        )

    hjtm = HJTMModel(
        HJTM_PATH
    )

    controller = HPCDController(
        hjtm
    )

    print(
        "HJTM B:",
        hjtm.B,
    )

    print(
        "HJTM E:",
        hjtm.E,
    )

    print(
        "orders:",
        hjtm.orders,
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
        device,
    )

    print(
        "dtype:",
        next(
            model.parameters()
        ).dtype,
    )

    tests = [
        {
            "domain":
                "news",

            "text":
                (
                    "Rewrite the following news "
                    "passage while preserving all "
                    "facts and attribution:\n"
                    "The government announced a "
                    "new transport policy on Monday."
                ),
        },
        {
            "domain":
                "reddit",

            "text":
                (
                    "Rewrite naturally while "
                    "preserving the meaning:\n"
                    "I bought this laptop last "
                    "week and the battery keeps "
                    "draining overnight."
                ),
        },
        {
            "domain":
                "reviews",

            "text":
                (
                    "Rewrite this review without "
                    "adding or removing claims:\n"
                    "The performances are strong, "
                    "but the final act feels "
                    "unnecessarily long."
                ),
        },
    ]

    records = []

    history_seed = [
        2,
        3,
    ]

    cumulative_cost_seed = 2.0 * 1.59

    for case_index, case in enumerate(
        tests,
        1,
    ):

        encoded = tokenizer(
            case[
                "text"
            ],
            return_tensors="pt",
            add_special_tokens=True,
        )

        encoded = {
            k:
                v.to(
                    device
                )
            for k, v
            in encoded.items()
        }

        with torch.inference_mode():

            logits = model(
                **encoded,
                use_cache=False,
            ).logits[
                0,
                -1,
                :
            ]

        print()
        print(
            f"===== CASE {case_index}: "
            f"{case['domain']} ====="
        )

        case_record = {
            "case":
                case_index,

            "domain":
                case[
                    "domain"
                ],

            "orders":
                [],
        }

        for order in (
            0,
            1,
            2,
        ):

            history = (
                []
                if order == 0
                else
                history_seed[
                    -order:
                ]
            )

            generated_count = len(
                history
            )

            cumulative_cost = (
                0.0
                if generated_count == 0
                else
                cumulative_cost_seed
                *
                (
                    generated_count
                    /
                    2.0
                )
            )

            beta0 = controller.control(
                logits=logits,
                history=history,
                cumulative_cost=cumulative_cost,
                generated_count=generated_count,
                order=order,
                beta=0.0,
                domain=case[
                    "domain"
                ],
                mode="typicality",
                temperature=1.0,
                top_p=0.95,
                top_k=None,
            )

            base = beta0[
                "base_distribution"
            ]

            q0 = beta0[
                "distribution"
            ]

            beta0_max_error = float(
                torch.max(
                    torch.abs(
                        base
                        -
                        q0
                    )
                ).item()
            )

            if beta0_max_error != 0.0:
                raise RuntimeError(
                    f"beta=0 regression failed: "
                    f"{beta0_max_error}"
                )

            if float(
                q0.sum().item()
            ) != float(
                base.sum().item()
            ):
                raise RuntimeError(
                    "beta=0 sum mismatch"
                )

            likelihood = controller.control(
                logits=logits,
                history=history,
                cumulative_cost=cumulative_cost,
                generated_count=generated_count,
                order=order,
                beta=0.5,
                domain=case[
                    "domain"
                ],
                mode="likelihood",
                temperature=1.0,
                top_p=0.95,
                top_k=None,
            )

            typicality = controller.control(
                logits=logits,
                history=history,
                cumulative_cost=cumulative_cost,
                generated_count=generated_count,
                order=order,
                beta=0.5,
                domain=case[
                    "domain"
                ],
                mode="typicality",
                temperature=1.0,
                top_p=0.95,
                top_k=None,
            )

            if (
                likelihood[
                    "expected_cost_controlled"
                ]
                >
                likelihood[
                    "expected_cost_base"
                ]
                +
                1e-6
            ):
                raise RuntimeError(
                    "Likelihood control failed "
                    "to lower expected cost"
                )

            if (
                typicality[
                    "expected_penalty_controlled"
                ]
                >
                typicality[
                    "expected_penalty_base"
                ]
                +
                1e-6
            ):
                raise RuntimeError(
                    "Typicality control failed "
                    "to lower expected penalty"
                )

            base_support = (
                base
                >
                0
            )

            like_support = (
                likelihood[
                    "distribution"
                ]
                >
                0
            )

            typ_support = (
                typicality[
                    "distribution"
                ]
                >
                0
            )

            if not torch.equal(
                base_support,
                like_support,
            ):
                raise RuntimeError(
                    "Likelihood support changed"
                )

            if not torch.equal(
                base_support,
                typ_support,
            ):
                raise RuntimeError(
                    "Typicality support changed"
                )

            for name, result in (
                (
                    "likelihood",
                    likelihood,
                ),
                (
                    "typicality",
                    typicality,
                ),
            ):

                q = result[
                    "distribution"
                ]

                if not torch.all(
                    torch.isfinite(
                        q
                    )
                ):
                    raise RuntimeError(
                        f"{name}: nonfinite q"
                    )

                q_sum = float(
                    q.sum().item()
                )

                if abs(
                    q_sum
                    -
                    1.0
                ) > 1e-5:
                    raise RuntimeError(
                        f"{name}: q sum "
                        f"{q_sum}"
                    )

            cpu_q = (
                typicality[
                    "distribution"
                ]
                .detach()
                .cpu()
            )

            g1 = torch.Generator(
                device="cpu"
            )

            g1.manual_seed(
                2027
            )

            a = int(
                torch.multinomial(
                    cpu_q,
                    1,
                    generator=g1,
                ).item()
            )

            g2 = torch.Generator(
                device="cpu"
            )

            g2.manual_seed(
                2027
            )

            b = int(
                torch.multinomial(
                    cpu_q,
                    1,
                    generator=g2,
                ).item()
            )

            if a != b:
                raise RuntimeError(
                    "Seeded sampling "
                    "not deterministic"
                )

            selected_state, selected_cost = (
                controller.observe_selected(
                    typicality,
                    a,
                )
            )

            ph = typicality[
                "human_state_probabilities"
            ]

            if (
                not np.all(
                    np.isfinite(
                        ph
                    )
                )
                or
                np.any(
                    ph <= 0
                )
                or
                abs(
                    float(
                        ph.sum()
                    )
                    -
                    1.0
                )
                >
                1e-10
            ):
                raise RuntimeError(
                    "Invalid HJTM state "
                    "probabilities"
                )

            rec = {
                "order":
                    order,

                "support_size":
                    beta0[
                        "support_size"
                    ],

                "entropy":
                    beta0[
                        "entropy"
                    ],

                "entropy_state":
                    beta0[
                        "entropy_state"
                    ],

                "beta0_max_abs_error":
                    beta0_max_error,

                "likelihood_expected_cost_base":
                    likelihood[
                        "expected_cost_base"
                    ],

                "likelihood_expected_cost_controlled":
                    likelihood[
                        "expected_cost_controlled"
                    ],

                "typicality_expected_penalty_base":
                    typicality[
                        "expected_penalty_base"
                    ],

                "typicality_expected_penalty_controlled":
                    typicality[
                        "expected_penalty_controlled"
                    ],

                "target_mu":
                    typicality[
                        "target_mu"
                    ],

                "target_sd":
                    typicality[
                        "target_sd"
                    ],

                "deterministic_sample_token_id":
                    a,

                "deterministic_sample_text":
                    tokenizer.decode(
                        [
                            a
                        ]
                    ),

                "selected_mass_state":
                    selected_state,

                "selected_hjtm_cost":
                    selected_cost,

                "human_state_probabilities":
                    [
                        float(
                            x
                        )
                        for x in ph
                    ],
            }

            case_record[
                "orders"
            ].append(
                rec
            )

            print(
                f"n={order}",
                "support=",
                rec[
                    "support_size"
                ],
                "entropy_state=",
                rec[
                    "entropy_state"
                ],
                "beta0_error=",
                rec[
                    "beta0_max_abs_error"
                ],
            )

            print(
                "  likelihood cost:",
                round(
                    rec[
                        "likelihood_expected_cost_base"
                    ],
                    6,
                ),
                "->",
                round(
                    rec[
                        "likelihood_expected_cost_controlled"
                    ],
                    6,
                ),
            )

            print(
                "  typicality penalty:",
                round(
                    rec[
                        "typicality_expected_penalty_base"
                    ],
                    6,
                ),
                "->",
                round(
                    rec[
                        "typicality_expected_penalty_controlled"
                    ],
                    6,
                ),
            )

            print(
                "  HJTM probs:",
                [
                    round(
                        float(
                            x
                        ),
                        4,
                    )
                    for x in ph
                ],
            )

            print(
                "  deterministic token:",
                a,
                repr(
                    tokenizer.decode(
                        [
                            a
                        ]
                    )
                ),
                "state=",
                selected_state,
                "cost=",
                round(
                    selected_cost,
                    6,
                ),
            )

        records.append(
            case_record
        )

        del logits
        del encoded

        torch.mps.empty_cache()

    result = {
        "status":
            "PASS",

        "stage":
            "HPCD T2A-1 controller unit test",

        "HJTM_SHA256":
            hjtm_sha,

        "model":
            str(
                MODEL_PATH
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

        "temperature":
            1.0,

        "top_p":
            0.95,

        "beta_test":
            0.5,

        "orders":
            [
                0,
                1,
                2,
            ],

        "modes":
            [
                "likelihood",
                "typicality",
            ],

        "records":
            records,
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
        "===== OUTPUT HASHES ====="
    )

    print(
        "controller SHA256:",
        sha256_file(
            ROOT
            /
            "code/hpcd/"
            "hpcd_controller_v1.py"
        ),
    )

    print(
        "test SHA256:",
        sha256_file(
            Path(
                __file__
            ).resolve()
        ),
    )

    print(
        "result SHA256:",
        sha256_file(
            OUT
        ),
    )

    print()
    print(
        "=" * 100
    )

    print(
        "HPCD T2A-1 CONTROLLER UNIT TEST: PASS"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()
