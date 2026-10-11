from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import torch

from hpcd_controller_v1 import (
    HJTMModel,
    HPCDController,
)


class ReferenceAlignedHPCDController(
    HPCDController
):
    """
    Final reference-aligned HPCD controller.

    Base/prompt logits:
        define p_t(v), temperature and safe support.

    Standalone generated-prefix logits:
        define the probability landscape used to map
        candidate tokens to HJTM human-process states.

    The reference landscape ALWAYS uses raw temperature=1
    logits because that is how HJTM was trained.
    """

    @staticmethod
    def _reference_landscape(
        reference_logits: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> Dict:

        if reference_logits.ndim != 1:
            raise ValueError(
                "reference_logits must be 1D"
            )

        z = reference_logits.float()

        logp = torch.log_softmax(
            z,
            dim=-1,
        )

        p = torch.exp(
            logp
        )

        entropy = -(
            p
            *
            logp
        ).sum()

        sorted_logits, sorted_ids = (
            torch.sort(
                z,
                descending=True,
            )
        )

        sorted_probs = p[
            sorted_ids
        ]

        cumulative = torch.cumsum(
            sorted_probs,
            dim=0,
        )

        # Exact >= definition with tied logits.
        _, tie_counts = (
            torch.unique_consecutive(
                sorted_logits,
                return_counts=True,
            )
        )

        tie_ends = (
            torch.cumsum(
                tie_counts,
                dim=0,
            )
            -
            1
        )

        tie_mass = cumulative[
            tie_ends
        ]

        mass_sorted = (
            torch.repeat_interleave(
                tie_mass,
                tie_counts,
            )
        )

        mass_by_token = torch.empty_like(
            p
        )

        mass_by_token[
            sorted_ids
        ] = mass_sorted

        candidate_mass = mass_by_token[
            candidate_ids
        ]

        if not torch.all(
            torch.isfinite(
                candidate_mass
            )
        ):
            raise RuntimeError(
                "Non-finite reference mass positions"
            )

        return {
            "entropy":
                float(
                    entropy.item()
                ),

            "candidate_mass":
                candidate_mass,
        }

    def control_reference_aligned(
        self,
        *,
        base_logits: torch.Tensor,
        reference_logits: torch.Tensor,
        history: List[int],
        cumulative_cost: float,
        order: int,
        beta: float,
        domain: Optional[str],
        mode: str,
        terminal_token_ids: List[int],
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: Optional[int] = None,
    ) -> Dict:

        if mode not in (
            "likelihood",
            "typicality",
        ):
            raise ValueError(
                "Invalid control mode"
            )

        if beta < 0:
            raise ValueError(
                "beta must be >=0"
            )

        base = self._base_distribution(
            logits=base_logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        support_ids = base[
            "support_ids"
        ]

        support_base = base[
            "support_base_probs"
        ]

        terminal_set = {
            int(x)
            for x in terminal_token_ids
            if x is not None
        }

        terminal_mask_cpu = np.asarray(
            [
                int(x) in terminal_set
                for x in (
                    support_ids
                    .detach()
                    .cpu()
                    .tolist()
                )
            ],
            dtype=bool,
        )

        terminal_mask = torch.tensor(
            terminal_mask_cpu,
            dtype=torch.bool,
            device=support_ids.device,
        )

        nonterminal_mask = ~terminal_mask

        nonterminal_ids = support_ids[
            nonterminal_mask
        ]

        if nonterminal_ids.numel() == 0:

            controlled_full = (
                base[
                    "base_full"
                ].clone()
            )

            return {
                "distribution":
                    controlled_full,

                "base_distribution":
                    base[
                        "base_full"
                    ],

                "support_ids":
                    support_ids,

                "support_size":
                    base[
                        "support_size"
                    ],

                "reference_entropy":
                    float("nan"),

                "entropy_state":
                    None,

                "state_by_token":
                    torch.full(
                        (
                            base_logits.numel(),
                        ),
                        -1,
                        dtype=torch.int16,
                        device=base_logits.device,
                    ),

                "cost_by_token":
                    torch.full(
                        (
                            base_logits.numel(),
                        ),
                        float("nan"),
                        dtype=torch.float32,
                        device=base_logits.device,
                    ),

                "expected_cost_base":
                    float("nan"),

                "expected_cost_controlled":
                    float("nan"),

                "expected_penalty_base":
                    float("nan"),

                "expected_penalty_controlled":
                    float("nan"),

                "terminal_probability_base":
                    1.0,

                "terminal_probability_controlled":
                    1.0,
            }

        ref = self._reference_landscape(
            reference_logits=reference_logits,
            candidate_ids=nonterminal_ids,
        )

        entropy_state = (
            self.hjtm.entropy_state(
                ref[
                    "entropy"
                ]
            )
        )

        mass_np = (
            ref[
                "candidate_mass"
            ]
            .detach()
            .cpu()
            .numpy()
        )

        state_np = np.searchsorted(
            self.hjtm.mass_boundaries,
            mass_np,
            side="right",
        ).astype(
            np.int64
        )

        if (
            np.any(
                state_np < 0
            )
            or
            np.any(
                state_np
                >=
                self.hjtm.B
            )
        ):
            raise RuntimeError(
                "Invalid reference HJTM state"
            )

        ph = (
            self.hjtm
            .conditional_vector(
                history=history,
                entropy_state=entropy_state,
                order=order,
            )
        )

        state_costs = -np.log(
            ph
        )

        token_cost_np = state_costs[
            state_np
        ]

        token_cost = torch.tensor(
            token_cost_np,
            dtype=torch.float32,
            device=base_logits.device,
        )

        base_nonterminal = support_base[
            nonterminal_mask
        ]

        nonterminal_mass = float(
            base_nonterminal.sum().item()
        )

        terminal_mass = float(
            support_base[
                terminal_mask
            ].sum().item()
        )

        if (
            nonterminal_mass
            <= 0
            or
            nonterminal_mass
            >
            1.00001
        ):
            raise RuntimeError(
                "Invalid nonterminal base mass"
            )

        base_conditional = (
            base_nonterminal
            /
            base_nonterminal.sum()
        )

        if mode == "likelihood":

            penalty = token_cost

        else:

            mu, sd = (
                self.hjtm
                .calibration_for(
                    order=order,
                    domain=domain,
                )
            )

            projected_nll = (
                float(
                    cumulative_cost
                )
                +
                token_cost
            ) / float(
                len(
                    history
                )
                +
                1
            )

            z = (
                projected_nll
                -
                float(
                    mu
                )
            ) / float(
                sd
            )

            penalty = (
                z
                *
                z
            )

        if beta == 0:

            controlled_conditional = (
                base_conditional.clone()
            )

        else:

            log_weight = (
                torch.log(
                    base_conditional
                )
                -
                float(
                    beta
                )
                *
                penalty
            )

            controlled_conditional = (
                self._stable_support_softmax(
                    log_weight
                )
            )

        controlled_nonterminal = (
            controlled_conditional
            *
            nonterminal_mass
        )

        controlled_support = (
            support_base.clone()
        )

        # Preserve terminal probability EXACTLY.
        controlled_support[
            nonterminal_mask
        ] = controlled_nonterminal

        controlled_support[
            terminal_mask
        ] = support_base[
            terminal_mask
        ]

        controlled_support = (
            controlled_support
            /
            controlled_support.sum()
        )

        controlled_full = torch.zeros_like(
            base[
                "base_full"
            ]
        )

        controlled_full[
            support_ids
        ] = controlled_support

        state_by_token = torch.full(
            (
                base_logits.numel(),
            ),
            -1,
            dtype=torch.int16,
            device=base_logits.device,
        )

        state_by_token[
            nonterminal_ids
        ] = torch.tensor(
            state_np,
            dtype=torch.int16,
            device=base_logits.device,
        )

        cost_by_token = torch.full(
            (
                base_logits.numel(),
            ),
            float("nan"),
            dtype=torch.float32,
            device=base_logits.device,
        )

        cost_by_token[
            nonterminal_ids
        ] = token_cost

        expected_cost_base = float(
            torch.sum(
                base_conditional
                *
                token_cost
            ).item()
        )

        expected_cost_controlled = float(
            torch.sum(
                controlled_conditional
                *
                token_cost
            ).item()
        )

        expected_penalty_base = float(
            torch.sum(
                base_conditional
                *
                penalty
            ).item()
        )

        expected_penalty_controlled = float(
            torch.sum(
                controlled_conditional
                *
                penalty
            ).item()
        )

        if not torch.all(
            torch.isfinite(
                controlled_full
            )
        ):
            raise RuntimeError(
                "Non-finite controlled distribution"
            )

        if abs(
            float(
                controlled_full.sum().item()
            )
            -
            1.0
        ) > 1e-5:
            raise RuntimeError(
                "Controlled distribution not normalized"
            )

        controlled_terminal_mass = float(
            controlled_support[
                terminal_mask
            ].sum().item()
        )

        terminal_mass_delta = abs(
            controlled_terminal_mass
            -
            terminal_mass
        )

        # Numerical invariant tolerance.
        #
        # Terminal mass is mathematically preserved by the controller,
        # but reductions/reconstruction in finite-precision arithmetic
        # can differ by a few ULPs. Use a dtype-aware tolerance while
        # retaining a strict invariant against substantive changes.
        terminal_mass_tol = max(
            1e-7,
            8.0
            *
            float(
                torch.finfo(
                    controlled_full.dtype
                ).eps
            )
            *
            max(
                1.0,
                abs(terminal_mass),
                abs(controlled_terminal_mass),
            ),
        )

        if terminal_mass_delta > terminal_mass_tol:
            raise RuntimeError(
                "Terminal probability changed: "
                f"base={terminal_mass:.12g} "
                f"controlled={controlled_terminal_mass:.12g} "
                f"delta={terminal_mass_delta:.12g} "
                f"tol={terminal_mass_tol:.12g} "
                f"dtype={controlled_full.dtype}"
            )

        return {
            "distribution":
                controlled_full,

            "base_distribution":
                base[
                    "base_full"
                ],

            "support_ids":
                support_ids,

            "support_size":
                base[
                    "support_size"
                ],

            "reference_entropy":
                ref[
                    "entropy"
                ],

            "entropy_state":
                entropy_state,

            "human_state_probabilities":
                ph,

            "state_by_token":
                state_by_token,

            "cost_by_token":
                cost_by_token,

            "expected_cost_base":
                expected_cost_base,

            "expected_cost_controlled":
                expected_cost_controlled,

            "expected_penalty_base":
                expected_penalty_base,

            "expected_penalty_controlled":
                expected_penalty_controlled,

            "terminal_probability_base":
                terminal_mass,

            "terminal_probability_controlled":
                controlled_terminal_mass,
        }
