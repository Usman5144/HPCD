from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


class HJTMModel:
    def __init__(
        self,
        model_path: str | Path,
    ):
        self.model_path = Path(
            model_path
        )

        obj = json.loads(
            self.model_path.read_text(
                encoding="utf-8"
            )
        )

        if (
            obj.get("model_name")
            !=
            "HJTM_FINAL_V1"
        ):
            raise RuntimeError(
                "Unexpected HJTM model"
            )

        self.obj = obj

        state = obj[
            "state_definition"
        ]

        self.B = int(
            state[
                "B"
            ]
        )

        self.E = int(
            state[
                "E"
            ]
        )

        self.mass_boundaries = np.asarray(
            state[
                "mass_boundaries"
            ],
            dtype=np.float64,
        )

        self.entropy_boundaries = np.asarray(
            state[
                "entropy_boundaries"
            ],
            dtype=np.float64,
        )

        self.alpha = float(
            obj[
                "smoothing"
            ][
                "alpha"
            ]
        )

        self.orders = tuple(
            int(x)
            for x in obj[
                "supported_decoder_orders"
            ]
        )

        if self.orders != (
            0,
            1,
            2,
        ):
            raise RuntimeError(
                f"Unexpected orders "
                f"{self.orders}"
            )

        self.tables = {}

        for order_string, table in obj[
            "true_tables"
        ].items():

            order = int(
                order_string
            )

            parsed = {}

            for key, counts in table.items():

                entropy_string, history_string = (
                    key.split(
                        "|",
                        1,
                    )
                )

                entropy_state = int(
                    entropy_string
                )

                if history_string:
                    history = tuple(
                        int(x)
                        for x in
                        history_string.split(
                            ","
                        )
                    )
                else:
                    history = ()

                parsed[
                    (
                        entropy_state,
                        history,
                    )
                ] = np.asarray(
                    counts,
                    dtype=np.float64,
                )

            self.tables[
                order
            ] = parsed

        self.calibration = obj[
            "human_nll_calibration"
        ]

    def entropy_state(
        self,
        entropy: float,
    ) -> int:

        return int(
            np.searchsorted(
                self.entropy_boundaries,
                float(
                    entropy
                ),
                side="right",
            )
        )

    def mass_state(
        self,
        cumulative_mass: float,
    ) -> int:

        return int(
            np.searchsorted(
                self.mass_boundaries,
                float(
                    cumulative_mass
                ),
                side="right",
            )
        )

    def calibration_for(
        self,
        order: int,
        domain: Optional[str],
    ) -> Tuple[float, float]:

        if (
            domain is not None
            and
            domain
            in
            self.calibration[
                "by_domain"
            ]
        ):
            rec = self.calibration[
                "by_domain"
            ][
                domain
            ][
                str(
                    order
                )
            ]
        else:
            rec = self.calibration[
                "global"
            ][
                str(
                    order
                )
            ]

        mu = float(
            rec[
                "mean"
            ]
        )

        sd = float(
            rec[
                "sd"
            ]
        )

        if (
            not math.isfinite(
                mu
            )
            or
            not math.isfinite(
                sd
            )
            or
            sd <= 0
        ):
            raise RuntimeError(
                "Invalid human NLL "
                "calibration"
            )

        return (
            mu,
            sd,
        )

    def conditional_vector(
        self,
        history: List[int],
        entropy_state: int,
        order: int,
    ) -> np.ndarray:

        order = int(
            order
        )

        if order not in self.orders:
            raise ValueError(
                f"Unsupported order "
                f"{order}"
            )

        effective = min(
            order,
            len(
                history
            ),
        )

        if effective == 0:

            counts = self.tables[
                0
            ].get(
                (
                    int(
                        entropy_state
                    ),
                    (),
                )
            )

            prior = np.full(
                self.B,
                1.0
                /
                self.B,
                dtype=np.float64,
            )

            if counts is None:
                return prior

            total = float(
                counts.sum()
            )

            result = (
                counts
                +
                self.alpha
                *
                prior
            ) / (
                total
                +
                self.alpha
            )

            return (
                result
                /
                result.sum()
            )

        lower = self.conditional_vector(
            history=history,
            entropy_state=entropy_state,
            order=effective - 1,
        )

        hist = tuple(
            history[
                -effective:
            ]
        )

        counts = self.tables[
            effective
        ].get(
            (
                int(
                    entropy_state
                ),
                hist,
            )
        )

        if counts is None:
            return lower

        total = float(
            counts.sum()
        )

        result = (
            counts
            +
            self.alpha
            *
            lower
        ) / (
            total
            +
            self.alpha
        )

        result = (
            result
            /
            result.sum()
        )

        if (
            not np.all(
                np.isfinite(
                    result
                )
            )
            or
            np.any(
                result
                <=
                0
            )
        ):
            raise RuntimeError(
                "Invalid HJTM conditional"
            )

        return result


class HPCDController:
    def __init__(
        self,
        hjtm: HJTMModel,
    ):
        self.hjtm = hjtm

    @staticmethod
    def _base_distribution(
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: Optional[int],
    ) -> Dict:

        if logits.ndim != 1:
            raise ValueError(
                "Expected 1D logits"
            )

        if (
            not math.isfinite(
                temperature
            )
            or
            temperature <= 0
        ):
            raise ValueError(
                "temperature must be >0"
            )

        if (
            not 0
            <
            top_p
            <=
            1
        ):
            raise ValueError(
                "top_p must be in "
                "(0,1]"
            )

        z = (
            logits
            .float()
            /
            float(
                temperature
            )
        )

        log_probs = torch.log_softmax(
            z,
            dim=-1,
        )

        probs = torch.exp(
            log_probs
        )

        if not torch.all(
            torch.isfinite(
                probs
            )
        ):
            raise RuntimeError(
                "Non-finite base "
                "probabilities"
            )

        entropy = -(
            probs
            *
            log_probs
        ).sum()

        sorted_probs, sorted_ids = (
            torch.sort(
                probs,
                descending=True,
            )
        )

        sorted_logits = z[
            sorted_ids
        ]

        cumulative = torch.cumsum(
            sorted_probs,
            dim=0,
        )

        # Exact implementation of:
        #
        # u(v) = sum_{w:p(w)>=p(v)} p(w)
        #
        # Equal-logit tokens therefore receive
        # the cumulative mass at the END of
        # their tie group.
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

        mass_position_sorted = (
            torch.repeat_interleave(
                tie_mass,
                tie_counts,
            )
        )

        previous_cumulative = (
            cumulative
            -
            sorted_probs
        )

        keep = (
            previous_cumulative
            <
            float(
                top_p
            )
        )

        if top_k is not None:

            top_k = int(
                top_k
            )

            if top_k < 1:
                raise ValueError(
                    "top_k must be >=1"
                )

            rank_positions = torch.arange(
                len(
                    keep
                ),
                device=keep.device,
            )

            keep = (
                keep
                &
                (
                    rank_positions
                    <
                    top_k
                )
            )

        if not bool(
            keep[
                0
            ].item()
        ):
            keep[
                0
            ] = True

        support_ids = sorted_ids[
            keep
        ]

        support_probs = probs[
            support_ids
        ]

        support_probs = (
            support_probs
            /
            support_probs.sum()
        )

        support_mass_positions = (
            mass_position_sorted[
                keep
            ]
        )

        base_full = torch.zeros_like(
            probs
        )

        base_full[
            support_ids
        ] = support_probs

        return {
            "scaled_logits":
                z,

            "full_probs":
                probs,

            "entropy":
                float(
                    entropy.item()
                ),

            "support_ids":
                support_ids,

            "support_base_probs":
                support_probs,

            "support_mass_positions":
                support_mass_positions,

            "base_full":
                base_full,

            "support_size":
                int(
                    support_ids.numel()
                ),
        }

    @staticmethod
    def _stable_support_softmax(
        log_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Numerically stable softmax restricted to an already-frozen
        candidate support.

        In float32, extremely strong control penalties can make valid
        support probabilities underflow to exact zero. That changes the
        numerical support even though the mathematical exponential-family
        solution has strictly positive probability everywhere p(v) > 0.

        We therefore floor only INSIDE the frozen support at the smallest
        positive normal float32 value and renormalize. Tokens outside the
        support are handled separately and remain exactly zero.
        """

        if log_weight.ndim != 1:
            raise ValueError(
                "Expected 1D support log-weights"
            )

        if not torch.all(
            torch.isfinite(
                log_weight
            )
        ):
            raise RuntimeError(
                "Non-finite HPCD support log-weights"
            )

        probs = torch.softmax(
            log_weight.float(),
            dim=0,
        )

        tiny = torch.finfo(
            probs.dtype
        ).tiny

        probs = torch.clamp(
            probs,
            min=tiny,
        )

        probs = (
            probs
            /
            probs.sum()
        )

        if not torch.all(
            torch.isfinite(
                probs
            )
        ):
            raise RuntimeError(
                "Non-finite stable support probabilities"
            )

        if bool(
            torch.any(
                probs <= 0
            ).item()
        ):
            raise RuntimeError(
                "Stable support softmax produced zero probability"
            )

        return probs

    def control(
        self,
        logits: torch.Tensor,
        history: List[int],
        cumulative_cost: float,
        generated_count: int,
        order: int,
        beta: float,
        domain: Optional[str],
        mode: str,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: Optional[int] = None,
    ) -> Dict:

        if mode not in (
            "likelihood",
            "typicality",
        ):
            raise ValueError(
                "mode must be "
                "'likelihood' or "
                "'typicality'"
            )

        if (
            not math.isfinite(
                beta
            )
            or
            beta < 0
        ):
            raise ValueError(
                "beta must be >=0"
            )

        if generated_count != len(
            history
        ):
            raise ValueError(
                "generated_count/history "
                "length mismatch"
            )

        base = self._base_distribution(
            logits=logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        entropy_state = (
            self.hjtm.entropy_state(
                base[
                    "entropy"
                ]
            )
        )

        mass_np = (
            base[
                "support_mass_positions"
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
                state_np
                <
                0
            )
            or
            np.any(
                state_np
                >=
                self.hjtm.B
            )
        ):
            raise RuntimeError(
                "Invalid candidate "
                "mass state"
            )

        ph = (
            self.hjtm
            .conditional_vector(
                history=history,
                entropy_state=entropy_state,
                order=order,
            )
        )

        costs = -np.log(
            ph
        )

        token_cost_np = costs[
            state_np
        ]

        token_cost = torch.tensor(
            token_cost_np,
            dtype=(
                base[
                    "support_base_probs"
                ].dtype
            ),
            device=(
                base[
                    "support_base_probs"
                ].device
            ),
        )

        support_base = base[
            "support_base_probs"
        ]

        if beta == 0:

            support_controlled = (
                support_base.clone()
            )

            penalty = torch.zeros_like(
                support_base
            )

            target_mu = None
            target_sd = None

        elif mode == "likelihood":

            log_weight = (
                torch.log(
                    support_base
                )
                -
                float(
                    beta
                )
                *
                token_cost
            )

            support_controlled = (
                self._stable_support_softmax(
                    log_weight
                )
            )

            penalty = token_cost

            target_mu = None
            target_sd = None

        else:

            target_mu, target_sd = (
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
                generated_count
                +
                1
            )

            z = (
                projected_nll
                -
                float(
                    target_mu
                )
            ) / float(
                target_sd
            )

            penalty = (
                z
                *
                z
            )

            log_weight = (
                torch.log(
                    support_base
                )
                -
                float(
                    beta
                )
                *
                penalty
            )

            support_controlled = (
                self._stable_support_softmax(
                    log_weight
                )
            )

        controlled_full = (
            torch.zeros_like(
                base[
                    "base_full"
                ]
            )
        )

        controlled_full[
            base[
                "support_ids"
            ]
        ] = support_controlled

        if not torch.all(
            torch.isfinite(
                controlled_full
            )
        ):
            raise RuntimeError(
                "Non-finite HPCD "
                "distribution"
            )

        controlled_sum = float(
            controlled_full.sum().item()
        )

        if abs(
            controlled_sum
            -
            1.0
        ) > 1e-5:
            raise RuntimeError(
                f"HPCD normalization "
                f"failed: {controlled_sum}"
            )

        outside_support = (
            controlled_full
            >
            0
        ) & (
            base[
                "base_full"
            ]
            ==
            0
        )

        if bool(
            torch.any(
                outside_support
            ).item()
        ):
            raise RuntimeError(
                "HPCD escaped base support"
            )

        expected_cost_base = float(
            torch.sum(
                support_base
                *
                token_cost
            ).item()
        )

        expected_cost_controlled = float(
            torch.sum(
                support_controlled
                *
                token_cost
            ).item()
        )

        expected_penalty_base = float(
            torch.sum(
                support_base
                *
                penalty
            ).item()
        )

        expected_penalty_controlled = float(
            torch.sum(
                support_controlled
                *
                penalty
            ).item()
        )

        state_by_token = torch.full(
            (
                logits.numel(),
            ),
            -1,
            dtype=torch.int16,
            device=logits.device,
        )

        state_by_token[
            base[
                "support_ids"
            ]
        ] = torch.tensor(
            state_np,
            dtype=torch.int16,
            device=logits.device,
        )

        cost_by_token = torch.full(
            (
                logits.numel(),
            ),
            float("nan"),
            dtype=torch.float32,
            device=logits.device,
        )

        cost_by_token[
            base[
                "support_ids"
            ]
        ] = token_cost.float()

        return {
            "distribution":
                controlled_full,

            "base_distribution":
                base[
                    "base_full"
                ],

            "support_ids":
                base[
                    "support_ids"
                ],

            "support_size":
                base[
                    "support_size"
                ],

            "entropy":
                base[
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

            "target_mu":
                target_mu,

            "target_sd":
                target_sd,

            "mode":
                mode,

            "order":
                int(
                    order
                ),

            "beta":
                float(
                    beta
                ),
        }

    @staticmethod
    def observe_selected(
        result: Dict,
        token_id: int,
    ) -> Tuple[int, float]:

        token_id = int(
            token_id
        )

        state = int(
            result[
                "state_by_token"
            ][
                token_id
            ].item()
        )

        if state < 0:
            raise RuntimeError(
                "Selected token outside "
                "HPCD support"
            )

        cost = float(
            result[
                "cost_by_token"
            ][
                token_id
            ].item()
        )

        if not math.isfinite(
            cost
        ):
            raise RuntimeError(
                "Invalid selected-token "
                "HJTM cost"
            )

        return (
            state,
            cost,
        )
