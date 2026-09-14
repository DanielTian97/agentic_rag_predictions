"""Prediction-guided early-stopping controller from the CIKM 2026 paper.

The controller operates on predicted partial answer quality and predicted partial
utility at each retrieval-reasoning iteration. Thresholds partition each
iteration into the four states described in the paper:

    State 0: high quality, high utility
    State 1: high quality, low utility
    State 2: low quality, high utility
    State 3: low quality, low utility

Stopping policy:
1. Stop immediately on State 0 and return the current answer.
2. On State 3, if a previous State 0 or State 1 exists, stop and return the
   earliest such high-quality answer. The decision cost is still the current
   iteration, because State 3 must be observed before rollback is possible.
3. Otherwise continue to natural termination.
"""

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ControllerDecision:
    """Result of applying the prediction-guided stopping policy.

    ``output_index`` is the iteration whose probed answer should be returned.
    ``decision_index`` is the iteration up to which computation was required.
    They differ for a rollback decision.
    """

    output_index: int
    decision_index: int
    reason: str
    states: tuple[int, ...]


def classify_state(
    predicted_quality: float,
    predicted_utility: float,
    quality_threshold: float,
    utility_threshold: float,
) -> int:
    """Map one pair of predictions to State 0--3.

    Values equal to a threshold are treated as ``high`` (``>=``). This is the
    conventional closed-boundary implementation of the threshold partition.
    """

    high_quality = predicted_quality >= quality_threshold
    high_utility = predicted_utility >= utility_threshold

    if high_quality and high_utility:
        return 0
    if high_quality and not high_utility:
        return 1
    if not high_quality and high_utility:
        return 2
    return 3


def assign_states(
    predicted_quality: Iterable[float],
    predicted_utility: Iterable[float],
    quality_threshold: float,
    utility_threshold: float,
) -> tuple[int, ...]:
    """Assign the four paper states to a trajectory of predictions."""

    quality = tuple(predicted_quality)
    utility = tuple(predicted_utility)
    if len(quality) != len(utility):
        raise ValueError("quality and utility predictions must have equal length")
    if not quality:
        raise ValueError("trajectory must contain at least one iteration")

    return tuple(
        classify_state(p, u, quality_threshold, utility_threshold)
        for p, u in zip(quality, utility)
    )


def apply_joint_controller(
    predicted_quality: Sequence[float],
    predicted_utility: Sequence[float],
    quality_threshold: float,
    utility_threshold: float,
) -> ControllerDecision:
    """Apply the paper's joint quality--utility early-stopping heuristic.

    Indices are zero-based and correspond to trajectory iteration rows. For a
    natural-stop decision, the last iteration is both the output and decision
    point.
    """

    states = assign_states(
        predicted_quality,
        predicted_utility,
        quality_threshold,
        utility_threshold,
    )

    earliest_high_quality = None

    for i, state in enumerate(states):
        # Paper rule 1: State 0 triggers immediate stopping.
        if state == 0:
            return ControllerDecision(
                output_index=i,
                decision_index=i,
                reason="state_0_immediate_stop",
                states=states,
            )

        # State 1 is high-quality but low-utility. It is retained as a rollback
        # candidate, but does not itself trigger stopping in the paper.
        if state == 1 and earliest_high_quality is None:
            earliest_high_quality = i

        # Paper rule 2: State 3 after a previous high-quality state triggers
        # rollback to the earliest previous State 0/1 answer. In a strictly
        # online execution, a previous State 0 would already have stopped under
        # rule 1; retaining the general wording here matches the paper.
        if state == 3 and earliest_high_quality is not None:
            return ControllerDecision(
                output_index=earliest_high_quality,
                decision_index=i,
                reason="state_3_rollback",
                states=states,
            )

    last = len(states) - 1
    return ControllerDecision(
        output_index=last,
        decision_index=last,
        reason="natural_stop",
        states=states,
    )
