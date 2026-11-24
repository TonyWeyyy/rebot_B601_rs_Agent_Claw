"""Grasp selection: filter candidates by physics, then by reachability.

A grasp is only offered to the executor if BOTH the grasp pose and its
pregrasp pose solve under IK inside the workspace -- the baseline's "IK walk"
idea, kept, but applied to every candidate up front so the agent gets an
honest 'nothing reachable' instead of a mid-motion abort.
"""

from __future__ import annotations

import numpy as np

from ..types import Grasp, SkillError, make_transform


def select_grasp(
    grasps: list[Grasp],
    kin,
    q_current: np.ndarray,
    max_width_m: float = 0.09,
    pregrasp_offset_m: float = 0.12,
    validate=None,
    ik_retries: int = 3,
    preserve_order: bool = False,
) -> tuple[Grasp, np.ndarray, np.ndarray]:
    """-> (grasp, q_pregrasp, q_grasp) for the best executable candidate.

    validate(grasp, q_pre, q_grasp) -> reason-string|None lets the caller
    veto candidates on grounds IK cannot see (safety-harness geometry): a
    candidate that would abort mid-descent must lose the ranking here, not
    kill the attempt later.

    When ``preserve_order=True`` (e.g. candidates were already ranked by
    ``GraspOutcomeMemory.rerank``), keep the caller's ordering instead of
    re-sorting strictly by raw ``g.quality``.
    """
    reasons: list[str] = []
    ordered = list(grasps) if preserve_order else sorted(grasps, key=lambda g: -g.quality)
    for g in ordered:
        if g.width_m > max_width_m:
            reasons.append(
                f"{g.label}: required width {g.width_m * 1000:.0f}mm > gripper "
                f"max {max_width_m * 1000:.0f}mm (consider push or regrasp)"
            )
            continue
        T_grasp = make_transform(g.rotation, g.position)
        T_pre = make_transform(g.rotation, g.pregrasp_position(pregrasp_offset_m))
        pre = kin.ik(T_pre, q_current, retries=ik_retries)
        if not pre.success:
            reasons.append(f"pregrasp IK failed (err {pre.error:.4f})")
            continue
        grasp = kin.ik(T_grasp, pre.q, retries=ik_retries)
        if not grasp.success:
            reasons.append(f"grasp IK failed (err {grasp.error:.4f})")
            continue
        if validate is not None:
            reason = validate(g, pre.q, grasp.q)
            if reason:
                reasons.append(f"{g.label}: {reason}")
                continue
        return g, pre.q, grasp.q
    raise SkillError("no executable grasp: " + "; ".join(reasons[:4]))
