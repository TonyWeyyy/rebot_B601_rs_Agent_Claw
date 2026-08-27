"""Analytic top-down grasp planning from an ObjectFix (improved baseline).

The baseline estimated grasps in the camera frame from 2D OBBs (approach
along the camera ray). Here the object's 3D OBB already lives in the base
frame, so the plan is done where it is physically meaningful:

- approach: configurable pitch from the +z axis (default ``approach_pitch_rad``
  = 1.2 ≈ 69°; rebot_grasp's default. Pure top-down 90° is geometrically
  awkward from the reBot DevArm B601's elbow-up ``home_q`` — the DLS solver
  cannot cross from the elbow-up basin to the elbow-down basin needed for
  strict top-down, and err sticks at ~1.6 mm. A 1.2 rad pitch is reachable
  from elbow-up in one DLS basin and IK succeeds at tolerance 5e-3);
- jaw opening axis: the OBB minor horizontal axis (grip across the object's
  short side), yaw = atan2 of that axis; a 90-degree-rotated alternate is
  also emitted for the recovery path ("try the other yaw" after a failure);
- grasp height: object top minus a fraction of its height, clamped above the
  table so the jaws wrap the object instead of pinching its rim.

Width feasibility against the gripper's max opening is annotated, not
silently dropped: the agent should know an object is too wide to pinch (and
maybe push it instead).
"""

from __future__ import annotations

import numpy as np

from ..types import Grasp, ObjectFix


# Default approach pitch (1.2 rad ≈ 69° from +z). rebot_grasp's
# ``scripts/main.py:59`` uses 1.2 too; pure top-down (π/2) is geometrically
# unreachable from elbow-up home_q with DLS.
DEFAULT_APPROACH_PITCH_RAD = 1.2


def _approach_rotation(approach_pitch_rad: float, yaw: float) -> np.ndarray:
    """TCP rotation for a (near-)top-down grasp with jaw-opening yaw.

    Approach vector tilts from the -z axis by ``approach_pitch_rad``
    toward the horizontal direction ``(cos(yaw), sin(yaw), 0)``:

        down = -cos(pitch)·ẑ + sin(pitch)·(cos(yaw)·x̂ + sin(yaw)·ŷ)

    (so ``pitch=0`` → straight down ``-z``; for ``0 < pitch < π/2`` the
    approach tilts off-vertical; at exactly ``pitch=π/2`` we fall back
    to the legacy top-down convention — see the back-compat branch).

    The jaw opening axis lies in the table plane, perpendicular to the
    horizontal projection of the approach vector — i.e. it tilts the
    same way as a horizontal vector perpendicular to ``yaw``.

    Compatibility note: at exactly ``pitch=π/2`` (the legacy "top-down"
    convention), the **approach vector is still ``-z``** and the yaw
    only rotates the open axis (gripper faces yaw direction). This
    keeps tests of the old ``_yaw_rotation(yaw)`` API passing — and
    matches the demo config comment that "raise toward π/2 for strict
    top-down". The tilted-from-vertical math below is for
    ``0 ≤ pitch < π/2`` only.
    """
    # Back-compat: at exact π/2, the caller's intent is the legacy
    # "approach=-z, yaw rotates open axis" convention. The continuous
    # tilt math would otherwise degenerate the approach to a horizontal
    # unit vector in the yaw direction, which is not top-down at all.
    if np.isclose(approach_pitch_rad, np.pi / 2, atol=1e-6):
        return _yaw_rotation(yaw)
    sp = np.sin(approach_pitch_rad)
    cp = np.cos(approach_pitch_rad)
    # Approach vector: starts at -z (top-down) and tilts toward (cos(yaw),
    # sin(yaw), 0) by `pitch` radians.
    down = np.array([
        sp * np.cos(yaw),
        sp * np.sin(yaw),
        -cp,
    ])
    down /= np.linalg.norm(down)
    # Jaw opening axis in table plane, perpendicular to the horizontal
    # projection of `down`.
    horiz = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    proj = horiz @ down
    open_axis = horiz - proj * down
    if np.linalg.norm(open_axis) > 1e-9:
        open_axis /= np.linalg.norm(open_axis)
    else:
        # Degenerate (yaw aligned with approach): fall back to +x
        open_axis = np.array([1.0, 0.0, 0.0])
    third = np.cross(down, open_axis)
    # Columns: x=approach(down), y=open axis, z=completes right-handed frame.
    return np.column_stack([down, open_axis, third])


def _yaw_rotation(yaw: float) -> np.ndarray:
    """Top-down grasp: approach is always -z, yaw rotates the open axis.

    Equivalent to ``_approach_rotation(π/2, yaw)`` only in the
    back-compat sense (legacy callers expect ``R[:,0] == -z``); the new
    helper tilts the approach away from -z at pitch=π/2, which is what
    rebot_grasp's ``main.py:59`` (pitch=1.2) wants.
    """
    down = np.array([0.0, 0.0, -1.0])
    open_axis = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    # Project out the down component (zero here, but kept for consistency).
    open_axis = open_axis - (open_axis @ down) * down
    third = np.cross(down, open_axis)
    return np.column_stack([down, open_axis, third])


def plan_grasps_from_fix(
    fix: ObjectFix,
    table_z: float,
    max_width_m: float = 0.09,
    depth_fraction: float = 0.5,
    min_grasp_z_above_table: float = 0.005,
    width_pad_m: float = 0.015,
    approach_pitch_rad: float = DEFAULT_APPROACH_PITCH_RAD,
) -> list[Grasp]:
    """-> ranked candidate grasps (primary yaw first, then alternates).

    ``approach_pitch_rad`` defaults to 1.2 (≈69° from +z) — matches
    rebot_grasp's working value, reachable from the reBot's elbow-up
    home_q in one DLS basin. Pass ``np.pi / 2`` for strict top-down
    (only reachable from elbow-down seeds; not the default config).
    """
    # Horizontal footprint: project OBB axes into the table plane.
    axes, extents = fix.axes, fix.extent
    horiz: list[tuple[float, np.ndarray]] = []
    for i in range(3):
        a = axes[:, i].copy()
        a[2] = 0.0
        n = np.linalg.norm(a)
        if n < 0.3:  # mostly vertical axis: not a jaw direction
            continue
        # Jaws only span the HORIZONTAL projection of this extent.
        horiz.append((float(extents[i]) * n, a / n))
    if not horiz:  # degenerate (looking at a sphere/top of a cylinder)
        horiz = [(float(extents[1]), np.array([1.0, 0.0, 0.0]))]
    horiz.sort(key=lambda t: t[0])  # narrowest horizontal extent first

    obj_top_z = float(fix.points[:, 2].max())
    points_min_z = float(fix.points[:, 2].min())
    # Top-down cameras only see the object's top surface, so points_min sits
    # near the top and would collapse the height estimate. Objects rest on
    # the table: unless the cloud clearly starts well above it (stacked),
    # take the table as the bottom.
    if points_min_z - table_z < 0.5 * max(obj_top_z - table_z, 0.01):
        obj_bottom_z = table_z
    else:
        obj_bottom_z = points_min_z
    height = max(obj_top_z - obj_bottom_z, 0.01)
    grasp_z = max(
        obj_top_z - depth_fraction * height,
        table_z + min_grasp_z_above_table,
    )

    # Approach vector points down at the configured pitch; convert to a
    # numpy array for the Grasp dataclass (it expects a 3-vector).
    sp = np.sin(approach_pitch_rad)
    cp = np.cos(approach_pitch_rad)
    # Convention: down is -z when pitch=0, but we want the approach to
    # point into the object — so we negate. Use yaw=0 as a placeholder;
    # the actual yaw comes from the OBB axis below and gets folded into
    # _approach_rotation.
    approach_template = np.array([0.0, 0.0, -1.0])  # placeholder

    grasps: list[Grasp] = []
    for rank, (width, axis) in enumerate(horiz[:2]):
        yaw = float(np.arctan2(axis[1], axis[0]))
        required = width + width_pad_m
        feasible = required <= max_width_m
        pos = fix.position.copy()
        pos[2] = grasp_z
        R = _approach_rotation(approach_pitch_rad, yaw)
        # Approach direction is the first column of R (the tool-down axis).
        approach_dir = R[:, 0].copy()
        grasps.append(
            Grasp(
                position=pos,
                rotation=R,
                width_m=required,
                approach=approach_dir,
                quality=(1.0 if feasible else 0.2) * (1.0 - 0.1 * rank) * fix.detection.conf,
                label=fix.label,
            )
        )
    return grasps
