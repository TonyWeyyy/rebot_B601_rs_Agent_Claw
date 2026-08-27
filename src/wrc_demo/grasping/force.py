"""Material-aware grip effort ("how hard to squeeze" challenge).

The gripper is a single RobStride motor in MIT mode: commanded stiffness
(kp scale) bounds the stall torque, so `effort` is our force proxy until
current-loop telemetry is calibrated onsite. Profiles are WIDTH-AWARE
(not frac-of-travel): the gripper closes to ``grasp.width_m * object_close_frac``
of the detected object width, so a 3D-printed banana (30 mm) gets
~22.5 mm of jaw opening instead of the previous fixed 8 mm (= 90% of the
95 mm travel) that crushed it.

The MIT position PD ``tau = kp*(q_target - q) - kd*dq + tau_ff`` means
the motor only pushes until ``q`` reaches ``q_target``: if the object
is wider than the target, the motor stalls at ``q`` ≈ object surface
and the surface limits the force, not the position target. The previous
fixed-frac close pushed past the object with high effort and crushed it.

The material class comes from the VLM advisor when available ("ceramic mug,
fragile") or from a keyword heuristic over the object label as fallback.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GripProfile:
    name: str
    object_close_frac: float  # gripper closes to grasp.width_m * this (0.75 = press to 75%)
    min_close_frac: float  # safety floor (gripper never looser than this)
    max_close_frac: float  # safety ceiling (gripper never tighter than this)
    effort: float  # 0..1.5 tau forward-feed in MIT mode
    lift_speed_scale: float  # slow down lifts for risky grips


# Default safety: 0.30 — gripper closes to 30% of the detected object
# width (so a 30 mm banana gets a 9 mm jaw opening, gripping it at
# ~70% squish). v2 0.75, v3 0.50, v4 0.40 — the user still found
# these too loose on the real rig, so v5 tightens further and cranked
# `effort` to 1.0 (the MIT tau clip cap) so the gripper actually
# develops friction. The ``finger_drop_m`` mask cleaning in
# ``plan_grasp_from_mask`` keeps the depth-tail gripper fingers out
# of the grasp center. Fragile still gets lighter squish so glass
# doesn't shatter; heavy still gets the tightest. min/max act as
# safety bounds if the planner's width estimate is off.
GRIP_PROFILES: dict[str, GripProfile] = {
    "rigid":      GripProfile("rigid",      0.30, 0.30, 0.95, 1.00, 1.0),
    "fragile":    GripProfile("fragile",    0.70, 0.15, 0.75, 0.30, 0.5),  # 30% squish
    "soft":       GripProfile("soft",       0.45, 0.20, 0.90, 0.65, 0.8),
    "deformable": GripProfile("deformable", 0.45, 0.20, 0.90, 0.65, 0.8),
    "slippery":   GripProfile("slippery",   0.30, 0.30, 0.95, 1.00, 0.5),  # banana: max grip
    "heavy":      GripProfile("heavy",      0.25, 0.40, 0.95, 1.00, 0.5),  # squeeze hardest
}

_KEYWORDS = {
    "fragile": ("glass", "egg", "ceramic", "porcelain", "wine", "light bulb"),
    "soft": ("plush", "sponge", "foam", "stuffed", "toy animal", "bread"),
    "deformable": ("bag", "cloth", "towel", "paper cup"),
    "slippery": ("bottle", "can", "metal", "banana", "apple", "orange"),
    "heavy": ("drill", "hammer", "brick"),
}


def select_profile(label: str, material_hint: str | None = None) -> GripProfile:
    if material_hint:
        hint = material_hint.strip().lower()
        if hint in GRIP_PROFILES:
            return GRIP_PROFILES[hint]
        for name in GRIP_PROFILES:
            if name in hint:
                return GRIP_PROFILES[name]
    low = label.lower()
    for name, words in _KEYWORDS.items():
        if any(w in low for w in words):
            return GRIP_PROFILES[name]
    return GRIP_PROFILES["rigid"]


def compute_close_target(
    profile: GripProfile,
    grasp_width_m: float,
    max_width_m: float,
) -> tuple[float, float]:
    """Compute absolute gripper positions for stage 1 (scout) and stage 2 (seat).

    Stage 1: fixed 50% travel — a fast scout to get jaws into contact range.
    Stage 2: width-aware — gripper target opening = grasp.width_m *
    ``profile.object_close_frac`` (default 0.75, user directive 2026-08-10),
    converted to close-fraction (1 - target_opening/max_width) and clipped
    to ``[profile.min_close_frac, profile.max_close_frac]`` of full travel
    as a safety bound.

    Returns ``(pos_stage1, pos_stage2)`` as close-fractions in [0, 1] of
    full travel (0 = open, 1 = closed). The backend multiplies by the
    gripper's ``(closed - open)`` travel to land on an absolute position.
    """
    if max_width_m <= 0:
        return 0.5, 0.5
    # Stage 1: scout at half-travel.
    pos1 = 0.5
    # Stage 2: width-aware target. close_frac is the "fraction of full
    # travel" the gripper should close (0=open, 1=closed). The MIT motor
    # only pushes until q reaches q_target; if the object is wider than
    # (max_width * (1 - close_frac)), the motor stalls at the object
    # surface and the surface limits the force.
    target_opening_m = max(float(grasp_width_m) * profile.object_close_frac, 0.0)
    open_frac = min(target_opening_m / max_width_m, 1.0)  # fraction of full-open
    close_frac = 1.0 - open_frac                          # fraction of full-close
    close_frac = max(profile.min_close_frac, min(profile.max_close_frac, close_frac))
    return pos1, close_frac
