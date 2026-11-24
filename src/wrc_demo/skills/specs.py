"""Skill metadata, tool JSON schemas, and small runtime helpers."""

from __future__ import annotations

import cv2
import numpy as np



_MOTION_SKILLS = {
    "grasp_object", "place_at", "place_on_object", "push_object",
    "open_gripper", "close_gripper", "move_home", "pick_and_place",
    "point_at", "wave", "handover", "sort_by_color", "move_relative",
    "throw",
    "move_to_pose",
    "move_to_joint_zero",
    "teach_record",
    "teach_replay",
}


_DESTINATION_FAMILY: dict[str, set[str]] = {
    "box": {"box", "cardboard box", "cardboard tray", "carton",
            "kraft box", "open box", "tray", "bin", "basket",
            "crate", "container"},
    "cardboard box": {"box", "cardboard box", "cardboard tray", "carton",
                      "kraft box", "open box", "tray", "bin", "basket",
                      "crate", "container"},
    "cardboard tray": {"box", "cardboard box", "cardboard tray", "carton",
                       "kraft box", "open box", "tray", "bin", "basket",
                       "crate", "container"},
    "carton": {"box", "cardboard box", "cardboard tray", "carton",
               "kraft box", "open box", "tray", "bin", "basket",
               "crate", "container"},
    "bin": {"box", "bin", "basket", "crate", "container"},
    "basket": {"box", "bin", "basket", "crate", "container"},
    "crate": {"box", "bin", "basket", "crate", "container"},
    "container": {"box", "bin", "basket", "crate", "container"},
}


def _jpeg(rgb: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


def _is_mock(obj: object) -> bool:
    """True if ``obj`` is a ``unittest.mock.Mock`` / ``MagicMock`` instance.

    Duck-typed via the private ``_mock_methods`` attribute that all Mock
    subclasses carry. Used by ``_safe_bridge_move`` to skip its post-move
    FK verify in unit tests where the bridge and arm are Mocks and don't
    simulate actual motion -- the FK comparison would always report the
    arm parked at the initial state and spuriously fail every call.

    Production code paths instantiate ``RebotGraspBridge`` / ``SafeArm``
    / ``MockArm``, none of which carry ``_mock_methods``, so the verify
    runs normally there.
    """
    return hasattr(obj, "_mock_methods")


def _short(args: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in args.items())


TOOL_SPECS: list[dict] = [
    {
        "name": "list_objects",
        "description": "List known objects (visible + remembered) with positions. Call first.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "pick_and_place",
        "description": "Complete pick-and-place: detect+grasp+place atomically. For 'put X in Y', 'pick up X then place Y', etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "object": {"type": "string", "description": "what to pick, e.g. 'banana', 'red cube'"},
                "destination": {"type": "string", "description": "named destination, e.g. 'box', 'plate'. Omit for default drop zone."},
                "material": {"type": "string", "enum": ["rigid", "fragile", "soft", "deformable", "slippery", "heavy"]},
            },
            "required": ["object"],
        },
    },
    {
        "name": "move_home",
        "description": "Park the arm at home pose. Use to reset between tasks or after failures.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "wave",
        "description": (
            "Wave hello: oscillate the forearm at home position to greet. "
            "Use for 'wave hello' / 'say hi' / '打招呼' / '挥手'. No parameters."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "task_done",
        "description": "Declare task finished with success bool and one-paragraph summary. ALWAYS call last.",
        "parameters": {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "summary": {"type": "string"},
            },
            "required": ["success", "summary"],
        },
    },
    {
        "name": "teach_record",
        "description": (
            "WHAT: Enter master-slave teleoperation mode. PiPER Mate master arm "
            "(operator-controlled via /dev/ttyUSB0) drives the reBot B601-RS slave "
            "arm in POS_VEL mode. Capture joint positions at 30Hz. User presses q "
            "to start/stop recording into slot N (Enter also works as fallback). "
            "Long-running (waits for q or Esc). "
            "WHEN: User asks to 'record an action' / 'save a motion' / 'teach mode' / "
            "'record slot N'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "slot": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": "which slot to save the recording into",
                },
            },
            "required": ["slot"],
        },
    },
    {
        "name": "teach_replay",
        "description": (
            "WHAT: Read the saved slot{N}.json and execute the recorded trajectory "
            "via move_joints_traj. After playback, return the arm to motor mechanical "
            "zero (np.zeros(6)) via safe_home. "
            "WHEN: User asks to 'replay the action' / 'play back' / 'repeat it' / "
            "'replay slot N'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "slot": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": "which slot to replay",
                },
            },
            "required": ["slot"],
        },
    },
]

