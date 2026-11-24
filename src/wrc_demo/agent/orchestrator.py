"""The agent loop: reflex first, then decompose -> act -> verify -> recover.

Inference-time composition of the two papers' ideas plus the Anthropic
robotics study's latency lesson:
- reflex/habit tier (agent.reflex): routine commands compile straight to
  skill calls (µs) or replay a proven plan from experience memory -- the LLM
  never blocks the hot path;
- ASPIRE: curated skill API + per-call multimodal traces + honest failure
  reporting back into context;
- Agentic-VLA: LLM task decomposition into checkable milestones, and a VLM
  advisor consulted only after failures whose one-sentence spatial suggestion
  is appended to the context for the retry.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from ..skills.runtime import TOOL_SPECS, SkillRuntime
from .advisor import Advisor
from .llm import LLMClient, LLMResponse, ToolCall
from .prompts import DECOMPOSE_PROMPT, SYSTEM_PROMPT
from .reflex import FastPlanner


def _short_args(args: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in args.items())


def _as_bool(value) -> bool:
    """Schema-lax backends send booleans as strings ('false' is truthy!)."""
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return bool(value)


@dataclass
class TaskReport:
    task: str
    success: bool
    summary: str
    steps: int
    milestones: list[str] = field(default_factory=list)
    tool_log: list[dict] = field(default_factory=list)
    path: str = "llm"  # "reflex" | "experience" | "llm"
    duration_s: float = 0.0


class AgentOrchestrator:
    def __init__(
        self,
        llm: LLMClient,
        runtime: SkillRuntime,
        advisor: Advisor | None = None,
        max_steps: int = 30,
        decompose: bool = True,
        attach_images: bool = True,
        fast_planner: FastPlanner | None = None,
        cfg=None,
        fallback_chain: tuple[str, ...] = (),
    ):
        self.llm = llm
        self.runtime = runtime
        self.advisor = advisor
        self.max_steps = max_steps
        self.decompose = decompose
        self.attach_images = attach_images and bool(getattr(llm, "supports_vision", False))
        self.fast_planner = fast_planner
        # ADR-0008: profile switching + fallback. `cfg` is the live
        # wrc_demo.config.Cfg (or any object with a writable `_data["llm"]`
        # dict); `fallback_chain` is an ordered list of profile names tried
        # when the primary chat() raises a transient SDK error. Both default
        # to inert values, so existing callers (demo.py, tests) are
        # backward-compatible.
        self.cfg = cfg
        self.fallback_chain = list(fallback_chain)
        self._want_images = bool(attach_images)

    # ───────────────────────────────────────────────────────────────────
    # ADR-0008: profile switching + transient-error fallback
    # ───────────────────────────────────────────────────────────────────

    def _load_llm_profile_cfg(self, profile_name: str):
        """Load the YAML profile for ``profile_name`` as a ``Cfg``."""
        from ..config import Cfg, load_profile
        try:
            prof = load_profile("llm", profile_name)
            data = prof.as_dict()
            data.setdefault("profile", profile_name)
            return Cfg(data)
        except FileNotFoundError:
            if self.cfg is None:
                raise
            sub = dict(self.cfg._data.get("llm", {}))
            sub["profile"] = profile_name
            return Cfg(sub)

    def switch_llm(self, profile_name: str) -> "LLMClient":
        """Swap the live LLM client for a new one loaded from ``configs/llm/<profile_name>.yaml``.

        The new client becomes ``self.llm``; the next ``chat()`` uses it. An
        ``llm_switch`` event is recorded in the trace.
        """
        from .llm import make_llm
        if self.cfg is None:
            raise RuntimeError(
                "switch_llm requires the orchestrator to be constructed "
                "with a `cfg` argument (ADR-0008)."
            )
        prof_cfg = self._load_llm_profile_cfg(profile_name)
        new_llm = make_llm(prof_cfg)
        self.cfg._data["llm"] = prof_cfg.as_dict()
        self.llm = new_llm
        self.attach_images = self._want_images and bool(getattr(new_llm, "supports_vision", False))
        if getattr(self.runtime, "trace", None) is not None:
            self.runtime.trace.record_event(
                "llm_switch", {"profile": profile_name}
            )
        return new_llm

    @staticmethod
    def _transient_llm_errors() -> tuple[type[BaseException], ...]:
        """Collect transient network/API error classes from installed LLM SDKs."""
        errs: list[type[BaseException]] = [ConnectionError, TimeoutError]
        try:
            import openai
            errs.extend([
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.RateLimitError,
                openai.AuthenticationError,
                openai.InternalServerError,
                openai.BadRequestError,
            ])
        except ImportError:
            pass
        try:
            import anthropic
            errs.extend([
                anthropic.APIConnectionError,
                anthropic.APITimeoutError,
                anthropic.RateLimitError,
                anthropic.AuthenticationError,
                anthropic.InternalServerError,
                anthropic.BadRequestError,
            ])
        except ImportError:
            pass
        return tuple(errs)

    def _chat_with_fallback(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Try `self.llm` first; on a transient SDK error, walk
        `self.fallback_chain` in order. Stops on success or chain
        exhaustion. `LLMConfigError` (raised from `make_llm` itself)
        is NOT caught here — misconfiguration is a start-up issue that
        should surface, not be silently masked by a fallback.
        """
        from .llm import LLMConfigError
        transient = self._transient_llm_errors()
        last_exc: Exception | None = None
        # Try primary FIRST; only construct fallback clients lazily if primary
        # fails with a transient error (saves startup/per-turn overhead).
        try:
            return self.llm.chat(
                system, messages, tools=tools, max_tokens=max_tokens
            )
        except transient as e:
            last_exc = e
            if getattr(self.runtime, "trace", None) is not None:
                self.runtime.trace.record_event(
                    "llm_fallback",
                    {"from": "primary", "err": str(e)[:200]},
                )

        for profile_name in self.fallback_chain:
            try:
                candidate = self._client_for(profile_name)
            except LLMConfigError:
                # Misconfigured fallback profile — surface immediately
                raise
            except Exception as e:
                if getattr(self.runtime, "trace", None) is not None:
                    self.runtime.trace.record_event(
                        "llm_fallback_build_failed",
                        {"profile": profile_name, "err": str(e)[:200]},
                    )
                continue
            try:
                return candidate.chat(
                    system, messages, tools=tools, max_tokens=max_tokens
                )
            except transient as e:
                last_exc = e
                if getattr(self.runtime, "trace", None) is not None:
                    self.runtime.trace.record_event(
                        "llm_fallback",
                        {"from": profile_name, "err": str(e)[:200]},
                    )
                continue
        # Chain exhausted
        assert last_exc is not None
        raise last_exc

    def _client_for(self, profile_name: str) -> "LLMClient":
        """Build an LLMClient for a named profile (used by the fallback chain)."""
        from .llm import make_llm
        if self.cfg is None:
            raise RuntimeError(
                "fallback_chain requires the orchestrator to be constructed "
                "with a `cfg` argument (ADR-0008)."
            )
        return make_llm(self._load_llm_profile_cfg(profile_name))

    def run_task(self, task: str) -> TaskReport:
        t_start = time.monotonic()
        fast_note = None
        if self.fast_planner is not None:
            report, fast_note = self._try_fast_path(task, t_start)
            if report is not None:
                return report

        milestones = self._decompose(task) if self.decompose else []
        messages: list[dict] = []
        tool_log: list[dict] = []

        intro = f"Task: {task}\n"
        if milestones:
            intro += "Milestones:\n" + "\n".join(f"{i+1}. {m}" for i, m in enumerate(milestones)) + "\n"
        if fast_note:
            intro += f"\n{fast_note}\n"
        intro += (
            "\nMemory (last 15 s):\n" + self.runtime.memory.digest(max_lines=5)
            + "\n\nBegin. Observe first, then act. Call one tool now."
        )
        messages.append({"role": "user", "content": intro})

        consecutive_failures = 0
        # Track completed (object, destination) pick_and_place pairs so the
        # agent cannot loop re-placing the SAME object, while still allowing
        # legitimate multi-object tasks ("put the banana and cube in the box").
        self._completed_pick_places: set[tuple[str, str]] = set()
        self._consecutive_pick_failures = 0
        self._consecutive_text_only = 0
        PICK_AND_PLACE_FAILURE_LIMIT = 3
        TEXT_ONLY_LIMIT = 5
        # Allow per-profile max_tokens override (small 8 GB local models can
        # pin 128-256 in configs/llm/*.yaml; cloud/reasoning models default to 512).
        llm_cfg = (self.cfg._data.get("llm", {}) if self.cfg is not None and hasattr(self.cfg, "_data") else {})
        step_max_tokens = int(llm_cfg.get("max_tokens", 512))
        for step in range(1, self.max_steps + 1):
            resp = self._chat_with_fallback(
                system=SYSTEM_PROMPT,
                messages=self._prune_images(messages),
                tools=TOOL_SPECS,
                max_tokens=step_max_tokens,
            )
            if not resp.tool_calls:
                # 2026-09-03: small local LLMs sometimes forget to emit
                # tool calls and just chat. Track consecutive text-only
                # turns and force task_done after TEXT_ONLY_LIMIT (=5).
                self._consecutive_text_only += 1
                if self._consecutive_text_only >= TEXT_ONLY_LIMIT:
                    last_text = (resp.text or "(no text)")[:200]
                    if self.runtime.trace is not None:
                        self.runtime.trace.record_event(
                            "llm_force_task_done",
                            {
                                "reason": (
                                    f"LLM emitted no tool calls for "
                                    f"{self._consecutive_text_only} consecutive turns"
                                ),
                                "last_text": last_text,
                            },
                        )
                    forced_call = ToolCall(
                        name="task_done",
                        arguments={
                            "success": False,
                            "summary": (
                                f"LLM did not emit any tool call for "
                                f"{self._consecutive_text_only} consecutive turns; aborting. "
                                f"(last text: {last_text[:80]})"
                            ),
                        },
                    )
                    forced_result = self.runtime.execute(
                        forced_call.name, forced_call.arguments
                    )
                    tool_log.append({
                        "step": step, "tool": forced_call.name,
                        "args": forced_call.arguments, "result": forced_result,
                        "forced_after_text_only": self._consecutive_text_only,
                    })
                    return TaskReport(
                        task, False, forced_call.arguments["summary"],
                        step, milestones, tool_log,
                        duration_s=round(time.monotonic() - t_start, 2),
                    )
                messages.append({"role": "assistant", "content": resp.text or "(no text)"})
                messages.append(
                    {
                        "role": "user",
                        "content": "Respond with exactly one tool call (use task_done to finish).",
                    }
                )
                continue
            # Reset text-only counter when LLM does emit a tool call.
            self._consecutive_text_only = 0

            # Execute only the first call, but echo ALL calls and answer each
            # one -- OpenAI-style APIs reject the next request if any
            # tool_call in the assistant message lacks a matching result.
            call = resp.tool_calls[0]

            # Guard against "verify by re-placing" loops on the SAME
            # (object, destination) pair, while still permitting multi-object
            # tasks to pick a different object next.
            _pp_key = None
            if call.name == "pick_and_place":
                _args = call.arguments or {}
                _pp_key = (
                    str(_args.get("object", "")).strip().lower(),
                    str(_args.get("destination", "") or "").strip().lower(),
                )
            if _pp_key is not None and _pp_key in self._completed_pick_places:
                if self.runtime.trace is not None:
                    self.runtime.trace.record_event(
                        "llm_force_task_done",
                        {
                            "reason": (
                                "pick_and_place already succeeded for this "
                                "(object, destination) pair in this task; "
                                "calling it again would re-grab an object already "
                                "at its destination"
                            ),
                            "original_call": {
                                "name": call.name,
                                "arguments": dict(call.arguments or {}),
                            },
                        },
                    )
                call = ToolCall(
                    name="task_done",
                    arguments={
                        "success": True,
                        "summary": (
                            "pick_and_place already succeeded earlier in this "
                            "task; the object is at its destination. (Re-placing "
                            "suppressed.)"
                        ),
                    },
                )

            messages.append(
                {"role": "assistant", "content": resp.text, "tool_calls": list(resp.tool_calls)}
            )
            result = self.runtime.execute(call.name, call.arguments)
            tool_log.append({"step": step, "tool": call.name, "args": call.arguments, "result": result})

            if (
                call.name == "pick_and_place"
                and bool(result.get("ok"))
            ):
                if _pp_key is not None:
                    self._completed_pick_places.add(_pp_key)
                self._consecutive_pick_failures = 0
            elif call.name == "pick_and_place" and not bool(result.get("ok")):
                self._consecutive_pick_failures += 1
                if self._consecutive_pick_failures >= PICK_AND_PLACE_FAILURE_LIMIT:
                    # Force task_done: small local LLMs ignore the
                    # "fail-then-task_done" rule in prompts.py and retry
                    # pick_and_place until max_steps is hit (~2 min waste).
                    last_err = str(result.get("error", ""))[:200]
                    if self.runtime.trace is not None:
                        self.runtime.trace.record_event(
                            "llm_force_task_done",
                            {
                                "reason": (
                                    f"pick_and_place failed "
                                    f"{self._consecutive_pick_failures} times "
                                    "consecutively; forcing task_done"
                                ),
                                "last_result_error": last_err,
                            },
                        )
                    # Actually execute task_done (the prior pick_and_place
                    # has already run; the result var holds its outcome).
                    forced_call = ToolCall(
                        name="task_done",
                        arguments={
                            "success": False,
                            "summary": (
                                f"pick_and_place failed "
                                f"{self._consecutive_pick_failures} times in a row "
                                f"(last: {last_err[:80]}); aborting."
                            ),
                        },
                    )
                    forced_result = self.runtime.execute(
                        forced_call.name, forced_call.arguments
                    )
                    tool_log.append({
                        "step": step, "tool": forced_call.name,
                        "args": forced_call.arguments, "result": forced_result,
                        "forced_after_failures": self._consecutive_pick_failures,
                    })
                    if forced_result.get("task_complete"):
                        return TaskReport(
                            task, False, forced_call.arguments["summary"],
                            step, milestones, tool_log,
                            duration_s=round(time.monotonic() - t_start, 2),
                        )
                    # If task_done itself errored, fall through to next step
                    # (rare; we trust max_steps as the final backstop).
            elif call.name == "move_home":
                # moving home between attempts resets the failure streak
                self._consecutive_pick_failures = 0

            if call.name == "task_done" and result.get("task_complete"):
                summary = str(call.arguments.get("summary", ""))
                success = _as_bool(call.arguments.get("success", False))
                self.runtime.trace.finish(
                    f"task: {task}\nsuccess: {success}\nsteps: {step}\n{summary}"
                )
                return TaskReport(
                    task, success, summary, step, milestones, tool_log,
                    duration_s=round(time.monotonic() - t_start, 2),
                )

            ok = bool(result.get("ok"))
            consecutive_failures = 0 if ok else consecutive_failures + 1
            if self.advisor is not None:
                self.advisor.note_outcome(ok)

            content = json.dumps(result)
            messages.append(
                {"role": "tool", "tool_call_id": call.id or "call_0", "name": call.name, "content": content}
            )
            for extra in resp.tool_calls[1:]:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": extra.id or "call_x",
                        "name": extra.name,
                        "content": json.dumps(
                            {"ok": False, "error": "skipped: one tool call per turn; re-issue if still needed"}
                        ),
                    }
                )
            # Sliding window: keep intro + newest tail, but NEVER orphan a
            # role="tool" message from its preceding assistant(tool_calls)
            # turn (OpenAI & Anthropic APIs reject orphaned tool messages
            # with HTTP 400).
            messages = self._trim_history(messages, max_history=8)

            # Failure escalation: one visual suggestion from the advisor.
            if not ok and self.advisor is not None and self.advisor.should_consult:
                jpeg = self.runtime.frame_jpeg()
                if jpeg:
                    suggestion = self.advisor.suggest(
                        task, jpeg, failure=str(result.get("error", ""))
                    )
                    if suggestion:
                        self.runtime.memory.add("note", f"advisor: {suggestion}")
                        messages.append(
                            {"role": "user", "content": f"Advisor suggestion: {suggestion}"}
                        )

            # Refresh the agent's situational context periodically.
            extra: dict = {}
            if self.attach_images and self.runtime.last_frame is not None:
                extra = {"images": [self.runtime.frame_jpeg()]}
            if not ok and consecutive_failures >= 2:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Two consecutive failures. Change strategy: re-observe, "
                            "try a different approach, or finish with task_done("
                            "success=false, honest summary).\nMemory:\n"
                            + self.runtime.memory.digest(max_lines=5)
                        ),
                        **extra,
                    }
                )

        self.runtime.trace.finish(f"task: {task}\nsuccess: false\nran out of steps ({self.max_steps})")
        return TaskReport(
            task, False, "step budget exhausted", self.max_steps, milestones, tool_log,
            duration_s=round(time.monotonic() - t_start, 2),
        )

    def _try_fast_path(self, task: str, t_start: float) -> tuple[TaskReport | None, str | None]:
        """Reflex/experience execution; (report, None) on success, or
        (None, note-for-the-LLM) when the fast attempt failed or no fast
        plan exists."""
        plan = self.fast_planner.plan(task)
        if plan is None:
            return None, None
        tool_log: list[dict] = []
        for i, (name, args) in enumerate(plan.calls, start=1):
            result = self.runtime.execute(name, args)
            tool_log.append({"step": i, "tool": name, "args": args, "result": result})
            if not result.get("ok", False):
                self.fast_planner.note_outcome(
                    task, plan.calls, False, time.monotonic() - t_start
                )
                note = (
                    f"(A fast {plan.source} plan was tried first and FAILED at "
                    f"{name}({json.dumps(args)}): {str(result.get('error', ''))[:200]}. "
                    "Diagnose before retrying the same thing.)"
                )
                return None, note
        duration = round(time.monotonic() - t_start, 2)
        self.fast_planner.note_outcome(task, plan.calls, True, duration)
        summary = (
            f"done via {plan.source} path in {duration}s: "
            + "; ".join(f"{n}({_short_args(a)})" for n, a in plan.calls)
        )
        self.runtime.trace.finish(
            f"task: {task}\nsuccess: true\npath: {plan.source}\nduration_s: {duration}\n{summary}"
        )
        return (
            TaskReport(
                task, True, summary, len(plan.calls), [], tool_log,
                path=plan.source, duration_s=duration,
            ),
            None,
        )

    @staticmethod
    def _trim_history(messages: list[dict], max_history: int = 8) -> list[dict]:
        """Keep intro + recent tail without ever starting the tail on an
        orphaned role='tool' message (which breaks OpenAI/Anthropic APIs)."""
        if len(messages) <= max_history:
            return messages
        start = len(messages) - (max_history - 1)
        while start < len(messages) and messages[start].get("role") == "tool":
            start += 1
        return messages[:1] + messages[start:]

    @staticmethod
    def _prune_images(messages: list[dict]) -> list[dict]:
        """Keep only the newest image in context: frames are ~50-100 KB of
        base64 each and every retained one is re-sent on every request."""
        last_with_images = None
        for i, m in enumerate(messages):
            if m.get("images"):
                last_with_images = i
        if last_with_images is None:
            return messages
        pruned = []
        for i, m in enumerate(messages):
            if m.get("images") and i != last_with_images:
                m = {k: v for k, v in m.items() if k != "images"}
            pruned.append(m)
        return pruned

    def _decompose(self, task: str) -> list[str]:
        try:
            resp: LLMResponse = self._chat_with_fallback(
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": DECOMPOSE_PROMPT.format(task=task)}],
                max_tokens=300,
            )
        except Exception:
            return []
        lines = []
        for raw in resp.text.splitlines():
            line = raw.strip().lstrip("0123456789.-* ").strip()
            if line:
                lines.append(line)
        return lines[:6]
