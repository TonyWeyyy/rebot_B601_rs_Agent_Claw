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
        self.attach_images = attach_images and llm.supports_vision
        self.fast_planner = fast_planner
        # ADR-0008: profile switching + fallback. `cfg` is the live
        # wrc_demo.config.Cfg (or any object with a writable `_data["llm"]`
        # dict); `fallback_chain` is an ordered list of profile names tried
        # when the primary chat() raises a transient SDK error. Both default
        # to inert values, so existing callers (demo.py, tests) are
        # backward-compatible.
        self.cfg = cfg
        self.fallback_chain = list(fallback_chain)

    # ───────────────────────────────────────────────────────────────────
    # ADR-0008: profile switching + transient-error fallback
    # ───────────────────────────────────────────────────────────────────

    def switch_llm(self, profile_name: str) -> "LLMClient":
        """Swap the live LLM client for a new one built from `cfg`.

        `profile_name` overrides `cfg.llm["profile"]` (or sets it if
        absent) before re-deriving the client. The new client becomes
        `self.llm`; the next `chat()` uses it. An `llm_switch` event is
        recorded in the trace.
        """
        from .llm import make_llm
        if self.cfg is None:
            raise RuntimeError(
                "switch_llm requires the orchestrator to be constructed "
                "with a `cfg` argument (ADR-0008)."
            )
        sub_cfg = dict(self.cfg._data["llm"])
        sub_cfg["profile"] = profile_name
        new_llm = make_llm(sub_cfg)
        self.llm = new_llm
        if getattr(self.runtime, "trace", None) is not None:
            self.runtime.trace.record_event(
                "llm_switch", {"profile": profile_name}
            )
        return new_llm

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
        import openai
        transient = (
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.RateLimitError,
            openai.AuthenticationError,
            openai.InternalServerError,
            openai.BadRequestError,  # 4xx often means the model can't parse the request — recoverable
        )
        last_exc: Exception | None = None
        chain: list[tuple[str, LLMClient]] = [("primary", self.llm)]
        for profile_name in self.fallback_chain:
            try:
                chain.append((profile_name, self._client_for(profile_name)))
            except LLMConfigError:
                # Misconfigured fallback profile — surface immediately
                raise
            except Exception as e:
                # Building the fallback itself failed; record and move on
                if getattr(self.runtime, "trace", None) is not None:
                    self.runtime.trace.record_event(
                        "llm_fallback_build_failed",
                        {"profile": profile_name, "err": str(e)[:200]},
                    )
                continue
        for label, candidate in chain:
            try:
                return candidate.chat(
                    system, messages, tools=tools, max_tokens=max_tokens
                )
            except transient as e:
                last_exc = e
                if getattr(self.runtime, "trace", None) is not None:
                    self.runtime.trace.record_event(
                        "llm_fallback",
                        {"from": label, "err": str(e)[:200]},
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
        sub_cfg = dict(self.cfg._data["llm"])
        sub_cfg["profile"] = profile_name
        return make_llm(sub_cfg)

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
# 2026-08-13: track whether pick_and_place has succeeded in this task.
        # If yes, the next call must be task_done (or move_home + task_done) —
        # never another pick_and_place. The LLM is prone to "verify by re-placing"
        # loops when list_objects still shows the object at its destination.
        self._pick_and_place_succeeded_in_task = False
        # 2026-09-03: same shape for failures. Small local LLMs (Qwen3-VL-2B
        # AWQ-4bit) sometimes ignore the "if pick_and_place fails, call
        # task_done" rule in prompts.py and re-issue the same failing call
        # until max_steps (30) is hit (~2 min of wasted GPU). Cap consecutive
        # pick_and_place failures and force task_done instead.
        # Also: some small LLMs stop calling tools and just emit text
        # ("I see a red cube...") over and over. Track consecutive text-only
        # turns and force task_done.
        self._consecutive_pick_failures = 0
        self._consecutive_text_only = 0
        PICK_AND_PLACE_FAILURE_LIMIT = 3
        # Bumped 2026-09-03: minimax-M3 cloud LLM does more reasoning per turn
        # than local Qwen3-VL did; 3 turns was too tight — the LLM was on
        # the verge of correctly calling task_done(success=false, "no wave
        # tool") when the counter fired. 5 gives one more thinking turn
        # of headroom without losing the runaway-loop protection.
        TEXT_ONLY_LIMIT = 5
        for step in range(1, self.max_steps + 1):
            resp = self._chat_with_fallback(
                system=SYSTEM_PROMPT,
                messages=self._prune_images(messages),
                tools=TOOL_SPECS,
                # Tool calls are structured JSON; ~128 tokens is plenty
                # (largest spec is `pick_and_place` at ~80 tokens). 256 was
                # the historical value but it leaves no headroom on 8 GB
                # VRAM RTX 5070 (model_len=2048, system prompt ~1.8 k tokens
                # after a few turns).
                max_tokens=128,
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
                        str(step), milestones, tool_log,
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

            # 2026-08-13: guard against "verify by re-placing" loops. After a
            # successful pick_and_place, the object is already at its
            # destination. Re-running pick_and_place just re-grabs and
            # re-places the same object. Force task_done.
            if (
                self._pick_and_place_succeeded_in_task
                and call.name == "pick_and_place"
            ):
                self.runtime.trace.record_event(
                    "llm_force_task_done",
                    {
                        "reason": (
                            "pick_and_place already succeeded in this task; "
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

            # 2026-08-13: flag successful pick_and_place so the guard above can
            # intercept any subsequent pick_and_place calls.
            if (
                call.name == "pick_and_place"
                and bool(result.get("ok"))
            ):
                self._pick_and_place_succeeded_in_task = True
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
                            str(step), milestones, tool_log,
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
            # Sliding window: keep intro + last N to bound context growth
            # (default N=8 ~= 4 assistant turns + 4 tool results). Each turn
            # appends up to 3 entries (assistant w/ tool_calls, tool, extras);
            # N=8 caps at ~3 turns of history which is plenty for short tasks.
            _MAX_HISTORY = 8
            if len(messages) > _MAX_HISTORY:
                messages = messages[:1] + messages[-( _MAX_HISTORY - 1):]

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
