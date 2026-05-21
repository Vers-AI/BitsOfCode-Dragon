"""
Telemetry module
Purpose: Structured JSONL logging via TELEM-prefixed stdout lines.
Key Decisions: TELEM prefix for easy grep from AI Arena logs; sampling by env;
              module-level state for transition detection (single-process, one game at a time);
              local games write to data/games/<match_id>.jsonl (same structure as puller pipeline);
              atexit handler emits incomplete match record on crash/forced exit.
Limitations: No async/batched I/O needed — print() is synchronous but cheap for ~20 events/game.
"""

import atexit
import json
import random
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from bot import BOT_VERSION

_GAMES_DIR = Path("data/games")

SCHEMA_VERSION = 1

TELEMETRY_SAMPLE_RATE = {"local": 0.1, "ci": 0.5, "ladder": 1.0}

_event_context: dict = {}
_match_context: dict = {}
_previous_transitions: dict = {}
_initialized: bool = False
_match_completed: bool = False


def _atexit_handler() -> None:
    """Emit incomplete match record if game was in progress but on_end never fired."""
    if _initialized and not _match_completed:
        record = {
            **_match_context,
            "ts": datetime.now(timezone.utc).isoformat(),
            "result": "incomplete",
            "length": None,
        }
        _emit(record)


atexit.register(_atexit_handler)


def _get_env() -> str:
    """Return 'ladder' or 'local' based on sys.argv (mirrors run.py pattern)."""
    return "ladder" if "--LadderServer" in sys.argv else "local"


def init_context(bot) -> None:
    """Bind game-level context once per game. Must be called in on_start()."""
    global _event_context, _match_context, _previous_transitions, _initialized

    _event_context = {
        "schema_version": SCHEMA_VERSION,
        "version": BOT_VERSION,
        "env": _get_env(),
        "match_id": str(uuid.uuid4()),
    }

    _match_context = {
        **_event_context,
        "arena_match_id": None,
        "opponent_id": getattr(bot, 'opponent_id', None),
        "bot_race": bot.race.name if hasattr(bot, 'race') else "Protoss",
        "enemy_race": bot.enemy_race.name,
        "map": bot.game_info.map_name,
        "chosen_opening": bot.build_order_runner.chosen_opening,
        "rush_time_seconds": round(getattr(bot, '_rush_time_seconds', 0.0), 1),
    }

    _previous_transitions = {}
    _initialized = True


def log_event(subsystem: str, action: str, reason: str, **kwargs) -> None:
    """Emit an event record. Subject to sampling by environment."""
    if not _initialized:
        return

    env = _event_context.get("env", "local")
    if random.random() > TELEMETRY_SAMPLE_RATE.get(env, 0.1):
        return

    ts = kwargs.pop("_ts", None)
    record = {**_event_context, "subsystem": subsystem, "action": action, "reason": reason, **kwargs}
    if ts is not None:
        record["ts"] = ts

    _emit(record)


def log_event_no_sample(subsystem: str, action: str, reason: str, **kwargs) -> None:
    """Emit an event record bypassing sampling (for transition and decision events)."""
    if not _initialized:
        return

    ts = kwargs.pop("_ts", None)
    record = {**_event_context, "subsystem": subsystem, "action": action, "reason": reason, **kwargs}
    if ts is not None:
        record["ts"] = ts

    _emit(record)


def log_match(**kwargs) -> None:
    """Emit a match record (one per game). Never sampled."""
    global _match_completed
    if not _initialized:
        return

    record = {
        **_match_context,
        "ts": datetime.now(timezone.utc).isoformat(),
        **kwargs,
    }

    _emit(record)
    _match_completed = True


def log_transition(subsystem: str, action: str, reason: str, key: str, value, **kwargs) -> bool:
    """Emit a transition event only if value changed from previous.

    Returns True if emitted (value changed), False if skipped (same value).
    """
    if not _initialized:
        return False

    if _previous_transitions.get(key) == value:
        return False

    _previous_transitions[key] = value
    log_event_no_sample(subsystem=subsystem, action=action, reason=reason, **{key: value}, **kwargs)
    return True


def get_context() -> dict:
    """Return current match context dict (for external access to match_id etc.)."""
    return _match_context.copy()


def reset() -> None:
    """Reset all state. Called between games if needed."""
    global _event_context, _match_context, _previous_transitions, _initialized, _match_completed
    _event_context = {}
    _match_context = {}
    _previous_transitions = {}
    _initialized = False
    _match_completed = False


def _emit(record: dict) -> None:
    """Write telemetry record. Ladder: stdout (captured by bot controller). Local: file only."""
    try:
        # Round game-time floats to avoid noise like 30.000000000000004
        if "ts" in record and isinstance(record["ts"], float):
            record["ts"] = round(record["ts"], 1)
        line = json.dumps(record, default=str)
        env = _event_context.get("env", "local")
        if env == "ladder":
            # stdout is the transport out of the container
            print("TELEM " + line)
        else:
            # Local: per-game file under data/games/ (same structure as puller pipeline)
            match_id = _event_context.get("match_id", "unknown")
            _GAMES_DIR.mkdir(parents=True, exist_ok=True)
            with open(_GAMES_DIR / f"{match_id}.jsonl", "a") as f:
                f.write(line + "\n")
    except (TypeError, ValueError) as e:
        print(f"TELEM_ERROR: Failed to serialize: {e}", file=sys.stderr)