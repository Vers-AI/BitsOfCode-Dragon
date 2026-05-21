"""
Telemetry module
Purpose: Structured JSONL logging via TELEM-prefixed stdout lines.
Key Decisions: TELEM prefix for easy grep from AI Arena logs; sampling by env;
              module-level state for transition detection (single-process, one game at a time).
Limitations: No async/batched I/O needed — print() is synchronous but cheap for ~20 events/game.
"""

import json
import random
import sys
import uuid
from datetime import datetime, timezone

from bot import BOT_VERSION

SCHEMA_VERSION = 1

TELEMETRY_SAMPLE_RATE = {"local": 0.1, "ci": 0.5, "ladder": 1.0}

_context: dict = {}
_previous_transitions: dict = {}
_initialized: bool = False


def _get_env() -> str:
    """Return 'ladder' or 'local' based on sys.argv (mirrors run.py pattern)."""
    return "ladder" if "--LadderServer" in sys.argv else "local"


def init_context(bot) -> None:
    """Bind game-level context once per game. Must be called in on_start()."""
    global _context, _previous_transitions, _initialized

    _context = {
        "schema_version": SCHEMA_VERSION,
        "version": BOT_VERSION,
        "env": _get_env(),
        "match_id": str(uuid.uuid4()),
        "arena_match_id": None,  # Filled by puller post-match
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

    env = _context.get("env", "local")
    if random.random() > TELEMETRY_SAMPLE_RATE.get(env, 0.1):
        return

    ts = kwargs.pop("_ts", None)
    record = {**_context, "subsystem": subsystem, "action": action, "reason": reason, **kwargs}
    if ts is not None:
        record["ts"] = ts

    _emit(record)


def log_event_no_sample(subsystem: str, action: str, reason: str, **kwargs) -> None:
    """Emit an event record bypassing sampling (for transition and decision events)."""
    if not _initialized:
        return

    ts = kwargs.pop("_ts", None)
    record = {**_context, "subsystem": subsystem, "action": action, "reason": reason, **kwargs}
    if ts is not None:
        record["ts"] = ts

    _emit(record)


def log_match(**kwargs) -> None:
    """Emit a match record (one per game). Never sampled."""
    if not _initialized:
        return

    record = {
        **_context,
        "ts": datetime.now(timezone.utc).isoformat(),
        **kwargs,
    }

    _emit(record)


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
    """Return current context dict (for external access to match_id etc.)."""
    return _context.copy()


def reset() -> None:
    """Reset all state. Called between games if needed."""
    global _context, _previous_transitions, _initialized
    _context = {}
    _previous_transitions = {}
    _initialized = False


def _emit(record: dict) -> None:
    """Print TELEM-prefixed JSON to stdout."""
    try:
        print("TELEM " + json.dumps(record, default=str))
    except (TypeError, ValueError) as e:
        print(f"TELEM_ERROR: Failed to serialize: {e}", file=sys.stderr)