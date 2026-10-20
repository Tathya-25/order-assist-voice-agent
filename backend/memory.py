import os
import json
import time
import logging
from typing import List, Dict, Any, Tuple
import redis
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("orderassist.memory")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))
MAX_TURNS = 6

# Graceful in-memory fallback dictionary: { session_id: {"expires_at": float, "turns": list} }
_FALLBACK_MEMORY: Dict[str, Dict[str, Any]] = {}
_redis_client: redis.Redis | None = None
_redis_available: bool = False


def get_redis_client() -> Tuple[redis.Redis | None, bool]:
    """Attempts to connect to Redis with a short timeout; falls back to in-memory dict if unreachable."""
    global _redis_client, _redis_available
    if _redis_client is not None and _redis_available:
        try:
            _redis_client.ping()
            return _redis_client, True
        except Exception:
            _redis_available = False

    try:
        client = redis.Redis.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=1.0,
            socket_timeout=1.0,
        )
        client.ping()
        _redis_client = client
        _redis_available = True
        return _redis_client, True
    except Exception as exc:
        logger.info("Redis unreachable (%s). Using resilient in-memory TTL store.", exc)
        _redis_available = False
        return None, False


def _session_key(session_id: str) -> str:
    return f"orderassist:session:{session_id}"


def get_session_history(session_id: str) -> List[Dict[str, str]]:
    """
    Retrieves the last 6 conversational turns for `session_id`.
    Each turn is a dictionary: {"user": "...", "agent": "...", "timestamp": "..."}.
    """
    client, is_redis = get_redis_client()
    key = _session_key(session_id)

    if is_redis and client is not None:
        try:
            raw = client.get(key)
            if not raw:
                return []
            data = json.loads(raw)
            return data[-MAX_TURNS:] if isinstance(data, list) else []
        except Exception as exc:
            logger.warning("Redis read error on %s: %s", key, exc)

    # Fallback path with 3600s TTL enforcement
    entry = _FALLBACK_MEMORY.get(key)
    if not entry:
        return []
    if time.time() > entry.get("expires_at", 0):
        _FALLBACK_MEMORY.pop(key, None)
        return []
    turns = entry.get("turns", [])
    return turns[-MAX_TURNS:]


def save_session_turn(session_id: str, user_msg: str, agent_msg: str) -> Tuple[int, str]:
    """
    Appends a new (user_msg, agent_msg) turn to session history, trims to the last 6 turns,
    and persists using Redis SETEX with a 3600-second (1-hour) TTL.
    Returns (turn_count, backend_mode).
    """
    history = get_session_history(session_id)
    history.append(
        {
            "user": user_msg,
            "agent": agent_msg,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    trimmed = history[-MAX_TURNS:]
    key = _session_key(session_id)

    client, is_redis = get_redis_client()
    if is_redis and client is not None:
        try:
            client.setex(key, SESSION_TTL_SECONDS, json.dumps(trimmed))
            return len(trimmed), "redis"
        except Exception as exc:
            logger.warning("Redis SETEX failed on %s: %s. Writing to fallback.", key, exc)

    _FALLBACK_MEMORY[key] = {
        "expires_at": time.time() + SESSION_TTL_SECONDS,
        "turns": trimmed,
    }
    return len(trimmed), "in_memory_fallback"


def clear_session_history(session_id: str) -> bool:
    """Deletes a session key from both Redis and in-memory fallback."""
    key = _session_key(session_id)
    _FALLBACK_MEMORY.pop(key, None)
    client, is_redis = get_redis_client()
    if is_redis and client is not None:
        try:
            client.delete(key)
        except Exception:
            pass
    return True
