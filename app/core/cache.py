"""Small cache abstraction: in-process TTL cache by default, Redis when configured.

Used for query embeddings, retrieval results and the component alias index.
Values must be picklable.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import threading
from abc import ABC, abstractmethod
from typing import Any

from cachetools import TTLCache

from app.core.logging import get_logger

log = get_logger(__name__)


def make_key(namespace: str, *parts: Any) -> str:
    raw = json.dumps(parts, sort_keys=True, default=str)
    return f"rag:{namespace}:{hashlib.sha256(raw.encode()).hexdigest()}"


class Cache(ABC):
    @abstractmethod
    def get(self, key: str) -> Any | None: ...

    @abstractmethod
    def set(self, key: str, value: Any, ttl: int | None = None) -> None: ...

    @abstractmethod
    def clear(self) -> None: ...


class InMemoryCache(Cache):
    def __init__(self, max_items: int = 2048, ttl_seconds: int = 600) -> None:
        self._cache: TTLCache = TTLCache(maxsize=max_items, ttl=ttl_seconds)
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            return self._cache.get(key)

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        with self._lock:
            self._cache[key] = value

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


class RedisCache(Cache):
    def __init__(self, url: str, ttl_seconds: int = 600) -> None:
        import redis  # optional dependency

        self._client = redis.Redis.from_url(url)
        self._ttl = ttl_seconds

    def get(self, key: str) -> Any | None:
        try:
            raw = self._client.get(key)
            return pickle.loads(raw) if raw is not None else None
        except Exception as exc:  # cache must never break a request
            log.warning("cache_get_failed", error=str(exc))
            return None

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        try:
            self._client.setex(key, ttl or self._ttl, pickle.dumps(value))
        except Exception as exc:
            log.warning("cache_set_failed", error=str(exc))

    def clear(self) -> None:
        try:
            for key in self._client.scan_iter("rag:*"):
                self._client.delete(key)
        except Exception as exc:
            log.warning("cache_clear_failed", error=str(exc))


class NullCache(Cache):
    def get(self, key: str) -> Any | None:
        return None

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        return None

    def clear(self) -> None:
        return None


def build_cache(redis_url: str | None, max_items: int, ttl_seconds: int) -> Cache:
    if redis_url:
        try:
            return RedisCache(redis_url, ttl_seconds)
        except ImportError:
            log.warning("redis_not_installed_falling_back_to_memory_cache")
    if ttl_seconds <= 0:
        return NullCache()
    return InMemoryCache(max_items=max_items, ttl_seconds=ttl_seconds)
