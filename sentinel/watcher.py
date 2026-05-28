from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable

from watchdog.events import FileSystemEventHandler, FileSystemEvent
from watchdog.observers import Observer

from sentinel.policy import SentinelPolicy, load_policy


class _PolicyEventHandler(FileSystemEventHandler):
    def __init__(
        self,
        policy_path: Path,
        on_reload: Callable[[SentinelPolicy], None],
        debounce_seconds: float = 0.5,
    ) -> None:
        super().__init__()
        self._policy_path = policy_path.resolve()
        self._on_reload = on_reload
        self._debounce = debounce_seconds
        self._last_reload: float = 0.0
        self._lock = threading.Lock()

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        if Path(event.src_path).resolve() != self._policy_path:
            return
        now = time.monotonic()
        with self._lock:
            if now - self._last_reload < self._debounce:
                return
            self._last_reload = now
        threading.Thread(target=self._reload, daemon=True).start()

    def _reload(self) -> None:
        try:
            new_policy = load_policy(self._policy_path)
            self._on_reload(new_policy)
            print(f"[sentinel] Policy reloaded from {self._policy_path}", flush=True)
        except Exception as exc:
            print(f"[sentinel] Policy reload failed: {exc}", flush=True)


class PolicyWatcher:
    def __init__(
        self,
        policy_path: str | Path,
        on_reload: Callable[[SentinelPolicy], None],
        debounce_seconds: float = 0.5,
    ) -> None:
        self._policy_path = Path(policy_path).resolve()
        self._handler = _PolicyEventHandler(
            self._policy_path, on_reload, debounce_seconds
        )
        self._observer = Observer()

    def start(self) -> None:
        watch_dir = str(self._policy_path.parent)
        if not self._policy_path.parent.exists():
            print(
                f"[sentinel] Watcher not started: directory does not exist: {watch_dir}",
                flush=True,
            )
            return
        try:
            self._observer.schedule(self._handler, watch_dir, recursive=False)
            self._observer.start()
        except Exception as exc:
            print(f"[sentinel] Watcher failed to start: {exc}", flush=True)

    def stop(self) -> None:
        try:
            if self._observer.is_alive():
                self._observer.stop()
                self._observer.join()
        except Exception:
            pass
