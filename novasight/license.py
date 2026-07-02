from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass(frozen=True)
class LicenseStatus:
    configured: bool
    fingerprint: str = ""
    updated_at: float | None = None


class LicenseStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def status(self) -> LicenseStatus:
        if not self.path.exists():
            return LicenseStatus(configured=False)
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return LicenseStatus(configured=False)
        return LicenseStatus(
            configured=bool(raw.get("key_hash")),
            fingerprint=str(raw.get("fingerprint", "")),
            updated_at=raw.get("updated_at"),
        )

    def save(self, key: str) -> LicenseStatus:
        stripped = key.strip()
        if len(stripped) < 8:
            raise ValueError("license key must be at least 8 characters")
        digest = hashlib.sha256(stripped.encode("utf-8")).hexdigest()
        body = {
            "key_hash": digest,
            "fingerprint": digest[:12],
            "updated_at": time.time(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
        return self.status()

    def clear(self) -> LicenseStatus:
        if self.path.exists():
            self.path.unlink()
        return self.status()

    def asdict(self) -> dict:
        return asdict(self.status())
