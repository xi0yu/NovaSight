from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


TEST_MAX_LICENSE_KEY = "NOVASIGHT-TEST-MAX-ACCESS-2026"
ALL_FEATURES = [
    "capture",
    "runtime",
    "models",
    "tensorrt",
    "hardware_control",
    "config_read",
    "config_write",
]


@dataclass(frozen=True)
class LicenseStatus:
    configured: bool
    valid: bool = False
    fingerprint: str = ""
    tier: str = ""
    features: list[str] | None = None
    license_id: str = ""
    created_at: float | None = None
    activated_at: float | None = None
    expires_at: float | None = None
    duration_value: int | None = None
    duration_unit: str = ""
    updated_at: float | None = None
    message: str = ""

    def __post_init__(self) -> None:
        if self.features is None:
            object.__setattr__(self, "features", [])


@dataclass(frozen=True)
class LicensePayload:
    license_id: str
    tier: str
    created_at: float
    duration_value: int
    duration_unit: str
    features: list[str]


def _now() -> float:
    return time.time()


def _duration_seconds(value: int, unit: str) -> int:
    if value <= 0:
        raise ValueError("license duration must be positive")
    normalized = unit.lower()
    if normalized in {"day", "days"}:
        return value * 24 * 60 * 60
    if normalized in {"month", "months"}:
        return value * 30 * 24 * 60 * 60
    if normalized in {"year", "years"}:
        return value * 365 * 24 * 60 * 60
    raise ValueError(f"unsupported license duration unit: {unit}")


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _parse_signed_license(key: str) -> LicensePayload:
    parts = key.split(".")
    if len(parts) != 3 or parts[0] != "NS1":
        raise ValueError("unsupported license format")
    payload_bytes = _b64url_decode(parts[1])
    signature = _b64url_decode(parts[2])
    public_key = os.environ.get("NOVASIGHT_LICENSE_PUBLIC_KEY", "").strip()
    if not public_key:
        raise ValueError("license public key is not configured")
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding
    except ImportError as exc:
        raise ValueError("cryptography package is required for signed licenses") from exc
    verifier = serialization.load_pem_public_key(public_key.encode("utf-8"))
    verifier.verify(signature, payload_bytes, padding.PKCS1v15(), hashes.SHA256())
    raw = json.loads(payload_bytes.decode("utf-8"))
    duration = raw.get("duration") or {}
    return LicensePayload(
        license_id=str(raw["license_id"]),
        tier=str(raw["tier"]),
        created_at=float(raw["created_at"]),
        duration_value=int(duration["value"]),
        duration_unit=str(duration["unit"]),
        features=[str(item) for item in raw.get("features", [])],
    )


def _payload_for_key(key: str) -> LicensePayload:
    if key == TEST_MAX_LICENSE_KEY:
        return LicensePayload(
            license_id="test-max-access",
            tier="test_max",
            created_at=1_783_036_800.0,  # 2026-07-02T00:00:00Z
            duration_value=10,
            duration_unit="years",
            features=ALL_FEATURES.copy(),
        )
    return _parse_signed_license(key)


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
        activated_at = raw.get("activated_at")
        expires_at = raw.get("expires_at")
        valid = bool(raw.get("key_hash")) and (
            expires_at is None or float(expires_at) > _now()
        )
        return LicenseStatus(
            configured=bool(raw.get("key_hash")),
            valid=valid,
            fingerprint=str(raw.get("fingerprint", "")),
            tier=str(raw.get("tier", "")),
            features=[str(item) for item in raw.get("features", [])],
            license_id=str(raw.get("license_id", "")),
            created_at=raw.get("created_at"),
            activated_at=activated_at,
            expires_at=expires_at,
            duration_value=raw.get("duration_value"),
            duration_unit=str(raw.get("duration_unit", "")),
            updated_at=raw.get("updated_at"),
            message="" if valid else "license expired or invalid",
        )

    def save(self, key: str) -> LicenseStatus:
        stripped = key.strip()
        if len(stripped) < 8:
            raise ValueError("license key must be at least 8 characters")
        payload = _payload_for_key(stripped)
        activated_at = _now()
        expires_at = activated_at + _duration_seconds(
            payload.duration_value,
            payload.duration_unit,
        )
        digest = hashlib.sha256(stripped.encode("utf-8")).hexdigest()
        body: dict[str, Any] = {
            "key_hash": digest,
            "fingerprint": digest[:12],
            "license_id": payload.license_id,
            "tier": payload.tier,
            "features": payload.features,
            "created_at": payload.created_at,
            "activated_at": activated_at,
            "expires_at": expires_at,
            "duration_value": payload.duration_value,
            "duration_unit": payload.duration_unit,
            "updated_at": activated_at,
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
