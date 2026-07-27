from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import novasight.license as license_module
from novasight.license import LicenseStore


TESTDATA = Path(__file__).resolve().parents[1] / "testdata"
SIGNED_LICENSE = (TESTDATA / "license-signed.key").read_text(encoding="utf-8").strip()
PUBLIC_KEY = (TESTDATA / "license-public.pem").read_text(encoding="utf-8")


def test_offline_license_duration_starts_at_signed_creation_time(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    activated_at = 1_900_000_000.0
    monkeypatch.setattr(license_module, "_now", lambda: activated_at)
    monkeypatch.setenv("NOVASIGHT_LICENSE_PUBLIC_KEY", PUBLIC_KEY)
    store = LicenseStore(tmp_path / "license.json")

    status = store.save(SIGNED_LICENSE)

    expected_expiry = 1_783_036_800.0 + 100 * 365 * 24 * 60 * 60
    assert status.activated_at == activated_at
    assert status.expires_at == expected_expiry
    assert status.expires_at != activated_at + 100 * 365 * 24 * 60 * 60


def test_python_document_keeps_only_the_signed_key_hash(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NOVASIGHT_LICENSE_PUBLIC_KEY", PUBLIC_KEY)
    path = tmp_path / "license.json"
    store = LicenseStore(path)

    store.save(SIGNED_LICENSE)

    document = path.read_text(encoding="utf-8")
    assert SIGNED_LICENSE not in document
    assert hashlib.sha256(SIGNED_LICENSE.encode()).hexdigest() in document
