from __future__ import annotations

import hashlib

import pytest

import novasight.license as license_module
from novasight.license import LicenseStore, TEST_MAX_LICENSE_KEY


def test_offline_license_duration_starts_at_signed_creation_time(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    activated_at = 1_900_000_000.0
    monkeypatch.setattr(license_module, "_now", lambda: activated_at)
    store = LicenseStore(tmp_path / "license.json")

    status = store.save(TEST_MAX_LICENSE_KEY)

    expected_expiry = 1_783_036_800.0 + 10 * 365 * 24 * 60 * 60
    assert status.activated_at == activated_at
    assert status.expires_at == expected_expiry
    assert status.expires_at != activated_at + 10 * 365 * 24 * 60 * 60


def test_legacy_python_document_keeps_only_the_key_hash(tmp_path) -> None:
    path = tmp_path / "license.json"
    store = LicenseStore(path)

    store.save(TEST_MAX_LICENSE_KEY)

    document = path.read_text(encoding="utf-8")
    assert TEST_MAX_LICENSE_KEY not in document
    assert hashlib.sha256(TEST_MAX_LICENSE_KEY.encode()).hexdigest() in document
