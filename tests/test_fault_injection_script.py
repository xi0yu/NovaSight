from pathlib import Path

import pytest

from scripts import fault_injection


@pytest.mark.parametrize(
    ("registration_status", "registration_body", "expected_passed"),
    [
        (
            400,
            {"detail": "catalog registration currently supports TensorRT .engine files only"},
            True,
        ),
        (404, {"detail": "Not Found"}, False),
    ],
)
def test_bad_model_probe_requires_expected_catalog_validation_error(
    tmp_path,
    monkeypatch,
    registration_status: int,
    registration_body: dict[str, str],
    expected_passed: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    requests: list[tuple[str, str, dict[str, object] | None]] = []

    def fake_request(
        _base_url: str,
        path: str,
        *,
        method: str = "GET",
        headers: dict[str, str],
        body: dict[str, object] | None = None,
        allow_http_error: bool = False,
    ) -> dict[str, object]:
        del headers, allow_http_error
        requests.append((method, path, body))
        if path == "/api/models/catalog/register":
            return {"status_code": registration_status, "body": registration_body}
        if path == "/healthz":
            return {"status_code": 200, "body": {"ok": True}}
        if path == "/api/runtime/state":
            return {"status_code": 200, "body": {}}
        raise AssertionError(f"unexpected request path: {path}")

    monkeypatch.setattr(fault_injection, "_request_json", fake_request)

    result = fault_injection.run_bad_model_probe(
        base_url="http://127.0.0.1:8000",
        headers={},
        model_id="fault/test",
        source_path="",
    )

    assert result.passed is expected_passed
    assert requests[0][0:2] == ("POST", "/api/models/catalog/register")
    assert requests[0][2] is not None
    assert str(requests[0][2]["relative_path"]).endswith(".onnx")
    assert not Path(result.detail["invalid_source_path"]).exists()

