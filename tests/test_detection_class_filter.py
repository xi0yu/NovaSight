import pytest

from novasight.config.runtime import parse_runtime_config
from novasight.runtime.candidates import parse_allowed_class_ids


def test_detection_class_filter_accepts_multiple_classes() -> None:
    cfg = parse_runtime_config({"inference": {"detection_class_filter": "1,0,3"}})

    assert cfg.inference.detection_class_filter == "1,0,3"
    assert parse_allowed_class_ids(cfg.inference.detection_class_filter) == {0, 1, 3}
    assert parse_allowed_class_ids("all") is None


@pytest.mark.parametrize("value", ["", "1,head", "1,256"])
def test_detection_class_filter_rejects_invalid_multi_select_values(value: str) -> None:
    with pytest.raises(ValueError, match="detection_class_filter"):
        parse_runtime_config({"inference": {"detection_class_filter": value}})
