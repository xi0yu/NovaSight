from pathlib import Path


STUDIO_CONSOLE = (
    Path(__file__).resolve().parents[1]
    / "web"
    / "src"
    / "features"
    / "studio"
    / "StudioConsoleView.tsx"
)


def _roi_card_source() -> str:
    source = STUDIO_CONSOLE.read_text(encoding="utf-8")
    start = source.index('<h2 className="console-title">ROI 裁剪</h2>')
    end = source.index('<NumberControl label="水平偏移"', start)
    return source[start:end]


def test_roi_size_control_commits_once_instead_of_writing_on_every_change() -> None:
    roi_card = _roi_card_source()

    assert 'onChange={(event) => void updateConfigField("roi", "size"' not in roi_card
    assert 'onCommit={(value) => updateConfigField("roi", "size", nearestRoiSize(value))}' in roi_card
