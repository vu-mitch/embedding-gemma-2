"""The manufacturing demo renders timelines, rejects the 15s miss, and shows metrics."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "app.py"


def _click(at: AppTest, label: str) -> None:
    for button in at.button:
        if button.label == label:
            button.click()
            return
    raise AssertionError(label)


def _markdown(at: AppTest) -> str:
    return "\n".join(el.value for el in at.markdown)


def test_manufacturing_demo_and_evaluation():
    app = AppTest.from_file(str(APP), default_timeout=60)
    app.run()
    assert not app.exception

    _click(app, "Load manufacturing example")
    app.run()
    assert not app.exception
    body = _markdown(app)
    assert "Line A · figure timeline" in body
    assert "Abnormal vibration" in body
    assert "does not hold" in body
    assert "begins 30.0s before" in body
    assert "inside the 15s window" in body
    assert "**2** of **3** recordings satisfy every constraint." in body
    # Semantic search still returns the wide windows, including the line that fails the window.
    assert "00:00-00:50" in body
    assert "03:00-04:00" in body

    _click(app, "Run temporal evaluation")
    app.run()
    assert not app.exception
    frame = app.dataframe[0].value
    temporal = dict(zip(frame["metric"], frame["temporal pipeline"]))
    semantic = dict(zip(frame["metric"], frame["semantic baseline"]))
    assert temporal["Temporal IoU"] == 1
    assert temporal["Recall@5"] == 1
    assert temporal["Event-ordering accuracy"] == 1
    assert temporal["Answer correctness"] == 1
    assert all(temporal[name] > semantic[name] for name in temporal)
