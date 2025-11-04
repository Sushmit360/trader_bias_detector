from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd
import pytest

import app_modules.data as data_mod
from core.pipeline import RiskBundle


@dataclass
class StubStreamlit:
    session_state: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    successes: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def warning(self, message: str) -> None:
        self.warnings.append(message)

    def success(self, message: str) -> None:
        self.successes.append(message)

    def error(self, message: str) -> None:
        self.errors.append(message)


def use_stub_streamlit(monkeypatch: pytest.MonkeyPatch) -> StubStreamlit:
    stub = StubStreamlit()
    monkeypatch.setattr(data_mod, "st", stub)
    return stub


def test_tidy_dataframe_applies_formatting():
    df = pd.DataFrame(
        {
            "trader_id": ["T1", "T2"],
            "risk_score": [0.756, 0.123],
            "buy_after_spike_rate": [0.4, 0.25],
        }
    )
    renamed = data_mod.tidy_dataframe(
        df,
        rename_map={"trader_id": "Trader", "risk_score": "Risk Score"},
        percent_cols=["buy_after_spike_rate"],
    )

    expected_cols = {"Trader", "Risk Score", "Buy After Spike Rate"}
    assert expected_cols == set(renamed.columns)
    assert list(renamed["Risk Score"]) == [0.76, 0.12]
    assert list(renamed["Buy After Spike Rate"]) == [40.0, 25.0]


def test_load_sample_dataset_produces_bundle():
    bundle = data_mod.load_sample_dataset("Sample Pack A")
    assert isinstance(bundle, RiskBundle)
    assert not bundle.trades.empty
    assert not bundle.scores.empty
    assert "risk_score" in bundle.scores.columns


def test_parse_upload_returns_none_when_missing(monkeypatch):
    stub = use_stub_streamlit(monkeypatch)
    assert data_mod.parse_upload(None) is None
    assert stub.errors == []


def test_parse_upload_reads_csv(monkeypatch):
    stub = use_stub_streamlit(monkeypatch)
    csv_bytes = io.StringIO("a,b\n1,2\n3,4\n")
    setattr(csv_bytes, "name", "demo.csv")
    df = data_mod.parse_upload(csv_bytes)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["a", "b"]
    assert stub.errors == []


def test_parse_upload_handles_errors(monkeypatch):
    stub = use_stub_streamlit(monkeypatch)

    class FakeFile(io.BytesIO):
        name = "broken.csv"

    fake_file = FakeFile(b"")

    def boom(*args, **kwargs):
        raise ValueError("cannot parse")

    monkeypatch.setattr(pd, "read_csv", boom)
    df = data_mod.parse_upload(fake_file)
    assert df is None
    assert stub.errors == ["Could not read broken.csv: cannot parse"]


def test_generate_from_uploads_requires_trades(monkeypatch):
    stub = use_stub_streamlit(monkeypatch)
    stub.session_state.update({"uploaded_trades": None, "uploaded_daily": None})
    status, message = data_mod.generate_from_uploads()
    assert not status
    assert message
    assert stub.warnings == [message]


def test_generate_from_uploads_runs_pipeline(monkeypatch):
    stub = use_stub_streamlit(monkeypatch)
    trades = pd.DataFrame(
        {
            "trader_id": ["T1", "T1"],
            "timestamp": ["2024-01-02T10:00:00Z", "2024-01-03T11:00:00Z"],
            "side": ["B", "S"],
            "qty": [10, 5],
            "price": [100, 110],
        }
    )
    stub.session_state.update(
        {
            "uploaded_trades": trades,
            "uploaded_daily": None,
            "bundle": None,
            "bundle_label": "",
            "active_view": "landing",
            "warning_message": "",
        }
    )
    status, message = data_mod.generate_from_uploads()
    assert status
    assert message is None
    assert isinstance(stub.session_state["bundle"], RiskBundle)
    assert stub.session_state["bundle_label"] == "Your Upload"
    assert stub.session_state["active_view"] == "dashboard"
    assert stub.successes == ["Scoring complete! Jump to the dashboard below."]
