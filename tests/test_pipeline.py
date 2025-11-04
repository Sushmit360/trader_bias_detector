import math
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from core.pipeline import (
    MODEL_FEATURES,
    RiskBundle,
    _compute_drawdown,
    _ensure_datetime,
    _prepare_daily,
    _prepare_trades,
    build_alerts,
    compute_features,
    run_pipeline,
    score_traders,
)


def make_trades(
    trader_id: str = "T1",
    timestamps: Optional[list[str]] = None,
    sides: Optional[list[str]] = None,
    qtys: Optional[list[float]] = None,
    prices: Optional[list[float]] = None,
) -> pd.DataFrame:
    if timestamps is None:
        timestamps = [
            "2024-01-02T10:00:00Z",
            "2024-01-03T11:00:00Z",
            "2024-01-03T12:00:00Z",
        ]
    if sides is None:
        sides = ["B", "S", "B"]
    if qtys is None:
        qtys = [10, 5, 2]
    if prices is None:
        prices = [100, 110, 95]
    return pd.DataFrame(
        {
            "trader_id": trader_id,
            "timestamp": timestamps,
            "side": sides,
            "qty": qtys,
            "price": prices,
        }
    )


def make_daily(trader_id: str = "T1", values: Optional[list[float]] = None) -> pd.DataFrame:
    if values is None:
        values = [1000, 950, 975]
    dates = pd.date_range("2024-01-02", periods=len(values), freq="D")
    return pd.DataFrame({"trader_id": trader_id, "date": dates, "portfolio_value": values})


def test_prepare_trades_adds_missing_columns():
    trades = make_trades()
    prepared = _prepare_trades(trades)

    assert "trade_id" in prepared.columns
    assert prepared["trade_id"].is_unique
    assert "date" in prepared.columns
    assert prepared["date"].dtype == object  # datetime.date results in object dtype
    np.testing.assert_allclose(prepared["notional"], np.abs(trades["qty"] * trades["price"]))


def test_prepare_trades_requires_date_or_timestamp():
    trades = pd.DataFrame({"trader_id": ["T1"], "side": ["B"], "qty": [1], "price": [10]})
    with pytest.raises(ValueError):
        _prepare_trades(trades)


def test_prepare_trades_works_with_date_column_only():
    trades = pd.DataFrame(
        {
            "trader_id": ["T1", "T1"],
            "date": ["2024-01-02", "2024-01-03"],
            "side": ["B", "S"],
            "qty": [5, 7],
            "price": [100, 105],
        }
    )
    prepared = _prepare_trades(trades)
    assert "timestamp" not in prepared.columns
    assert prepared["date"].tolist() == [pd.Timestamp("2024-01-02").date(), pd.Timestamp("2024-01-03").date()]
    np.testing.assert_allclose(prepared["notional"], [500, 735])


def test_ensure_datetime_drops_invalid_rows():
    df = pd.DataFrame({"timestamp": ["2024-01-02T10:00:00Z", "invalid"], "value": [1, 2]})
    cleaned = _ensure_datetime(df, "timestamp")
    assert len(cleaned) == 1
    assert pd.api.types.is_datetime64_any_dtype(cleaned["timestamp"])


def test_prepare_daily_handles_none_and_empty():
    assert _prepare_daily(None) is None
    assert _prepare_daily(pd.DataFrame()) is None

    daily = make_daily()
    prepared = _prepare_daily(daily)
    assert prepared is not None
    assert pd.api.types.is_datetime64_any_dtype(prepared["date"])


def test_compute_drawdown_returns_expected_value():
    values = pd.Series([100, 105, 90, 120], dtype=float)
    drawdown = _compute_drawdown(values)
    assert math.isclose(drawdown, -0.1428571428571429)


def test_compute_features_produces_expected_columns():
    trades = make_trades()
    daily = make_daily()
    features = compute_features(trades, daily)

    assert set(MODEL_FEATURES).issubset(features.columns)
    assert features["trader_id"].tolist() == ["T1"]
    assert features.loc[0, "max_drawdown"] >= 0
    # avg_down_events_rate mirrors max_drawdown
    assert features.loc[0, "avg_down_events_rate"] == features.loc[0, "max_drawdown"]


def test_compute_features_multiple_traders_calculations():
    trades = pd.concat(
        [
            make_trades("T1"),
            make_trades(
                "T2",
                timestamps=[
                    "2024-01-02T09:00:00Z",
                    "2024-01-04T09:30:00Z",
                ],
                sides=["B", "S"],
                qtys=[3, 4],
                prices=[50, 60],
            ),
        ],
        ignore_index=True,
    )
    features = compute_features(trades, None)
    assert sorted(features["trader_id"]) == ["T1", "T2"]
    # Ensure burstiness well-defined (no NaN)
    assert features["orders_burstiness"].notna().all()


def test_compute_features_without_daily_sets_drawdown_zero():
    trades = make_trades()
    features = compute_features(trades, None)
    assert "max_drawdown" in features
    assert features.loc[0, "max_drawdown"] == 0


def test_score_traders_shapes_and_ranges():
    trades = make_trades()
    daily = make_daily()
    features = compute_features(trades, daily)
    scores = score_traders(features)

    expected_cols = {"trader_id", "overtrading", "loss_aversion", "herding", "risk_score", "tier"}
    assert expected_cols.issubset(scores.columns)
    assert scores["risk_score"].between(0, 100).all()
    assert set(scores["tier"]).issubset({"Low", "Medium", "High"})


def test_score_traders_fallback_path(monkeypatch):
    class ZeroNoiseRng:
        def normal(self, loc, scale, size):
            return np.zeros(size)

    monkeypatch.setattr("core.pipeline.np.random.default_rng", lambda seed: ZeroNoiseRng())
    feature_block = pd.DataFrame(
        {
            "trader_id": ["T1", "T2"],
            "trades_per_active_day": [1, 1],
            "turnover": [100, 100],
            "pct_days_traded": [1, 1],
            "orders_burstiness": [0, 0],
            "avg_hold_days": [1, 1],
            "avg_down_events_rate": [0, 0],
            "buy_after_spike_rate": [0, 0],
            "max_drawdown": [0, 0],
        }
    )
    scores = score_traders(feature_block)
    # Fallback assigns probability 0.5 for identical signals -> risk score 50
    np.testing.assert_array_equal(scores["risk_score"], np.array([50.0, 50.0]))
    assert set(scores["tier"]) == {"Medium"}


def test_build_alerts_returns_reason_for_high_risk():
    # Construct scores with two traders so the top driver logic triggers
    features = pd.DataFrame(
        {
            "trader_id": ["T1", "T2"],
            "trades_per_active_day": [20, 1],
            "turnover": [100000, 2000],
            "pct_days_traded": [1.0, 0.2],
            "orders_burstiness": [0.5, 0.1],
            "avg_hold_days": [0.5, 3.0],
            "avg_down_events_rate": [0.9, 0.2],
            "buy_after_spike_rate": [0.6, 0.1],
            "max_drawdown": [0.9, 0.2],
        }
    )
    scores = score_traders(features)
    alerts = build_alerts(scores)

    assert not alerts.empty
    assert "reasons" in alerts.columns
    # best driver should map to an expected narration
    valid_reasons = {
        "High trade velocity vs. peers",
        "Holding losers too long",
        "Momentum chasing patterns",
    }
    assert set(alerts["reasons"]).issubset(valid_reasons)


def test_build_alerts_uses_top_n_when_no_high_risk():
    rows = []
    for idx in range(12):
        rows.append(
            {
                "trader_id": f"T{idx}",
                "overtrading": 0.1,
                "loss_aversion": 0.1,
                "herding": 0.1,
                "risk_score": 20 + idx,
                "tier": "Low",
            }
        )
    scores = pd.DataFrame(rows)
    alerts = build_alerts(scores)
    assert len(alerts) == 10
    assert alerts["risk_score"].is_monotonic_decreasing


def test_run_pipeline_returns_riskbundle():
    trades = make_trades()
    daily = make_daily()
    bundle = run_pipeline(trades, daily)

    assert isinstance(bundle, RiskBundle)
    assert {"trades", "daily", "features", "scores", "alerts"} <= set(bundle.__dict__.keys())
    assert not bundle.trades.empty
    assert bundle.scores["trader_id"].tolist() == ["T1"]


def test_run_pipeline_without_daily():
    trades = make_trades()
    bundle = run_pipeline(trades, None)
    assert bundle.daily is None
    assert "max_drawdown" in bundle.features.columns
