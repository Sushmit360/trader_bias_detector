import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class RiskBundle:
    trades: pd.DataFrame
    daily: Optional[pd.DataFrame]
    features: pd.DataFrame
    scores: pd.DataFrame
    alerts: pd.DataFrame


def _ensure_datetime(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column in df.columns and not pd.api.types.is_datetime64_any_dtype(df[column]):
        df = df.copy()
        df[column] = pd.to_datetime(df[column], errors="coerce")
        df = df.dropna(subset=[column])
    return df


def _prepare_trades(trades: pd.DataFrame) -> pd.DataFrame:
    df = trades.copy()
    if "trade_id" not in df.columns:
        df = df.reset_index(drop=True)
        df["trade_id"] = np.arange(1, len(df) + 1)
    if "timestamp" in df.columns:
        df = _ensure_datetime(df, "timestamp")
        df["date"] = df["timestamp"].dt.date
    elif "date" in df.columns:
        df = _ensure_datetime(df, "date")
        df["date"] = df["date"].dt.date
    else:
        raise ValueError("Trades data must include either 'timestamp' or 'date'.")
    df["notional"] = df.get("qty", 0).abs() * df.get("price", 0).abs()
    return df


def _prepare_daily(daily: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if daily is None:
        return None
    if daily.empty:
        return None
    df = daily.copy()
    df = _ensure_datetime(df, "date")
    return df


def compute_features(trades: pd.DataFrame, daily: Optional[pd.DataFrame]) -> pd.DataFrame:
    trades = _prepare_trades(trades)

    grouped = trades.groupby(["trader_id", "date"]).agg(
        trades_per_day=("trade_id", "count"),
        total_notional=("notional", "sum"),
        buys=("side", lambda x: (x.astype(str).str.upper() == "B").sum()),
        sells=("side", lambda x: (x.astype(str).str.upper() == "S").sum()),
    )
    grouped = grouped.reset_index()

    features = []
    for trader_id, trader_df in grouped.groupby("trader_id"):
        active_days = len(trader_df)
        total_days = (trader_df["date"].max() - trader_df["date"].min()).days + 1
        total_notional = trader_df["total_notional"].sum()
        mean_trades = trader_df["trades_per_day"].mean()
        std_trades = trader_df["trades_per_day"].std(ddof=0)

        pct_days_traded = active_days / total_days if total_days > 0 else 0
        burstiness = std_trades / mean_trades if mean_trades > 0 else 0

        daily_notional = trader_df[["date", "total_notional"]].sort_values("date")
        if len(daily_notional) > 1:
            diffs = daily_notional["date"].diff().dt.days.fillna(0)
            avg_hold_days = diffs[diffs > 0].mean() or 0
        else:
            avg_hold_days = 0

        heavy_buy_days = trader_df["total_notional"].quantile(0.9)
        if heavy_buy_days == 0:
            heavy_buy_rate = 0
        else:
            heavy_buy_rate = (
                trader_df.loc[trader_df["total_notional"] >= heavy_buy_days, "buys"].sum()
                / trader_df["buys"].sum()
                if trader_df["buys"].sum()
                else 0
            )

        features.append(
            {
                "trader_id": trader_id,
                "trades_per_active_day": mean_trades,
                "turnover": total_notional,
                "pct_days_traded": pct_days_traded,
                "orders_burstiness": burstiness,
                "avg_hold_days": avg_hold_days or 0,
                "buy_after_spike_rate": heavy_buy_rate,
            }
        )

    features_df = pd.DataFrame(features)

    if daily is not None:
        daily = _prepare_daily(daily)
        if daily is not None:
            daily_sorted = daily.sort_values(["trader_id", "date"])
            drawdowns = (
                daily_sorted.groupby("trader_id")["portfolio_value"]
                .apply(_compute_drawdown)
                .rename("max_drawdown")
                .reset_index()
            )
            features_df = features_df.merge(drawdowns, on="trader_id", how="left")
        else:
            features_df["max_drawdown"] = np.nan
    else:
        features_df["max_drawdown"] = np.nan

    features_df["max_drawdown"] = features_df["max_drawdown"].abs()
    features_df["avg_down_events_rate"] = features_df["max_drawdown"]
    return features_df.fillna(0)


def _compute_drawdown(values: pd.Series) -> float:
    roll_max = values.cummax()
    drawdown = (values / roll_max) - 1.0
    return float(drawdown.min()) if len(drawdown) else 0.0


MODEL_FEATURES: List[str] = [
    "trades_per_active_day",
    "turnover",
    "pct_days_traded",
    "orders_burstiness",
    "avg_hold_days",
    "avg_down_events_rate",
    "buy_after_spike_rate",
    "max_drawdown",
]


def score_traders(features: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()
    model_inputs = df[MODEL_FEATURES].fillna(0.0)
    base_signal = (
        0.55 * df["trades_per_active_day"].values
        + 0.25 * df["avg_down_events_rate"].values
        + 0.20 * df["buy_after_spike_rate"].values
    )
    # time-split style heuristic: later traders considered "future"
    rank_order = df["trader_id"].rank(method="dense", ascending=True).values
    threshold = np.median(base_signal + 0.05 * (rank_order / rank_order.max()))
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 0.1, size=len(df))
    labels = ((base_signal + noise) > threshold).astype(int)

    if len(np.unique(labels)) > 1 and len(df) > 1:
        logistic_pipeline = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("logreg", LogisticRegression(max_iter=500, class_weight="balanced")),
            ]
        )
        logistic_pipeline.fit(model_inputs, labels)
        probabilities = logistic_pipeline.predict_proba(model_inputs)[:, 1]
    else:
        scaled_signal = (base_signal - base_signal.min())
        if scaled_signal.max() > 0:
            probabilities = scaled_signal / scaled_signal.max()
        else:
            probabilities = np.full(len(df), 0.5)

    overtrade_den = max(df["trades_per_active_day"].max(), 1e-6)
    df["overtrading"] = np.clip(
        df["trades_per_active_day"] / overtrade_den, 0, 1
    )
    loss_den = max(df["avg_down_events_rate"].max(), 1e-6)
    herd_den = max(df["buy_after_spike_rate"].max(), 1e-6)
    df["loss_aversion"] = np.clip(
        df["avg_down_events_rate"] / loss_den, 0, 1
    )
    df["herding"] = np.clip(
        df["buy_after_spike_rate"] / herd_den, 0, 1
    )

    df["risk_score"] = np.clip(probabilities, 0, 1) * 100
    df["risk_score"] = df["risk_score"].round(2)
    df["tier"] = pd.cut(
        df["risk_score"],
        bins=[-math.inf, 30, 60, math.inf],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    return df[
        [
            "trader_id",
            "overtrading",
            "loss_aversion",
            "herding",
            "risk_score",
            "tier",
        ]
    ]


def build_alerts(scores: pd.DataFrame) -> pd.DataFrame:
    df = scores.copy()
    high_risk = df[df["risk_score"] >= 60].copy()
    if high_risk.empty:
        high_risk = df.nlargest(10, "risk_score").copy()

    reasons = []
    for _, row in high_risk.iterrows():
        drivers: Dict[str, Any] = {
            "overtrading": row["overtrading"],
            "loss_aversion": row["loss_aversion"],
            "herding": row["herding"],
        }
        sorted_drivers = sorted(drivers.items(), key=lambda x: x[1], reverse=True)
        top_driver = sorted_drivers[0][0]
        if top_driver == "overtrading":
            reason = "High trade velocity vs. peers"
        elif top_driver == "loss_aversion":
            reason = "Holding losers too long"
        else:
            reason = "Momentum chasing patterns"
        reasons.append(reason)

    high_risk["reasons"] = reasons
    return high_risk.reset_index(drop=True)


def run_pipeline(trades: pd.DataFrame, daily: Optional[pd.DataFrame]) -> RiskBundle:
    prepared_trades = _prepare_trades(trades)
    prepared_daily = _prepare_daily(daily)
    features = compute_features(prepared_trades, prepared_daily)
    scores = score_traders(features)
    alerts = build_alerts(scores)
    return RiskBundle(
        trades=prepared_trades,
        daily=prepared_daily,
        features=features,
        scores=scores,
        alerts=alerts,
    )
