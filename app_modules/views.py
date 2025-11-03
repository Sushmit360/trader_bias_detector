from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.pipeline import RiskBundle
from .constants import SAMPLE_DATASETS, AI_PROMPT
from .data import (
    generate_from_uploads,
    load_sample_dataset,
    parse_upload,
    tidy_dataframe,
)


def render_ai_prompt() -> None:
    st.subheader("Need synthetic data quickly?", anchor="ai-prompt")
    st.write(
        "For convenience, here's a ready-to-use AI prompt you can drop into any LLM to generate "
        "compatible `trades.csv` and `daily.csv` files tailored to this scoring pipeline. "
        "Set `[TARGET_RISK_TIER]` if you want to bias the behaviour."
    )
    st.markdown(
        """
        <style>
        .prompt-code div[data-testid="stCodeBlock"] {
            max-height: 220px;
            overflow-y: auto !important;
        }
        .prompt-code div[data-testid="stCodeBlock"] pre {
            max-height: 220px;
            overflow-y: auto !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.container():
        st.markdown('<div class="prompt-code">', unsafe_allow_html=True)
        st.code(AI_PROMPT.strip(), language="text")
        st.markdown("</div>", unsafe_allow_html=True)


def render_samples_preview() -> None:
    st.markdown("#### Sample Files Preview")
    tabs = st.tabs(list(SAMPLE_DATASETS.keys()))
    for tab, label in zip(tabs, SAMPLE_DATASETS.keys()):
        with tab:
            paths = SAMPLE_DATASETS[label]
            st.caption(SAMPLE_DATASETS[label]["description"])
            st.markdown("**Trades snapshot (first 5 rows)**")
            trades_df = pd.read_csv(paths["trades"]).head(5)
            st.dataframe(
                tidy_dataframe(
                    trades_df,
                    rename_map={
                        "trader_id": "Trader ID",
                        "timestamp": "Timestamp",
                        "date": "Date",
                        "side": "Side",
                        "qty": "Quantity",
                        "price": "Price",
                        "fee": "Fee",
                    },
                ),
                use_container_width=True,
            )
            if paths.get("daily"):
                st.markdown("**Daily performance (first 5 rows)**")
                daily_df = pd.read_csv(paths["daily"]).head(5)
                st.dataframe(
                    tidy_dataframe(
                        daily_df,
                        rename_map={
                            "trader_id": "Trader ID",
                            "date": "Date",
                            "portfolio_value": "Portfolio Value",
                        },
                    ),
                    use_container_width=True,
                )
            bundle_preview = load_sample_dataset(label)
            risk_stats = bundle_preview.scores["risk_score"]
            st.markdown(
                f"**Risk score span:** {risk_stats.min():.1f} → {risk_stats.max():.1f}"
            )


def render_upload_section() -> None:
    st.subheader("1. Upload your files")
    st.write(
        "Provide at least a `trades` file with columns like `trader_id`, `timestamp`, `side`, `qty`, and `price`. "
        "Optionally add a `daily` portfolio snapshot (`trader_id`, `date`, `portfolio_value`) for richer drawdown analytics."
    )
    st.markdown(
        "Need a quick dataset? [Jump to the synthetic data prompt](#ai-prompt)."
    )
    if st.session_state.get("warning_message"):
        st.warning(st.session_state["warning_message"])

    col_left, col_right = st.columns(2)
    with col_left:
        trades_file = st.file_uploader(
            "Trades file (CSV or Excel)",
            type=["csv", "xlsx", "xls"],
            key="trades_uploader",
        )
        trades_df = parse_upload(trades_file)
        if trades_df is not None:
            st.session_state["uploaded_trades"] = trades_df
            st.success(f"Loaded {trades_file.name} ({len(trades_df)} rows)")
        elif trades_file is None:
            st.session_state["uploaded_trades"] = None
    with col_right:
        daily_file = st.file_uploader(
            "Daily portfolio file (optional)",
            type=["csv", "xlsx", "xls"],
            key="daily_uploader",
        )
        daily_df = parse_upload(daily_file)
        if daily_df is not None:
            st.session_state["uploaded_daily"] = daily_df
            st.success(f"Loaded {daily_file.name} ({len(daily_df)} rows)")
        elif daily_file is None:
            st.session_state["uploaded_daily"] = None

    st.subheader("2. Run the scoring pipeline")
    st.markdown(
        """
        <style>
        .primary-action button[kind="primary"] {
            background: linear-gradient(135deg, #1E88E5, #42A5F5) !important;
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    run_container = st.container()
    with run_container:
        st.markdown('<div class="primary-action">', unsafe_allow_html=True)
        if st.button(
            "Generate Risk Scores",
            key="generate_scores",
            type="primary",
            use_container_width=True,
        ):
            ran, _ = generate_from_uploads()
            if ran:
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("3. Need examples instead?")
    st.write(
        "Use one of the curated samples to explore the dashboards instantly. "
        "Each pack includes trades and daily portfolio values."
    )
    sample_cols = st.columns(len(SAMPLE_DATASETS))
    for i, label in enumerate(SAMPLE_DATASETS.keys()):
        with sample_cols[i]:
            st.caption(SAMPLE_DATASETS[label]["description"])
            st.markdown('<div class="primary-action">', unsafe_allow_html=True)
            if st.button(f"Explore {label}", key=f"sample_{label}", type="primary"):
                bundle = load_sample_dataset(label)
                st.session_state["bundle"] = bundle
                st.session_state["bundle_label"] = label
                st.session_state["active_view"] = "dashboard"
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)


def render_instructions() -> None:
    st.title("Behavioral Bias Detector")
    st.markdown(
        """
        Discover trading behaviors, risk tiers, and tailored interventions. Upload raw trades (and optional daily
        performance) and the app will engineer features, score each trader, surface alerts, and render all dashboards.
        """
    )

    with st.expander("What files do I need?"):
        st.markdown(
            """
            - **Trades** (required): CSV/XLSX with at least `trader_id`, `timestamp` (or `date`), `side`, `qty`, `price`.<br>
            - **Daily portfolio** (optional): CSV/XLSX with `trader_id`, `date`, `portfolio_value` for drawdown context.<br>
            - Filenames are flexible; the app reads anything you upload.<br>
            - Multiple tabs? Stick to one sheet per Excel file with a header row.
            """,
            unsafe_allow_html=True,
        )

    with st.expander("How does scoring work?"):
        st.markdown(
            """
            1. Trades are aggregated into daily activity per trader.<br>
            2. Feature engineering extracts trade velocity, turnover, activity consistency, and momentum chasing proxies.<br>
            3. Optional daily portfolio curves provide drawdown severity.<br>
            4. Behavioral scores (overtrading, loss aversion, herding) are blended into a 0–100 risk score and tiered.<br>
            5. Alerts highlight the riskiest traders with driver-specific coaching tips.
            """,
            unsafe_allow_html=True,
        )

    st.divider()


def render_landing() -> None:
    render_instructions()
    render_upload_section()
    st.divider()
    render_samples_preview()
    st.divider()
    render_ai_prompt()


def overview_tab(bundle: RiskBundle) -> None:
    df = bundle.scores.merge(bundle.features, on="trader_id", how="left")
    st.subheader(f"Overview — {st.session_state.get('bundle_label')}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Traders scored", f"{len(df):,}")
    col2.metric("Avg risk score", f"{df['risk_score'].mean():.1f}")
    tier_counts = df["tier"].value_counts()
    col3.metric("High risk", str(int(tier_counts.get("High", 0))))
    col4.metric("Medium risk", str(int(tier_counts.get("Medium", 0))))

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.caption("Risk score distribution")
        hist = px.histogram(df, x="risk_score", nbins=30)
        st.plotly_chart(hist, use_container_width=True)
    with chart_cols[1]:
        st.caption("Behavioral score mix")
        radar = px.line_polar(
            df[["overtrading", "loss_aversion", "herding"]]
            .mean()
            .reset_index()
            .rename(columns={"index": "trait", 0: "score"}),
            r="score",
            theta="trait",
            line_close=True,
        )
        radar.update_traces(fill="toself")
        st.plotly_chart(radar, use_container_width=True)

    st.caption("Top 15 traders by risk")
    top = df.sort_values("risk_score", ascending=False).head(15)
    top_display = tidy_dataframe(
        top[
            [
                "trader_id",
                "risk_score",
                "tier",
                "trades_per_active_day",
                "turnover",
                "avg_down_events_rate",
                "buy_after_spike_rate",
            ]
        ],
        rename_map={
            "trader_id": "Trader ID",
            "risk_score": "Risk Score",
            "tier": "Risk Tier",
            "trades_per_active_day": "Trades / Active Day",
            "turnover": "Turnover (Notional)",
            "avg_down_events_rate": "Drawdown Severity",
            "buy_after_spike_rate": "Momentum Buy Rate",
        },
        percent_cols=["buy_after_spike_rate", "avg_down_events_rate"],
    )
    st.dataframe(top_display, use_container_width=True, hide_index=True)

    dl_cols = st.columns(3)
    with dl_cols[0]:
        st.download_button(
            "Download risk scores",
            data=df[
                ["trader_id", "risk_score", "tier", "overtrading", "loss_aversion", "herding"]
            ]
            .round(4)
            .to_csv(index=False),
            file_name="risk_scores.csv",
            mime="text/csv",
        )
    with dl_cols[1]:
        st.download_button(
            "Download engineered features",
            data=bundle.features.round(4).to_csv(index=False),
            file_name="engineered_features.csv",
            mime="text/csv",
        )
    with dl_cols[2]:
        st.download_button(
            "Download alerts",
            data=bundle.alerts.round(4).to_csv(index=False),
            file_name="alerts.csv",
            mime="text/csv",
        )


def segments_tab(bundle: RiskBundle) -> None:
    st.subheader("Segments Explorer")
    df = bundle.scores.merge(bundle.features, on="trader_id", how="left")

    st.caption("Filter by risk thresholds")
    col1, col2 = st.columns(2)
    with col1:
        min_score = st.slider("Minimum risk score", 0, 100, 40)
    with col2:
        tiers = st.multiselect(
            "Risk tiers",
            options=["Low", "Medium", "High"],
            default=["Low", "Medium", "High"],
        )
    segment_df = df[(df["risk_score"] >= min_score) & (df["tier"].isin(tiers))]

    segment_df = segment_df.assign(
        bubble_size=segment_df["trades_per_active_day"].clip(lower=0.1) * 10
    )
    scatter = px.scatter(
        segment_df,
        x="turnover",
        y="max_drawdown",
        color="tier",
        size="bubble_size",
        hover_data=[
            "trader_id",
            "risk_score",
            "trades_per_active_day",
            "pct_days_traded",
        ],
    )
    scatter.update_layout(
        xaxis_title="Turnover (Σ|qty×price|)",
        yaxis_title="Max drawdown magnitude",
    )
    st.plotly_chart(scatter, use_container_width=True)

    st.caption("Download current segment")
    st.download_button(
        "Download CSV",
        data=segment_df.round(4).to_csv(index=False),
        file_name="segment_view.csv",
        mime="text/csv",
    )


def trader_detail_tab(bundle: RiskBundle) -> None:
    st.subheader("Trader Detail")
    df = bundle.scores.merge(bundle.features, on="trader_id", how="left")
    trader_id = st.selectbox("Select a trader", sorted(df["trader_id"].unique()))
    row = df[df["trader_id"] == trader_id].iloc[0]

    st.markdown(
        f"### Trader **{trader_id}** — Risk Score **{row['risk_score']:.1f}** ({row['tier']})"
    )

    gauges = st.columns(3)
    traits = [("overtrading", "Overtrading"), ("loss_aversion", "Loss Aversion"), ("herding", "Herding")]
    for col, (field, title) in zip(gauges, traits):
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=float(row[field]),
                gauge={"axis": {"range": [0, 1]}},
                title={"text": title},
            )
        )
        col.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Engineered features")
    feature_cols = [
        "trades_per_active_day",
        "turnover",
        "pct_days_traded",
        "orders_burstiness",
        "avg_hold_days",
        "avg_down_events_rate",
        "buy_after_spike_rate",
        "max_drawdown",
    ]
    feature_labels = {
        "trades_per_active_day": "Trades / Active Day",
        "turnover": "Turnover (Notional)",
        "pct_days_traded": "Active Days (%)",
        "orders_burstiness": "Orders Burstiness",
        "avg_hold_days": "Avg Hold Days",
        "avg_down_events_rate": "Drawdown Severity (%)",
        "buy_after_spike_rate": "Momentum Buy Rate (%)",
        "max_drawdown": "Max Drawdown (%)",
    }
    percent_like = {
        "pct_days_traded",
        "avg_down_events_rate",
        "buy_after_spike_rate",
        "max_drawdown",
    }
    feature_rows = []
    for col in feature_cols:
        val = row[col]
        if col in percent_like:
            val = float(val) * 100
        feature_rows.append(
            {
                "Metric": feature_labels.get(col, col),
                "Metric Value": round(float(val), 2),
            }
        )
    feature_table = pd.DataFrame(feature_rows)
    st.dataframe(feature_table, use_container_width=True, hide_index=True)

    st.markdown("#### Trade timeline")
    trades_df = bundle.trades[bundle.trades["trader_id"] == trader_id].copy()
    if not trades_df.empty:
        if "timestamp" in trades_df.columns:
            trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"], errors="coerce")
            trades_df = trades_df.dropna(subset=["timestamp"])
            trades_df["date"] = trades_df["timestamp"].dt.date
        daily_counts = (
            trades_df.groupby(["date", "side"])
            .size()
            .reset_index(name="trades")
        )
        time_chart = px.bar(daily_counts, x="date", y="trades", color="side")
        st.plotly_chart(time_chart, use_container_width=True)
    else:
        st.info("No trades found for this trader in the current dataset.")


def alerts_tab(bundle: RiskBundle) -> None:
    st.subheader("Alerts & Suggested Actions")
    st.write("Traders sorted by highest risk. Reasons highlight their primary behavioural driver.")
    alerts_display = tidy_dataframe(
        bundle.alerts[
            ["trader_id", "risk_score", "overtrading", "loss_aversion", "herding", "reasons"]
        ],
        rename_map={
            "trader_id": "Trader ID",
            "risk_score": "Risk Score",
            "overtrading": "Overtrading Score",
            "loss_aversion": "Loss Aversion Score",
            "herding": "Herding Score",
            "reasons": "Primary Driver",
        },
    )
    st.dataframe(alerts_display, use_container_width=True, hide_index=True)


def render_dashboard() -> None:
    bundle: Optional[RiskBundle] = st.session_state.get("bundle")
    if bundle is None:
        st.info("Upload data and generate scores to see the dashboards.")
        return

    st.markdown(
        """
        <style>
        .top-nav-title {
            font-size: 1.6rem;
            font-weight: 600;
            text-align: center;
            flex-grow: 1;
            color: #1565C0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns([1, 2, 1])
    with cols[0]:
        if st.button("← Back to Home", key="nav_home", type="primary"):
            st.session_state["active_view"] = "landing"
            st.session_state["bundle"] = None
            st.session_state["bundle_label"] = ""
            st.rerun()
    with cols[1]:
        st.markdown(
            '<div class="top-nav-title">Behavioral Bias Dashboard</div>',
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.empty()

    tabs = st.tabs(["Overview", "Segments", "Trader Detail", "Alerts"])
    with tabs[0]:
        overview_tab(bundle)
    with tabs[1]:
        segments_tab(bundle)
    with tabs[2]:
        trader_detail_tab(bundle)
    with tabs[3]:
        alerts_tab(bundle)
