# Behavioral Bias Detector — README

## 1. Vision

Modern brokerages and wealth platforms need early warning indicators of harmful trading habits. The Behavioral Bias Detector delivers those signals by ingesting raw trade logs, engineering bias-specific features, running a calibrated risk model, and surfacing transparent dashboards for surveillance and coaching. The product is designed to be BYOD (bring your own data): analysts upload trades and optional daily portfolio valuations, run the on-the-fly scoring pipeline, and explore alerts, segment scatterplots, and per-trader drill-downs.

## 2. System Architecture

```
┌─────────────────────┐
│   Streamlit App     │
│  (app.py entrypoint)│
└─────────┬───────────┘
          │ imports
┌─────────────────────┐
│   app_modules       │
│  ├─ constants.py    │  → Prompt text, bundled samples
│  ├─ state.py        │  → Session-state initialisation
│  ├─ data.py         │  → Upload parsing, pipeline orchestration
│  └─ views.py        │  → UI components (landing, dashboards)
└─────────┬───────────┘
          │ calls
┌─────────────────────┐
│   core/pipeline.py  │  → Feature engineering & ML scoring
└─────────┬───────────┘
          │ reads
┌─────────────────────┐
│ data/examples/*.csv │  → Sample datasets (A/B/C)
└─────────────────────┘
```

- **Streamlit entrypoint (`app.py`)** keeps the front door thin: set layout, apply global theming, initialise session state, and delegate to the view layer.
- **Modular UI (`app_modules/views.py`)** encapsulates landing controls, upload forms, sample previews, the AI prompt panel, and the multi-tab dashboard (overview, segments, trader detail, alerts).
- **Data helpers (`app_modules/data.py`)** handle parsing uploads (CSV/XLSX), invoking the scoring pipeline, formatting tables, and loading bundled examples.
- **Pipeline core (`core/pipeline.py`)** converts transactional data into behavioural signals, feeds a scikit-learn logistic regression pipeline, calibrates risk scores, and generates actionable alerts.

Running `streamlit run app.py` loads the full experience; the modular layout keeps each concern testable and reusable.

## 3. Data Contracts

### 3.1 Trades (`trades.csv`)
| Column      | Type    | Description                                      |
|-------------|---------|--------------------------------------------------|
| `trader_id` | string  | Stable identifier for each trader               |
| `trade_id`  | int     | Unique identifier per trade row                 |
| `timestamp` | ISO8601 | Executed time stamp                              |
| `side`      | string  | `B` (buy) or `S` (sell)                          |
| `ticker`    | string  | Equity ticker symbol                             |
| `qty`       | float   | Quantity traded                                  |
| `price`     | float   | Fill price                                       |
| `fee`       | float   | Transaction fee                                  |
| `date`      | string  | Trade date (YYYY-MM-DD)                          |

### 3.2 Daily portfolio (`daily.csv`, optional)
| Column          | Type    | Description                                 |
|-----------------|---------|---------------------------------------------|
| `trader_id`     | string  | Matches trades file                         |
| `date`          | string  | Trading day                                 |
| `portfolio_value` | float | End-of-day notional value for drawdown calc |

Uploaded data remains in-memory for the Streamlit session and is never persisted to disk.

## 4. Feature Engineering

The pipeline converts raw trades and optional daily valuations into behavioural signals:

| Feature                    | Logic                                                                           |
|----------------------------|----------------------------------------------------------------------------------|
| `trades_per_active_day`    | Mean trades per day during active days                                           |
| `turnover`                 | Sum of absolute notional traded                                                  |
| `pct_days_traded`          | Active trading days ÷ time window                                                |
| `orders_burstiness`        | σ(trades/day) ÷ μ(trades/day), capturing streakiness                             |
| `avg_hold_days`            | Average positive gap between trade days (proxy for holding period)               |
| `buy_after_spike_rate`     | Share of buys on high-notional days (herding / FOMO indicator)                   |
| `max_drawdown`             | Largest drawdown from daily portfolio values (if provided)                       |
| `avg_down_events_rate`     | Alias to `max_drawdown` to serve loss-aversion features                          |

**Mathematical formulation.** For trader *i*, define:

- `D_i = {t_1, ..., t_ki}` = active trading days (size `k_i`)
- `c_i,t` = trade count on day *t*
- `n_i,t = Σ_{trades ℓ on t} |q_ℓ| * |p_ℓ|` = total notional on day *t*
- `b_i,t` = number of buy orders on day *t*
- `T_i = max(D_i) - min(D_i) + 1` = calendar span in days

Using these:

- `trades_per_active_day_i = (1 / k_i) * Σ_{t ∈ D_i} c_i,t`
- `turnover_i = Σ_{t ∈ D_i} n_i,t`
- `pct_days_traded_i = k_i / T_i`
- `orders_burstiness_i = sqrt( (1 / k_i) * Σ_{t ∈ D_i} (c_i,t - c̄_i)^2 ) / c̄_i`, where `c̄_i` is `trades_per_active_day_i`
- `avg_hold_days_i = (1 / |G_i|) * Σ_{Δ ∈ G_i} Δ`, with `G_i = {Δ_j = t_j - t_{j-1} | Δ_j > 0}`
- `buy_after_spike_rate_i = Σ_{t ∈ D_i} 1{ n_i,t ≥ Q0.9(n_i,·) } * b_i,t  /  Σ_{t ∈ D_i} b_i,t`
- `max_drawdown_i = | min_t ( v_i,t / max_{τ ≤ t} v_i,τ - 1 ) |`, where `v_i,t` is the daily portfolio value (default `0` if missing)
- `avg_down_events_rate_i = max_drawdown_i`

Safeguards promote robustness:
- Automatically generate `trade_id` if absent.
- Coerce timestamps/dates to datetime, dropping invalid rows.
- Clip denominators to avoid divide-by-zero errors.

## 5. Risk Scoring Model

The scoring module combines heuristics with an on-the-fly logistic regression:

1. **Signal synthesis.** `s_i = 0.55 * trades_per_active_day_i + 0.25 * avg_down_events_rate_i + 0.20 * buy_after_spike_rate_i`
2. **Time-split style thresholding.** With dense rank `r_i`, compute `theta = median_j ( s_j + 0.05 * r_j / max_ℓ r_ℓ )`. Create pseudo-labels `y_i = 1{ s_i + ε_i > theta }`, where `ε_i` is Gaussian noise with mean `0` and standard deviation `0.1`.
3. **Model training.** Form feature vector `x_i` (the engineered metrics). Standardise it and fit logistic regression so that `p̂_i = 1 / (1 + exp(-βᵀ x̃_i))`. If only one class exists, use the fallback `p̂_i = (s_i - min_j s_j) / (max_j s_j - min_j s_j)` and default to `0.5` when the denominator is zero.
4. **Behavioural component scores.** Using cohort maxima `M_trade`, `M_loss`, and `M_herd`, scale each component:  
   `overtrading_i = min(trades_per_active_day_i / M_trade, 1)`  
   `loss_aversion_i = min(avg_down_events_rate_i / M_loss, 1)`  
   `herding_i = min(buy_after_spike_rate_i / M_herd, 1)`
5. **Risk score.** `R_i = 100 * clip(p̂_i, 0, 1)`. Tiers: Low if `R_i < 30`, Medium if `30 ≤ R_i < 60`, High if `R_i ≥ 60`.

Alerts focus on the riskiest traders (≥60 or top decile), supplemented with SHAP-style heuristics describing the dominant driver (“High trade velocity vs. peers”, “Holding losers too long”, or “Momentum chasing patterns”).

## 6. Sample Packs

Three example datasets ship with the repository:

| Pack            | Cohort summary                                           |
|-----------------|----------------------------------------------------------|
| Sample Pack A   | Diverse mix of low- and medium-risk activity             |
| Sample Pack B   | Larger book with several high-risk clusters              |
| Sample Pack C   | Synthetic high-volatility traders for stress-testing     |

Users can inspect sample tables on the landing page, download them, or run the scoring pipeline immediately to explore the dashboards.

## 7. Streamlit Experience

1. **Upload section** — CSV/XLSX uploaders for trades and optional daily values, blue CTA for running the pipeline, and quick access to sample packs.
2. **Persistent validation** — Missing uploads trigger a warning that remains visible until resolved.
3. **Sample preview** — Tabbed tables show the first five rows of trades/daily data plus the risk score range.
4. **AI prompt** — Scrollable, copy-enabled code block provides a ready-made request for any LLM to fabricate compliant datasets (with optional `[TARGET_RISK_TIER]` bias).
5. **Dashboards**:
   - *Overview* — KPIs, score histogram, radar chart, top-risk table, download exports.
   - *Segments* — Turnover vs. drawdown scatter with filters and CSV export.
   - *Trader Detail* — Behavioural gauges, feature table, trade timeline.
   - *Alerts* — Ranked list with driver narration.
   - A blue “Back to Home” button resets the app without needing to rerun.

UI theming applies a consistent gradient blue to all buttons for an enterprise-ready polish.

## 8. Extensibility

- **Model upgrades** — Swap in Gradient Boosting, XGBoost, or other calibrated models by modifying `core/pipeline.py` while keeping the RiskBundle schema stable.
- **Reason codes** — Integrate SHAP/LIME explanations for richer alert narratives.
- **Data persistence** — Add connectors to S3, BigQuery, or Delta Lake if long-lived storage is needed.
- **Scheduling** — Wrap the scoring pipeline in Airflow/Prefect for nightly batch runs.
- **Access control** — Embed authentication via Streamlit’s experimental auth or a reverse proxy if deployed company-wide.

## 9. Operational Considerations

| Area              | Notes                                                                 |
|-------------------|-----------------------------------------------------------------------|
| Data privacy      | System is session-based; sensitive data is never written to disk.     |
| Model monitoring  | Add drift detection on feature distributions when integrating live.   |
| Validation        | Ensure uploaded files meet schema expectations before scoring.        |
| Performance       | Current setup handles ~10⁵ trades comfortably; scale with caching or batch jobs if needed. |

## 10. Getting Started

1. Install dependencies (`pip install -r requirements.txt`).
2. Launch the UI (`streamlit run app.py`).
3. Upload trades/daily CSVs or click a sample pack.
4. Press “Generate Risk Scores” to run the pipeline and explore the dashboards.

The modular codebase, with clear boundaries between UI, data helpers, and the ML core, is ready for extension into production workflows or integration into broader compliance suites.
