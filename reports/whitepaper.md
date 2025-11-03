# Behavioral Bias Detection: Whitepaper

**Run date:** 2025-11-02T23:57:33.937763 UTC

## 1. Problem
Detect **overtrading**, **loss aversion (disposition)**, and **herding** per trader, convert to a single **Behavioral Risk Score** that informs coaching and risk controls.

## 2. Data
Synthetic dataset with **300** traders and **20,165** trades from 2024-01-01 to 2024-06-28.  
Tables: **trades**, **daily snapshots**, **prices** (with spike flags).

## 3. Features
- **Intensity:** trades per active day, % days traded, orders burstiness (Gini).  
- **Turnover:** Σ|qty×price| ÷ avg(AUM proxy).  
- **Hold proxy:** avg distinct days per ticker.  
- **Loss aversion proxy:** avg_down_events_rate (buys below prior buy).  
- **Herding proxy:** buy_after_spike_rate (buys on >95th pct 1D return days).

## 4. Labeling (Heuristic seed)
- **Overtrading**: turnover & trades_per_active_day > 90th pct.  
- **Loss aversion**: avg_down_events_rate > median & hold proxy < median.  
- **Herding**: buy_after_spike_rate > 80th pct.

## 5. Modeling
One‑vs‑rest **logistic regression** with isotonic calibration over standardized features.  
**AUROC** — Overtrading: 0.987, Loss aversion: 0.934, Herding: 0.998.

## 6. Behavioral Risk Score
R = 100*(0.35·p_overtrading + 0.40·p_loss_aversion + 0.25·p_herding).  
Tiers: **Low** (0–29), **Medium** (30–59), **High** (60–100).

## 7. Results
- Tier counts: {'Low': 267, 'Medium': 33, 'High': 0}  
- Top‑decile alerts: 30 traders (see reasons).  
- Figures: `reports/figures/risk_score_hist.png`, `reports/figures/risk_tier_bar.png`.

## 8. Business Actions
- **High**: coach on trade frequency, enable stop‑loss prompts, adjust day‑trading caps.  
- **Medium**: nudges on disposition, education on momentum chasing.  
- **Low**: positive reinforcement; optional advanced content.

## 9. Caveats & Next Steps
- Seed labels are noisy—add P&L‑true disposition metrics, time‑split validation, and model monitoring for drift.
