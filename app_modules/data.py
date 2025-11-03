from typing import Any, Tuple

import pandas as pd
import streamlit as st

from core.pipeline import run_pipeline, RiskBundle
from .constants import SAMPLE_DATASETS


def parse_upload(uploaded_file: Any) -> pd.DataFrame | None:
    """Read an uploaded CSV/XLSX file into a DataFrame."""
    if uploaded_file is None:
        return None
    name = uploaded_file.name.lower()
    try:
        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file)
        return pd.read_csv(uploaded_file)
    except Exception as err:  # pragma: no cover - defensive UI message
        st.error(f"Could not read {uploaded_file.name}: {err}")
        return None


def load_sample_dataset(label: str) -> RiskBundle:
    """Return a RiskBundle for one of the bundled sample datasets."""
    paths = SAMPLE_DATASETS[label]
    trades = pd.read_csv(paths["trades"])
    daily = None
    if paths.get("daily"):
        daily = pd.read_csv(paths["daily"])
    return run_pipeline(trades, daily)


def tidy_dataframe(
    df: pd.DataFrame,
    rename_map: dict | None = None,
    percent_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Apply consistent formatting for Streamlit display tables."""
    display = df.copy()
    if percent_cols:
        for col in percent_cols:
            if col in display.columns:
                display[col] = display[col].astype(float) * 100

    numeric_cols = display.select_dtypes(include="number").columns
    if len(numeric_cols):
        display[numeric_cols] = display[numeric_cols].round(2)

    def prettify(name: str) -> str:
        if rename_map and name in rename_map:
            return rename_map[name]
        return name.replace("_", " ").title()

    display.columns = [prettify(col) for col in display.columns]
    return display


def generate_from_uploads() -> Tuple[bool, str | None]:
    """Run the risk pipeline using uploaded data, returning status and warning."""
    if st.session_state["uploaded_trades"] is None:
        message = "Upload a trades file before running the scoring pipeline."
        st.session_state["warning_message"] = message
        st.warning(message)
        return False, message

    trades = st.session_state["uploaded_trades"]
    daily = st.session_state["uploaded_daily"]
    bundle = run_pipeline(trades, daily)
    st.session_state["bundle"] = bundle
    st.session_state["bundle_label"] = "Your Upload"
    st.session_state["active_view"] = "dashboard"
    st.session_state["warning_message"] = ""
    st.success("Scoring complete! Jump to the dashboard below.")
    return True, None
