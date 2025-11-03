import streamlit as st


DEFAULT_SESSION_STATE = {
    "active_view": "landing",
    "bundle": None,
    "bundle_label": "",
    "uploaded_trades": None,
    "uploaded_daily": None,
    "warning_message": "",
}


def init_state() -> None:
    """Ensure all keys used by the app exist in Streamlit's session state."""
    for key, value in DEFAULT_SESSION_STATE.items():
        st.session_state.setdefault(key, value)
