import streamlit as st

from app_modules import init_state, render_dashboard, render_landing


st.set_page_config(page_title="Behavioral Bias Detector", layout="wide")

st.markdown(
    """
    <style>
    .stButton > button {
        background: linear-gradient(135deg, #1E88E5, #42A5F5) !important;
        color: white !important;
        border: none !important;
        border-radius: 0.5rem !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1976D2, #1E88E5) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main() -> None:
    init_state()
    if st.session_state["active_view"] == "landing":
        render_landing()
    else:
        render_dashboard()


if __name__ == "__main__":
    main()
