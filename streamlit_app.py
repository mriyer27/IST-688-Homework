import streamlit as st

# ---------------------------------------------------------------------------
# HW Manager: multi-page entry point.
# Each homework lives in its own file under the HW/ folder and shows up as a
# page in the sidebar navigation.
# ---------------------------------------------------------------------------

st.set_page_config(page_title="HW Manager", page_icon="🏠", layout="wide")

hw1_page = st.Page("HW1.py", title="HW1: Document Q&A", icon="📄")
hw2_page = st.Page("HW2.py", title="HW2: URL Summarizer", icon="🌐")

pg = st.navigation({"HW Manager": [hw1_page, hw2_page]})
pg.run()