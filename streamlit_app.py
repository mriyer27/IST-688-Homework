import streamlit as st

# ---------------------------------------------------------------------------
# HW Manager: multi-page entry point.
# ---------------------------------------------------------------------------

st.set_page_config(page_title="HW Manager", page_icon="🏠", layout="wide")

hw1_page = st.Page("HW1.py", title="HW1: Document Q&A", icon="📄")
hw2_page = st.Page("HW2.py", title="HW2: URL Summarizer", icon="🌐")
hw3_page = st.Page("HW3.py", title="HW3: URL Chatbot", icon="💬", default = True)

pg = st.navigation({"HW Manager": [hw1_page, hw2_page, hw3_page]})
pg.run()