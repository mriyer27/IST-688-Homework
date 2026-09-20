import streamlit as st

# ---------------------------------------------------------------------------
# HW Manager: multi-page entry point.
# Each homework lives in its own file under the HW/ folder and shows up as a
# page in the sidebar navigation.
# ---------------------------------------------------------------------------

st.set_page_config(page_title="HW Manager", page_icon="🏠", layout="wide")

hw1_page = st.Page("HW1.py", title="HW1: Document Q&A", icon="📄")
hw2_page = st.Page("HW2.py", title="HW2: URL Summarizer", icon="🌐")
hw3_page = st.Page("HW3.py", title="HW3: URL Chatbot", icon="💬")
hw4_page = st.Page("HW4.py", title="HW4: Org Chatbot (RAG)", icon="🎓", default = True)

pg = st.navigation([hw1_page, hw2_page, hw3_page, hw4_page])
pg.run()