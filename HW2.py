import time

import requests
import streamlit as st
from anthropic import Anthropic
from bs4 import BeautifulSoup
from openai import OpenAI

# ---------------------------------------------------------------------------
# HW2: URL Summarizer with Multiple LLMs
# ---------------------------------------------------------------------------

st.title("HW2: URL Summarizer with Multiple LLMs")

# --- URL input: top of the screen, NOT the sidebar (item 4) -----------------
url = st.text_input("Enter a URL to summarize")

# --- Sidebar: summary type, kept from Lab 2 (item 5) -------------------------
st.sidebar.header("Summary options")
summary_type = st.sidebar.radio(
    "Type of summary",
    (
        "Summarize in 100 words",
        "Summarize in 2 connecting paragraphs",
        "Summarize in 5 bullet points",
    ),
)

# --- Sidebar: output language, at least 3 options (item 8) -------------------
language = st.sidebar.selectbox("Output language", ("English", "French", "Spanish"))

# --- Sidebar: LLM provider + advanced-model checkbox (items 10-11) ----------
st.sidebar.header("Model options")
llm_provider = st.sidebar.selectbox("Choose an LLM", ("OpenAI", "Anthropic (Claude)"))
use_advanced = st.sidebar.checkbox("Use advanced model")

# --- API keys: read from secrets.toml only (item 15). -----------------------
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    anthropic_api_key = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    st.error(
        "Missing OPENAI_API_KEY and/or ANTHROPIC_API_KEY. Add both to "
        ".streamlit/secrets.toml locally, or to this app's Secrets settings "
        "on Streamlit Community Cloud."
    )
    st.stop()

# --- Model choices: cheap vs. advanced, per provider (items 12-13) ----------
OPENAI_MODELS = {"cheap": "gpt-5-nano", "advanced": "gpt-5.6-sol"}
ANTHROPIC_MODELS = {"cheap": "claude-haiku-4-5-20251001", "advanced": "claude-opus-5"}

SUMMARY_INSTRUCTIONS = {
    "Summarize in 100 words": "Summarize the following text in about 100 words.",
    "Summarize in 2 connecting paragraphs": "Summarize the following text in exactly 2 connecting paragraphs.",
    "Summarize in 5 bullet points": "Summarize the following text as 5 concise bullet points.",
}


def read_url_content(url):
    """Given by the assignment - fetch a URL and return its visible text."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None


def summarize_with_openai(api_key, model, prompt):
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def summarize_with_anthropic(api_key, model, prompt):
    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    # FIX (added with Claude's help, after hitting:
    # "AttributeError: 'ThinkingBlock' object has no attribute 'text'"):
    # some models (e.g. claude-opus-5) return a "thinking" block ahead of
    # the actual answer, so content[0] isn't always the text - pull out
    # whichever block(s) are actually type "text" instead.
    return "".join(block.text for block in response.content if block.type == "text")


if url:
    if st.button("Summarize"):
        document = read_url_content(url)

        if document:
            instruction = SUMMARY_INSTRUCTIONS[summary_type]
            prompt = (
                f"{instruction}\n\n"
                f"Write your answer in {language}.\n\n"
                f"Here is the text:\n\n{document}"
            )

            if llm_provider == "OpenAI":
                try:
                    openai_client = OpenAI(api_key=openai_api_key)
                    openai_client.models.list()
                except Exception as e:
                    st.error(f"OpenAI API call failed: {e}", icon="🚫")
                else:
                    model = OPENAI_MODELS["advanced" if use_advanced else "cheap"]
                    start = time.time()
                    with st.spinner(f"Summarizing with {model}..."):
                        summary = summarize_with_openai(openai_api_key, model, prompt)
                    st.write(summary)
                    st.caption(f"Model: {model}  ·  ⏱️ {time.time() - start:.1f}s")

            else:  # Anthropic (Claude)
                try:
                    anthropic_client = Anthropic(api_key=anthropic_api_key)
                    anthropic_client.models.list()
                except Exception as e:
                    st.error(f"Anthropic API call failed: {e}", icon="🚫")
                else:
                    model = ANTHROPIC_MODELS["advanced" if use_advanced else "cheap"]
                    start = time.time()
                    with st.spinner(f"Summarizing with {model}..."):
                        summary = summarize_with_anthropic(anthropic_api_key, model, prompt)
                    st.write(summary)
                    st.caption(f"Model: {model}  ·  ⏱️ {time.time() - start:.1f}s")