# --- sqlite3 compatibility fix for Streamlit Community Cloud ---
# chromadb requires sqlite3 >= 3.35, but Streamlit Cloud's default Linux
# environment ships an older system sqlite3 that doesn't meet this
# requirement, causing chromadb to fail on import once deployed (it can
# still work fine locally, since local machines often have a newer sqlite3).
# The fix swaps in the pysqlite3-binary package (a modern, bundled sqlite3
# build) in place of the standard library's sqlite3 module before chromadb
# is imported.
# Reference: https://stackoverflow.com/questions/76958817
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import json
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.title("HW 7 - News Monitoring Bot")
st.write(
    "Ask about news on your firm's monitored companies. This bot only reports "
    "on the articles in its dataset, it will not invent news or draw on "
    "outside knowledge, since accuracy matters for legal/compliance monitoring.\n\n"
    "Try: \"Find the most interesting news\" or \"Find news about Apple.\""
)

CHROMA_PATH = "./NewsChromaDB"
COLLECTION_NAME = "NewsCollection"

openai_api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=openai_api_key)

# ---------------------------------------------------------------------------
# Load the prebuilt vector DB (built offline by build_news_db.py).
# This app NEVER embeds anything itself, it only reads what's already there,
# which is why it loads instantly rather than re-processing the whole dataset.
# ---------------------------------------------------------------------------


@st.cache_resource
def load_collection():
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai_api_key,
        model_name="text-embedding-3-small",
    )
    return chroma_client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )


try:
    collection = load_collection()
    if collection.count() == 0:
        st.error(
            "The news vector database is empty. Run build_news_db.py once "
            "(locally or in your Codespace, with news.csv present) before "
            "using this app."
        )
        st.stop()
except Exception:
    st.error(
        "Couldn't find the news vector database at "
        f"'{CHROMA_PATH}'. Run build_news_db.py once to build it, then make "
        "sure the NewsChromaDB folder is deployed alongside this app."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Step 4: model picker for the low-cost vs high-cost comparison
# ---------------------------------------------------------------------------

st.sidebar.header("Model")
MODEL_OPTIONS = {
    "GPT-5.6 Luna (lower cost)": "gpt-5.6-luna",
    "GPT-5.6 Sol (higher cost)": "gpt-5.6-sol",
}
model_choice = st.sidebar.selectbox("Choose a model", list(MODEL_OPTIONS.keys()))
model_name = MODEL_OPTIONS[model_choice]

# ---------------------------------------------------------------------------
# Retrieval tools
# ---------------------------------------------------------------------------


def _format_results(results) -> str:
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    if not docs:
        return "No matching articles found in the dataset."
    parts = []
    for doc, meta in zip(docs, metas):
        parts.append(
            f"[Company: {meta.get('company', '?')} | Date: {meta.get('date', '?')} | "
            f"URL: {meta.get('url', '?')}]\n{doc}"
        )
    return "\n\n".join(parts)


def search_news(query: str, k: int = 5) -> str:
    """General semantic search for a specific topic or company."""
    results = collection.query(query_texts=[query], n_results=k)
    return _format_results(results)


# A fixed internal query capturing what a law firm actually cares about.
# "Most interesting news" is answered by reusing the same vector search
# machinery as search_news, just with this query instead of user text, so
# there's only one retrieval mechanism to build, test, and reason about.
INTERESTING_QUERY = (
    "lawsuit, legal action, regulatory investigation, government scrutiny, "
    "fine, settlement, data breach, scandal, fraud, antitrust, bankruptcy, "
    "executive departure, major acquisition, significant financial risk"
)


def find_interesting_news(k: int = 5) -> str:
    """Retrieve the most legally/financially significant articles in the dataset."""
    results = collection.query(query_texts=[INTERESTING_QUERY], n_results=k)
    return _format_results(results)


search_tool = {
    "type": "function",
    "function": {
        "name": "search_news",
        "description": (
            "Search the news dataset for articles about a specific topic, "
            "company, or event. Use this for 'find news about X' style questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for, e.g. a company name or topic.",
                },
                "k": {
                    "type": "integer",
                    "description": "How many articles to retrieve (default 5).",
                },
            },
            "required": ["query"],
        },
    },
}

interesting_tool = {
    "type": "function",
    "function": {
        "name": "find_interesting_news",
        "description": (
            "Retrieve the most legally/financially significant news articles "
            "in the dataset (lawsuits, regulatory action, scandals, major "
            "financial risk, etc). Use this for 'find the most interesting "
            "news' style requests, when no specific company or topic is named."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "k": {
                    "type": "integer",
                    "description": "How many articles to retrieve (default 5).",
                },
            },
        },
    },
}

TOOLS = [search_tool, interesting_tool]
AVAILABLE_FUNCTIONS = {
    "search_news": search_news,
    "find_interesting_news": find_interesting_news,
}

# ---------------------------------------------------------------------------
# System prompt — strict grounding, no fallback to outside knowledge
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a news monitoring assistant for a law firm. You only report on "
    "articles contained in the firm's news dataset, retrieved via your tools. "
    "Never invent articles, facts, or details that aren't in the retrieved "
    "results. If a search returns no relevant articles, say so plainly rather "
    "than guessing or supplementing with outside knowledge, accuracy matters "
    "here far more than giving a complete-sounding answer.\n\n"
    "When asked to find the most interesting news, with no specific company "
    "or topic named, call find_interesting_news. When asked about a specific "
    "company or topic, call search_news with that company/topic as the query. "
    "When presenting results, always give a numbered, ranked list, and for "
    "each article explain briefly why it's relevant or noteworthy for a law "
    "firm monitoring this company (e.g. legal exposure, regulatory risk, "
    "reputational impact), don't just restate the headline. Include the "
    "source URL for each article so the user can read the original."
)

# ---------------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask about monitored news (e.g. 'find the most interesting news')")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    history = st.session_state.messages[-6:]
    first_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    # First call: let the model choose which retrieval tool fits the request.
    # reasoning_effort="none" is required here — GPT-5.6-family models don't
    # support function tools together with their default reasoning_effort on
    # the /v1/chat/completions endpoint.
    first_response = client.chat.completions.create(
        model=model_name,
        messages=first_messages,
        tools=TOOLS,
        tool_choice="auto",
        reasoning_effort="none",
    )
    assistant_message = first_response.choices[0].message

    if assistant_message.tool_calls:
        tool_call = assistant_message.tool_calls[0]
        func_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments or "{}")
        func = AVAILABLE_FUNCTIONS.get(func_name)
        retrieved = func(**args) if func else "No matching tool found."

        # Second call: fold retrieved articles into the system prompt. No
        # tools this time, so the model can only use what was just retrieved.
        final_system_prompt = (
            SYSTEM_PROMPT + "\n\nArticles retrieved for this request:\n\n" + retrieved
        )
        final_messages = [{"role": "system", "content": final_system_prompt}] + history

        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=model_name,
                messages=final_messages,
                stream=True,
            )
            response = st.write_stream(stream)
    else:
        response = assistant_message.content
        with st.chat_message("assistant"):
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})