# --- Fix for Streamlit Community Cloud's outdated system sqlite3 (chromadb needs 3.35+) ["Claude"] ---
# See: https://stackoverflow.com/questions/76958817
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass  # not needed locally on most machines, only on Streamlit Cloud

import os
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from bs4 import BeautifulSoup
from openai import OpenAI

# ---------------------------------------------------------------------------
# Part 2: Build (or reuse) a persistent ChromaDB vector database from HTML
# ---------------------------------------------------------------------------

st.title("HW 4 - iSchool Student Organizations Chatbot")
st.write(
    "Ask me about iSchool student organizations. This chatbot retrieves relevant "
    "passages from the organizations' pages and uses them to ground its answers.\n\n"
    "**Conversation memory:** I keep a running buffer of your last **5 exchanges** "
    "(5 questions and my 5 responses to them) so I can follow up naturally, but "
    "older turns are dropped to keep things fast and affordable."
)


HTML_FOLDER = "HW4_HTML"

CHROMA_PATH = "./HW4_ChromaDB"

openai_api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=openai_api_key)


def chunk_text(text: str, filename: str):
    """
    Split a document into exactly 2 chunks, breaking on paragraph boundaries
    near the midpoint rather than at an arbitrary character count.

    Chunking method chosen: fixed 2-way midpoint split on paragraph
    boundaries. The student-org pages here are short, single-topic pages
    (usually just a description, meeting info, and contact details), so
    a small fixed number of chunks is enough to let retrieval distinguish
    "what the org is" from "how to join/contact them" without fragmenting
    the text so much that individual chunks lose context. Splitting on
    paragraph breaks (rather than a raw character-count cutoff) avoids
    slicing a chunk off in the middle of a sentence.
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if len(paragraphs) < 2:
        return [text], [f"{filename}::chunk1"]

    midpoint = len(paragraphs) // 2
    chunk1 = "\n".join(paragraphs[:midpoint])
    chunk2 = "\n".join(paragraphs[midpoint:])
    return [chunk1, chunk2], [f"{filename}::chunk1", f"{filename}::chunk2"]


def build_vector_db():
    """
    Read every HTML file in HTML_FOLDER, chunk it into 2 pieces, embed each
    chunk, and store it in a persistent Chroma collection. If the collection
    already has data (from a previous run), skip re-embedding entirely.
    """
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai_api_key,
        model_name="text-embedding-3-small",
    )
    collection = chroma_client.get_or_create_collection(
        name="HW4Collection",
        embedding_function=embedding_fn,
    )

    if collection.count() > 0:
        return collection  # already built in a previous run — don't re-embed

    for filename in os.listdir(HTML_FOLDER):
        if not filename.lower().endswith((".html", ".htm")):
            continue
        path = os.path.join(HTML_FOLDER, filename)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw_html = f.read()

        soup = BeautifulSoup(raw_html, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        if not text.strip():
            continue

        chunks, ids = chunk_text(text, filename)
        collection.add(
            documents=chunks,
            ids=ids,
            metadatas=[{"filename": filename, "chunk": i} for i in range(len(chunks))],
        )

    return collection


# Only build once — check session_state first (fast path within a session),
# and build_vector_db() itself also checks the persisted collection (fast
# path across sessions/restarts), so embeddings only ever happen once total.
if "HW4_VectorDB" not in st.session_state:
    with st.spinner("Loading the student organizations knowledge base..."):
        st.session_state.HW4_VectorDB = build_vector_db()

collection = st.session_state.HW4_VectorDB


def retrieve_context(query: str, k: int = 3) -> str:
    """Fetch the k most relevant chunks and format them as context text."""
    results = collection.query(query_texts=[query], n_results=k)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    parts = [
        f"[Source: {meta.get('filename', 'unknown')}]\n{doc}"
        for doc, meta in zip(docs, metas)
    ]
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Part 3: Chat interface with a 5-interaction (10-message) conversation buffer
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


def build_buffer() -> list:
    # "Last 5 interactions" = last 5 user/assistant pairs = 10 messages.
    return st.session_state.messages[-10:]


user_input = st.chat_input("Ask about a student organization")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    context = retrieve_context(user_input)
    history = build_buffer()

    # System prompt is rebuilt fresh every turn and prepended separately from
    # the trimmed history buffer, so it's never at risk of being trimmed away.
    system_prompt = (
        "You are a helpful assistant that answers questions about iSchool "
        "student organizations. Use the reference material below, retrieved "
        "from the organizations' own pages, to answer the user's question. "
        "If you use this material, say so clearly (e.g., 'According to the "
        "organization's page...'). If the material doesn't contain a relevant "
        "answer, say so plainly and answer from your own general knowledge "
        "instead, making clear that you're doing so.\n\n"
        f"Reference material:\n{context}"
    )

    messages_to_send = [{"role": "system", "content": system_prompt}] + history

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="gpt-5-mini",
            messages=messages_to_send,
            stream=True,
        )
        response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})