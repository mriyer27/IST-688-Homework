# --- sqlite3 compatibility fix for Streamlit Community Cloud ---
# chromadb requires sqlite3 >= 3.35, but Streamlit Cloud's default Linux
# environment ships an older system sqlite3 that doesn't meet this
# requirement, causing chromadb to fail on import once deployed (it can
# still work fine locally, since local machines often have a newer sqlite3).
# The fix swaps in the pysqlite3-binary package (a modern, bundled sqlite3
# build) in place of the standard library's sqlite3 module before chromadb
# is imported. This snippet, and the reasoning behind it, was worked out
# with the help of Claude (Anthropic).
# Reference: https://stackoverflow.com/questions/76958817
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
import json
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from bs4 import BeautifulSoup
from openai import OpenAI



st.title("HW 5 - iSchool Org Chatbot (Tool-Based RAG)")
st.write(
    "Ask me about iSchool student organizations. Unlike HW4, this version "
    "doesn't always search the knowledge base on every turn, instead, the "
    "AI model itself decides when a lookup is actually needed and calls a "
    "search tool to fetch it, the same way Lab 5's weather bot decided when "
    "to check the weather.\n\n"
    "**Conversation memory:** I keep the last 5 exchanges (10 messages) of "
    "our conversation."
)

HTML_FOLDER = "HW4_HTML"
CHROMA_PATH = "./HW4_ChromaDB"

openai_api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=openai_api_key)


def chunk_text(text: str, filename: str):
    """Split a document into 2 chunks on paragraph boundaries near the midpoint."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if len(paragraphs) < 2:
        return [text], [f"{filename}::chunk1"]
    midpoint = len(paragraphs) // 2
    chunk1 = "\n".join(paragraphs[:midpoint])
    chunk2 = "\n".join(paragraphs[midpoint:])
    return [chunk1, chunk2], [f"{filename}::chunk1", f"{filename}::chunk2"]


def build_vector_db():
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
        return collection 

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


if "HW5_VectorDB" not in st.session_state:
    with st.spinner("Loading the student organizations knowledge base..."):
        st.session_state.HW5_VectorDB = build_vector_db()

collection = st.session_state.HW5_VectorDB




def relevant_club_info(query: str, k: int = 3) -> str:
    """Search the vector DB and return the k most relevant chunks as text."""
    results = collection.query(query_texts=[query], n_results=k)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    parts = [
        f"[Source: {meta.get('filename', 'unknown')}]\n{doc}"
        for doc, meta in zip(docs, metas)
    ]
    return "\n\n".join(parts) if parts else "No relevant information found."


club_info_tool = {
    "type": "function",
    "function": {
        "name": "relevant_club_info",
        "description": (
            "Search the student organizations knowledge base for information "
            "relevant to a question about iSchool clubs or organizations "
            "(e.g. what a club does, how to join, meeting times, contacts)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A search query describing the club/organization information needed.",
                }
            },
            "required": ["query"],
        },
    },
}

BASE_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about iSchool student "
    "organizations. You have a tool, relevant_club_info, that searches a "
    "knowledge base of the organizations' own pages. Call it whenever a "
    "question is about a specific club, its activities, membership, or "
    "contact details. If a question doesn't need that (e.g. small talk, or "
    "something clearly unrelated to student organizations), just answer "
    "directly without calling the tool. If you do use information from the "
    "tool, say so clearly (e.g., 'According to the organization's page...')."
)


# Chat interface with short-term memory


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


def build_buffer() -> list:
    # Last 5 interactions = last 5 user/assistant pairs = 10 messages.
    return st.session_state.messages[-10:]


user_input = st.chat_input("Ask about a student organization")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    history = build_buffer()
    first_messages = [{"role": "system", "content": BASE_SYSTEM_PROMPT}] + history


    first_response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=first_messages,
        tools=[club_info_tool],
        tool_choice="auto",
    )
    assistant_message = first_response.choices[0].message

    if assistant_message.tool_calls:
        tool_call = assistant_message.tool_calls[0]
        arguments = json.loads(tool_call.function.arguments)
        query = arguments.get("query", user_input)

  
        retrieved_info = relevant_club_info(query)


        final_system_prompt = (
            BASE_SYSTEM_PROMPT
            + "\n\nInformation retrieved from the student organizations "
            "knowledge base for this question:\n\n"
            + retrieved_info
        )
        final_messages = [{"role": "system", "content": final_system_prompt}] + history

        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model="gpt-5-mini",
                messages=final_messages,
                stream=True,
            )
            response = st.write_stream(stream)
    else:
        response = assistant_message.content
        with st.chat_message("assistant"):
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})