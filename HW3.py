import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import anthropic

# ---------------------------------------------------------------------------
#HW3: URL Chatbot with Conversation Memory
# ---------------------------------------------------------------------------

def read_url_content(url: str) -> str:
    """Fetch a URL and return its visible, cleaned-up text content."""
    try:
        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)
    except Exception as e:
        return f"[Could not read {url}: {e}]"


# ---------------------------------------------------------------------------
# Part 6 
# ---------------------------------------------------------------------------

st.title("HW 3 - Chatbot that Discusses a URL")
st.write(
    "This chatbot answers questions using the content of up to two web pages "
    "you provide in the sidebar, using the AI model you choose there.\n\n"
    "**Conversation memory:** the page content you provide becomes part of "
    "this chatbot's permanent instructions and is *never* forgotten, no "
    "matter how long the chat runs. On top of that, the chatbot keeps a "
    "short-term memory of the last **3 exchanges (6 messages)** of back-and-"
    "forth conversation — older turns are dropped to keep things fast and "
    "affordable, but the page content is always there."
)

# ---------------------------------------------------------------------------
# Sidebar: reference URLs
# ---------------------------------------------------------------------------

st.sidebar.header("Reference URLs")
url_1 = st.sidebar.text_input("URL 1 (optional)")
url_2 = st.sidebar.text_input("URL 2 (optional)")

# ---------------------------------------------------------------------------
# Sidebar: vendor + model picker
# ---------------------------------------------------------------------------

st.sidebar.header("Model")
VENDOR_MODELS = {
    "OpenAI - GPT-5": {"vendor": "openai", "model": "gpt-5"},
   "Anthropic - Claude Opus 5": {"vendor": "anthropic", "model": "claude-opus-5"},
}
model_choice = st.sidebar.selectbox("Choose a model", list(VENDOR_MODELS.keys()))
selected = VENDOR_MODELS[model_choice]
vendor = selected["vendor"]
model_name = selected["model"]

if vendor == "openai":
    openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
else:
    anthropic_client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

# ---------------------------------------------------------------------------
# Part 4: Build the system prompt from the URL content. If no URLs are provided, the system prompt will indicate that no reference material is available.
# ---------------------------------------------------------------------------

url_sections = []
if url_1:
    url_sections.append(f"--- Content from URL 1 ({url_1}) ---\n{read_url_content(url_1)}")
if url_2:
    url_sections.append(f"--- Content from URL 2 ({url_2}) ---\n{read_url_content(url_2)}")

if url_sections:
    context_block = "\n\n".join(url_sections)
    SYSTEM_PROMPT = (
        "You are a helpful assistant. Answer the user's questions using the "
        "reference material below whenever it's relevant. If the answer isn't "
        "in the material, say so rather than guessing.\n\n" + context_block
    )
else:
    SYSTEM_PROMPT = (
        "You are a helpful assistant. No reference URLs have been provided yet, "
        "so let the user know you don't have any page content to draw from."
    )

# ---------------------------------------------------------------------------
# Session state: full conversation history for display
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Part 5: Build a short-term memory buffer of the last 3 exchanges (6 messages) for context in the next response.
# ---------------------------------------------------------------------------

def build_buffer() -> list[dict]:
    return st.session_state.messages[-6:]


# ---------------------------------------------------------------------------
# Streaming helpers — one per vendor.
# ---------------------------------------------------------------------------

def stream_openai(model: str, system_prompt: str, history: list[dict]):
    messages = [{"role": "system", "content": system_prompt}] + history
    stream = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def stream_anthropic(model: str, system_prompt: str, history: list[dict]):
    with anthropic_client.messages.stream(
        model=model,
        max_tokens=1024,
        system=system_prompt,
        messages=history,
    ) as stream:
        for text in stream.text_stream:
            yield text


# ---------------------------------------------------------------------------
# Chat input + streaming response
# ---------------------------------------------------------------------------

user_input = st.chat_input("Ask a question about the page(s) above")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    history = build_buffer()

    with st.chat_message("assistant"):
        if vendor == "openai":
            response = st.write_stream(stream_openai(model_name, SYSTEM_PROMPT, history))
        else:
            response = st.write_stream(stream_anthropic(model_name, SYSTEM_PROMPT, history))

    st.session_state.messages.append({"role": "assistant", "content": response})