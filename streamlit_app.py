import time

import streamlit as st
from openai import OpenAI
from pypdf import PdfReader


def read_pdf(uploaded_file):
    """Read an uploaded PDF file and return its text content as a string."""
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


# Show title and description.
st.title("My Document question answering — gpt-5-nano")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:

    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

    # Validate the API key immediately, before showing the rest of the app.
    # `models.list()` is a cheap call that just checks the key works.
    try:
        client.models.list()
        key_is_valid = True
    except Exception as e:
        key_is_valid = False
        st.error(
            "Your API key doesn't seem to be valid. Please check it and try again.",
            icon="🚫",
        )

    if key_is_valid:
        st.success("API key validated!", icon="✅")

        # Let the user upload a file via `st.file_uploader`. (item 3a: .pdf and .txt only)
        uploaded_file = st.file_uploader(
            "Upload a document (.txt or .pdf)", type=("txt", "pdf")
        )

        # Ask the user for a question via `st.text_area`.
        question = st.text_area(
            "Now ask a question about the document!",
            placeholder="Can you give me a short summary?",
            disabled=not uploaded_file,
        )

        if uploaded_file and question:

            # Process the uploaded file and question (item 3a).
            file_extension = uploaded_file.name.split('.')[-1]
            if file_extension == 'txt':
                document = uploaded_file.read().decode()
            elif file_extension == 'pdf':
                document = read_pdf(uploaded_file)
            else:
                st.error("Unsupported file type.")
                document = None

            if document:
                messages = [
                    {
                        "role": "user",
                        "content": f"Here's a document: {document} \n\n---\n\n {question}",
                    }
                ]

                # Generate an answer using the OpenAI API.
                # gpt-5-nano: smallest/cheapest GPT-5-family model, still reasoning-tier.
                start = time.time()
                stream = client.chat.completions.create(
                    model="gpt-5-nano",
                    messages=messages,
                    stream=True,
                )

                # Stream the response to the app using `st.write_stream`.
                st.write_stream(stream)
                st.caption(f"⏱️ {time.time() - start:.1f}s (wall-clock, includes streaming)")