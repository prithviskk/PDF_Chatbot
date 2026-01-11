import streamlit as st
import fitz
from sentence_transformers import SentenceTransformer
import faiss, os
from google import genai
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.title("PDF RAG bot")

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

model = SentenceTransformer("all-MiniLM-L6-v2")

# INIT session
if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = None

uploaded = st.file_uploader("Upload PDFs", accept_multiple_files=True)

if uploaded and st.session_state.index is None:

    text = ""
    for f in uploaded:
        pdf_bytes = f.read()
        r = fitz.open(stream=pdf_bytes, filetype="pdf")
        for p in r:
            text += p.get_text()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)
    embeds = model.encode(chunks)

    dim = embeds.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeds)

    st.session_state.index = index
    st.session_state.chunks = chunks

    st.success("PDFs processed and stored!")

q = st.text_input("Ask question")

if st.button("Enter your question"):

    if st.session_state.index is None:
        st.warning("Upload PDF first!")
    else:
        qv = model.encode([q])
        _, ids = st.session_state.index.search(qv, 3)

        context = "\n".join(
            [st.session_state.chunks[i] for i in ids[0]]
        )

        prompt = f"""
Answer ONLY from the context.

Context:
{context}

Question:
{q}
"""

        res = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=prompt
        )

        st.write(res.text)
