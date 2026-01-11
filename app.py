import streamlit as st
import fitz
from sentence_transformers import SentenceTransformer
import faiss, os
import google.genai as genai
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import base64

st.set_page_config(layout="wide")
st.title("📄 PDF RAG Bot")

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

model = SentenceTransformer("all-MiniLM-L6-v2")

# SESSION INIT
if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = None
    st.session_state.pdf_bytes = None

# LAYOUT
left, right = st.columns([1, 1])

with left:
    st.subheader("📂 Uploaded PDF")
    uploaded = st.file_uploader(
        "Upload PDFs", 
        accept_multiple_files=False
    )

    if uploaded:
        pdf_bytes = uploaded.read()
        st.session_state.pdf_bytes = pdf_bytes

        # Display PDF with scroll
        b64 = base64.b64encode(pdf_bytes).decode()
        pdf_display = f"""
        <iframe 
            src="data:application/pdf;base64,{b64}" 
            width="100%" 
            height="700px">
        </iframe>
        """
        st.markdown(pdf_display, unsafe_allow_html=True)

        # Extract text
        r = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
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

        st.success("PDF processed successfully!")

with right:
    st.subheader("🤖 Ask your PDF")

    q = st.text_input("Ask question")

    if st.button("Get Answer"):

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

            st.markdown("### ✅ Answer")
            st.write(res.text)
