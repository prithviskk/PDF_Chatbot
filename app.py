import streamlit as st
import fitz
from sentence_transformers import SentenceTransformer
import faiss, os
import google.genai as genai
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import base64

# ---------------- CONFIG ----------------
st.set_page_config(layout="wide")
st.title("📄 PDF RAG Bot")

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------- SESSION STATE -------------
if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = None
    st.session_state.pdf_bytes = None

# ------------ PDF DISPLAY FN -------------
def display_pdf(pdf_bytes):
    base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

    pdf_html = f"""
    <object
        data="data:application/pdf;base64,{base64_pdf}"
        type="application/pdf"
        width="100%"
        height="800px">

        <p>PDF cannot be displayed.
        <a href="data:application/pdf;base64,{base64_pdf}" download="file.pdf">
        Download PDF</a></p>

    </object>
    """

    st.markdown(pdf_html, unsafe_allow_html=True)

# ---------------- LAYOUT -----------------
left, right = st.columns([1, 1])

# -------------- LEFT SIDE ---------------
with left:
    st.subheader("📂 Uploaded PDF")

    uploaded = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        accept_multiple_files=False
    )

    if uploaded and st.session_state.index is None:

        pdf_bytes = uploaded.read()
        st.session_state.pdf_bytes = pdf_bytes

        # Show PDF
        display_pdf(pdf_bytes)

        # Extract text
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()

        # Chunking
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )

        chunks = splitter.split_text(text)

        # Embeddings
        embeds = model.encode(chunks)

        dim = embeds.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeds)

        st.session_state.index = index
        st.session_state.chunks = chunks

        st.success("✅ PDF processed successfully!")

    elif st.session_state.pdf_bytes:
        # Show PDF again after rerun
        display_pdf(st.session_state.pdf_bytes)

# -------------- RIGHT SIDE --------------
with right:
    st.subheader("🤖 Ask your PDF")

    q = st.text_input("Ask your question")

    if st.button("Get Answer"):

        if st.session_state.index is None:
            st.warning("⚠ Upload a PDF first!")
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
