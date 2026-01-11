import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss,pickle,os

doc=fitz.open("data/sample.pdf")
text=""

for p in doc:
    text+=p.get_text()

#Each chunk will have 800 characters . chunk_overlap enables contuinity of context(meaning) by allowing overlaps
splitter= RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=100)
chunks=splitter.split_text(text)

model=SentenceTransformer("all-MiniLM-L6-v2")
embeddings=model.encode(chunks)

dim=embeddings.shape[1] #.shape() returns number of vectors,dimensions per vector
index=faiss.IndexFlatL2(dim)
index.add(embeddings)

os.makedirs("vectorstore",exist_ok=True)
faiss.write_index(index,"vectorstore/index.faiss")

with open("vectorstore/chunks.pkl","wb") as f:
    pickle.dump(chunks,f) #pkl files stores python objects.
print("vector store ready")
