from sentence_transformers import SentenceTransformer
import faiss,pickle
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
client=genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
index=faiss.read_index("vectorstore/index.faiss")
chunks=pickle.load(open("vectorstore/chunks.pkl","rb"))
#pickle.load(file_object);Loads reconstructs py objects
model=SentenceTransformer("all-MiniLM-L6-v2")
q=input("Ask:")
qv=model.encode([q])
_,ids=index.search(qv,3) #basically it returns distance,index=index.search(qv,3)

#for i in ids[0]:   #Semantic search only
#   print("\n---\n",chunks[i])


#Merge chunks
context="\n".join([chunks[i] for i in ids[0]])
prompt= f"""
Answer ONLY from context.

Context:{context}
Question:{q}
"""
res=client.models.generate_content(model="models/gemini-2.5-flash",contents=prompt)

print(res.text)
#choices has list of outputs . chocies[0] gives first ouptut .mesage->role+content .content ->actual reply
