from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from groq import Groq


# ---------- LOAD ENV ----------

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


app = Flask(__name__)


# ---------- LOAD PDF ----------

loader = PyPDFLoader("data/KEC_COLLEGE.pdf")
docs = loader.load()


# ---------- SPLIT ----------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)


# ---------- EMBEDDINGS ----------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ---------- VECTOR DB ----------

db = FAISS.from_documents(
    chunks,
    embeddings
)

retriever = db.as_retriever(
    search_kwargs={"k": 2}
)


# ---------- ASK FUNCTION ----------

def ask(q):

    docs = retriever.invoke(q)

    context = " ".join([d.page_content for d in docs])

    prompt = f"""
Answer shortly using context.

Context:
{context}

Question: {q}
Answer:
"""

    chat = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return chat.choices[0].message.content


# ---------- API ----------

@app.route("/ask", methods=["POST"])
def ask_api():

    q = request.json["question"]

    ans = ask(q)

    return jsonify({"answer": ans})


@app.route("/")
def home():
    return "RAG running"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)