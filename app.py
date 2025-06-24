# app.py
import os
import logging
from flask import Flask, request, jsonify, send_file
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

from langchain_azure_ai.embeddings import AzureAIEmbeddingsModel
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from langchain.schema import SystemMessage, HumanMessage

# ─── Logging Setup ───────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rag_app")

# ─── Helper Functions ────────────────────────────────────────────────────────

def load_and_split_pdf(pdf_path, chunk_size=1000, chunk_overlap=200):
    logger.info(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = splitter.split_documents(pages)
    logger.info(f"Split into {len(docs)} chunks")
    return docs

def build_faiss_index(docs, embed_model, index_path="faiss_index"):
    if os.path.isdir(index_path):
        import shutil
        shutil.rmtree(index_path)
        logger.info("Removed existing FAISS index")
    vs = FAISS.from_documents(docs, embed_model)
    vs.save_local(index_path)
    logger.info("FAISS index built and saved")
    return vs

def create_qa_chain(vs, chat_model, k=2):
    logger.info(f"Creating RetrievalQA chain with k={k}")
    retr = vs.as_retriever(search_kwargs={"k": k})
    chain = RetrievalQA.from_chain_type(
        llm=chat_model, chain_type="stuff",
        retriever=retr, return_source_documents=True
    )
    logger.info("RetrievalQA chain ready")
    return chain

# ─── App Initialization ─────────────────────────────────────────────────────

load_dotenv()
ENDPOINT    = os.getenv("AZURE_INFERENCE_ENDPOINT")
API_KEY     = os.getenv("AZURE_INFERENCE_KEY")
EMBED_MODEL = os.getenv("AZURE_EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL  = os.getenv("AZURE_CHAT_MODEL", "gpt-4o")
PDF_PATH    = os.getenv("PDF_FILE_PATH", "Build.pdf")
TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", 0.3))

logger.info("Initializing Azure OpenAI clients")
cred       = AzureKeyCredential(API_KEY)
embeddings = AzureAIEmbeddingsModel(endpoint=ENDPOINT, credential=cred, model=EMBED_MODEL)
chat_model = AzureAIChatCompletionsModel(
    endpoint=ENDPOINT, credential=cred,
    model=CHAT_MODEL, temperature=TEMPERATURE
)

# Build index & chain ONCE
docs       = load_and_split_pdf(PDF_PATH)
vs         = build_faiss_index(docs, embeddings)
qa_chain   = create_qa_chain(vs, chat_model)

# ─── Flask App ────────────────────────────────────────────────────────────────

app = Flask(__name__)  # no static_folder

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data  = request.get_json(force=True)
        query = data.get("query", "").strip()
        logger.info(f"Query received: {query!r}")

        # 1) Greeting fallback
        if query.lower() in {"", "hi", "hello", "hey"}:
            logger.debug("Greeting detected; using direct LLM")
            resp = chat_model([
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content=query or "Hello!")
            ])
            return jsonify(answer=resp.content, sources=[])

        # 2) RAG retrieval
        rag = qa_chain({"query": query})
        hits = rag["source_documents"]
        logger.info(f"RAG hits: {len(hits)}")

        if not hits:
            logger.debug("No RAG hits; falling back to direct LLM")
            resp = chat_model([
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content=query)
            ])
            return jsonify(answer=resp.content, sources=[])

        # 3) Return RAG answer with sources
        answer = rag["result"]
        sources = [
            f"page {d.metadata.get('page','?')}: {d.page_content[:100].replace(chr(10), ' ')}…"
            for d in hits
        ]
        logger.debug(f"Returning {len(sources)} sources")
        return jsonify(answer=answer, sources=sources)

    except Exception:
        logger.exception("Error in /api/chat")
        return jsonify(error="Internal server error"), 500

@app.route("/")
def index():
    # serve index.html from the same folder
    logger.debug("Serving index.html")
    return send_file("index.html")

if __name__ == "__main__":
    logger.info("Starting Flask on http://0.0.0.0:8000")
    app.run(host="0.0.0.0", port=8000)
