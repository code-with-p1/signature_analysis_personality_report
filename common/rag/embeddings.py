import warnings
warnings.filterwarnings(action='ignore')
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from common.rag.document_loader import fetch_document_chunks
from dotenv import load_dotenv
load_dotenv()


def fetch_vectorstore_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vectorstore = FAISS.from_documents(
        documents=fetch_document_chunks(),
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )

    return retriever


def fetch_relevant_document(topic="None"):
    retriever = fetch_vectorstore_retriever()
    docs = retriever.invoke(f"Explain Summary, Writing-style descriptions, Graphology-style Overall impression on given topic : {topic}")
    context = "\n\n".join(f"[Document {i+1}]\n{doc.page_content}\n" for i, doc in enumerate(docs))
    return context