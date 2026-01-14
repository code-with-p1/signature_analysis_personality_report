import warnings
warnings.filterwarnings(action='ignore')
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from common.rag.document_loader import fetch_document_chunks
from dotenv import load_dotenv
load_dotenv()


def fetch_vectorstore_retriever():
    """
    Create and return a FAISS-based retriever for graphology/handwriting analysis documents.

    This function:
    - Loads sentence-transformers/all-MiniLM-L6-v2 embeddings (GPU if available)
    - Builds a FAISS vector store from document chunks obtained via fetch_document_chunks()
    - Returns a similarity search retriever configured to return top 10 most relevant chunks

    Returns
    -------
    langchain_core.retrievers.BaseRetriever
        Configured FAISS retriever ready to be used with .invoke() or .get_relevant_documents()

    Notes
    -----
    - The vector store is **recreated from scratch every time** this function is called.
    - This can be slow on first run or when document collection is large.
    - Consider caching/persisting the vectorstore in production for better performance.
    - Uses normalize_embeddings=True → cosine similarity is used internally.
    """

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
    """
    Retrieve relevant document chunks for graphological analysis of a specific topic/trait.

    Constructs a detailed, structured query optimized for finding handwriting analysis content,
    then retrieves the top 10 most similar document chunks from the FAISS vector store.

    Parameters
    ----------
    topic : str, default="None"
        Personality trait, psychological characteristic, writing style aspect or any topic
        for which handwriting analysis information is requested.
        Examples: "ambition", "emotional stability", "aggressiveness", "introversion"

    Returns
    -------
    str
        Concatenated string containing up to 10 relevant document chunks, each prefixed
        with "[Document N]" for clear identification in the RAG context.
        Returns empty context string if topic is "None" or no relevant chunks are found.

    Notes
    -----
    - The query is intentionally very specific and structured to improve retrieval quality
      for handwriting/graphology related content.
    - Uses similarity (cosine) search with k=10 (top 10 results).
    - The returned context is meant to be directly passed into a RAG prompt for LLM analysis.
    """

    retriever = fetch_vectorstore_retriever()
    query = (
        f"Handwriting sample analysis for: {topic}\n"
        "Extract and summarize: \n"
        "- Observed writing style characteristics (slant, pressure, size, speed, spacing, margins, baseline, letter forms, connections, etc.)\n"
        "- Graphological interpretations of personality traits linked to those features\n"
        "- Overall psychological or personality impression"
    )
    docs = retriever.invoke(query)
    context = "\n\n".join(f"[Document {i+1}]\n{doc.page_content}\n" for i, doc in enumerate(docs))
    return context