import warnings
warnings.filterwarnings(action='ignore')
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
load_dotenv()


def fetch_document_chunks():
    """
    Load and split all PDF files from the designated folder into manageable text chunks.

    This function serves as the document ingestion step for the RAG pipeline.
    It:
    - Loads every PDF file found in the ./RAG_Documents directory
    - Splits documents into overlapping chunks optimized for vector embedding
      and retrieval in graphology/handwriting analysis context

    Configuration (hardcoded):
    - Source folder: ./RAG_Documents
    - Chunk size: 850 characters
    - Chunk overlap: 120 characters
    - Splitter: RecursiveCharacterTextSplitter with common separators
    - Includes start_index metadata for potential future reference/traceability

    Returns
    -------
    list[langchain_core.documents.Document]
        List of document chunks ready to be embedded and stored in vector database.
        Each chunk contains:
        - page_content: the text fragment
        - metadata: source file, page number, start_index

    Raises
    ------
    FileNotFoundError
        If the ./RAG_Documents directory does not exist
    ValueError
        If no PDF files are found or directory is empty

    Notes
    -----
    - This function loads and splits documents **every time it is called**.
    - In production, consider caching the chunks or using a persistent vector store
      to avoid repeated disk I/O and splitting.
    - Current parameters (850/120) are reasonable for most sentence-transformers
      models and graphology-related documents.
    """

    PDF_FOLDER = "./RAG_Documents"
    CHUNK_SIZE = 850
    CHUNK_OVERLAP = 120

    loader = PyPDFDirectoryLoader(PDF_FOLDER)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True
    )

    chunks = text_splitter.split_documents(docs)
    return chunks