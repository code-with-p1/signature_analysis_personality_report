import warnings
warnings.filterwarnings(action='ignore')
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
load_dotenv()


def fetch_document_chunks():
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