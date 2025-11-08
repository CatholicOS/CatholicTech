# retrieval.py
import os
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Define where the Church documents live
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data")
DB_PATH = os.path.join(os.path.dirname(__file__), "../chroma_db")

def build_vectorstore():
    """Loads text files, splits them, and creates a searchable vector database."""
    print("📖 Loading documents...")
    loader = DirectoryLoader(DATA_PATH, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()

    print(f"✅ Loaded {len(documents)} documents. Splitting text...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    print("⚙️ Creating embeddings and database...")
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=DB_PATH)
    vectordb.persist()

    print("🧠 Vector database built successfully!")
    return vectordb

def get_vectorstore():
    """Loads an existing vectorstore if present, otherwise builds it."""
    if os.path.exists(DB_PATH):
        print("🔍 Loading existing database...")
        embeddings = OpenAIEmbeddings()
        vectordb = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    else:
        vectordb = build_vectorstore()
    return vectordb