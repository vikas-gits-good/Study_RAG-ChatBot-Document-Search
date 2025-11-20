from src.documents.document_processor import DocumentProcessor
from src.vector_store.vectorstore import VectorStore


if __name__ == "__main__":
    data = "data/attention.pdf"

    docs = DocumentProcessor().load_urls_from_file([data])
    vs = VectorStore()
    vs.create(docs)
    print(vs.retrieve("What is transformer"))
