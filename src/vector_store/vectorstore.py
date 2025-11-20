from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


from src.config import OpenAIStuff


class VectorStore:
    def __init__(self) -> None:
        self.embedding = OpenAIStuff().embedding
        self.rtrv = None

    def create(self, documents: List[Document]):
        """Method to create a FAISS vector store using documents.

        Args:
            documents (List[Document]): A list of chunked data to process
        """
        vec_str = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding,
        )
        self.rtrv = vec_str.as_retriever()

    def get(self):
        if not self.rtrv:
            raise ValueError(
                "Vector store not initialised. Call 'VectorStore().create(docs)' first"
            )
        return self.rtrv

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        if not self.rtrv:
            raise ValueError(
                "Vector store not initialised. Call 'VectorStore().create(docs)' first"
            )
        return self.rtrv.invoke(query, search_kwargs={"k", k})
