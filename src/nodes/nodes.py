from langchain.chat_models import BaseChatModel
from langchain_core.retrievers import RetrieverLike

from src.state.rag_state import RAGState


class RAGNodes:
    def __init__(
        self,
        retriever: RetrieverLike,
        llm_model: BaseChatModel,
    ) -> None:
        self.retriever = retriever
        self.llm_model = llm_model

    def retrieve_docs(self, state: RAGState) -> RAGState:
        docs = self.retriever.invoke(state.question)
        return state.model_copy(update={"retrieved_docs": docs})

    def generate_answer(self, state: RAGState) -> RAGState:
        context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])
        prompt = f"""Answer the question concisely within 5 sectences 
        using the context given.
        
        Context:
        {context}
        
        Question: {state.question}
        
        Answer:        
        """
        response = self.llm_model.invoke(prompt)
        return state.model_copy(update={"answer": response.content})
