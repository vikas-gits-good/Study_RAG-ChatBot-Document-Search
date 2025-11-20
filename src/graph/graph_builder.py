from langgraph.graph import StateGraph, START, END
from langchain.chat_models import BaseChatModel
from langchain_core.retrievers import RetrieverLike

from src.state.rag_state import RAGState
from src.nodes.nodes import RAGNodes


class GraphBuilder:
    def __init__(
        self,
        retriever: RetrieverLike,
        llm_model: BaseChatModel,
    ):
        self.nodes = RAGNodes(retriever, llm_model)
        self.graph = None

    def build(self):
        # StateGraph
        bldr = StateGraph(RAGState)

        # Nodes
        bldr.add_node("retriever", self.nodes.retrieve_docs)
        bldr.add_node("responder", self.nodes.generate_answer)
        bldr.set_entry_point("retriever")

        # Edges
        bldr.add_edge("retriever", "responder")
        bldr.add_edge("responder", END)

        # Compile
        self.graph = bldr.compile()
        return self.graph

    def run(self, question: str) -> dict:
        _ = self.build() if not self.graph else None
        init_state = RAGState(question=question)
        return self.graph.invoke(init_state)
