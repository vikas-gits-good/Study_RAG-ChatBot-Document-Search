from typing import List, Optional

from langchain.agents import create_agent
from langchain_core.tools import Tool
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.retrievers import RetrieverLike
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain.chat_models import BaseChatModel

from src.state.rag_state import RAGState


class RAGNodes:
    def __init__(
        self,
        retriever: RetrieverLike,
        llm_model: BaseChatModel,
    ) -> None:
        self.retriever = retriever
        self.llm_model = llm_model
        self.agent = None

    def _build_tools(self) -> List[Tool]:
        def rtrv_tool(query: str) -> str:
            docs = self.retriever.invoke(query)
            if not docs:
                return "No documents found."
            merged = []
            for i, d in enumerate(docs[:8], start=1):
                meta = d.metadata if hasattr(d, "metadata") else {}
                title = meta.get("title") or meta.get("source") or f"doc_{i}"
                merged.append(f"[{i}] {title}\n{d.page_content}")
            return "\n\n".join(merged)

        wiki_tool = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=3, lang="en")
        )

        tool_rtrv = Tool(
            name="retriever",
            description="Retriever tool to get data from indexed vector store.",
            func=rtrv_tool,
        )

        tool_wiki = Tool(
            name="wikipedia",
            description="Wikipedia tool to search for general knowledge.",
            func=wiki_tool.run,
        )
        return [tool_rtrv, tool_wiki]

    def _build_agent(self):
        tools_list = self._build_tools()
        sys_prmt = """You are a helpful RAG agent.
        Prefer 'retriever' for user-provided docs and use 'wikipedia' for general knowledge.
        Return only final useful answer.
        """
        self._agent = create_agent(
            model=self.llm_model,
            tools=tools_list,
            system_prompt=sys_prmt,
        )

    def retrieve_docs(self, state: RAGState) -> RAGState:
        docs = self.retriever.invoke(state.question)
        return state.model_copy(update={"retrieved_docs": docs})

    def generate_answer(self, state: RAGState) -> RAGState:
        _ = self._build_agent() if not self.agent else None
        result = self._agent.invoke(
            {"messages": [HumanMessage(content=state.question)]}
        )
        messages = result.get("messages", [])
        answer: Optional[str] = ""

        if messages:
            ans_msg = messages[-1]
            answer = getattr(ans_msg, "content", None)

        return state.model_copy(update={"answer": answer or "Couldn't find answer."})
