import os
from typing import Literal
from dotenv import load_dotenv

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

openai_dict = {
    "models": [
        "gpt-4",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "o3",
        "o3-pro",
        "o4-mini",
    ],
    "embeddings": [
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002",
    ],
}


class OpenAIStuff:
    def __init__(
        self,
        embd_model: Literal[
            # "text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"
            *openai_dict["embeddings"]
        ] = "text-embedding-3-small",
        embd_dims: int = 1024,
        llm_model: Literal[
            # "gpt-4.1", "gpt-4.1-mini", "gpt-5", "gpt-5-mini"
            *openai_dict["models"]
        ] = "gpt-4.1-mini",
        llm_temp: float = 0.3,
    ):
        load_dotenv("src/Secrets/API_Keys.env")
        api_key = os.getenv("OPENAI_API_KEY")
        self.embedding = OpenAIEmbeddings(
            model=embd_model,
            dimensions=embd_dims,
            api_key=api_key,
        )
        self.llm = ChatOpenAI(
            model=llm_model,
            api_key=api_key,
            temperature=llm_temp,
        )


groq_dict = {
    "models": [
        "moonshotai/kimi-k2-instruct-0905",
        "openai/gpt-oss-20b",
        "qwen/qwen3-32b",
    ],
    "embeddings": [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-MiniLM-L12-v2",
        "sentence-transformers/all-mpnet-base-v2",
    ],
}


class GroqStuff:
    def __init__(
        self,
        embd_model: Literal[
            # "sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L12-v2"
            *groq_dict["embeddings"]
        ] = "sentence-transformers/all-MiniLM-L6-v2",
        embd_dims: int = 1024,
        llm_model: Literal[
            # "moonshotai/kimi-k2-instruct-0905", "openai/gpt-oss-20b", "qwen/qwen3-32b"
            *groq_dict["models"]
        ] = "moonshotai/kimi-k2-instruct-0905",
        llm_temp: float = 0.3,
    ):
        load_dotenv("src/Secrets/API_Keys.env")
        api_key = os.getenv("GROQ_API_KEY")
        self.embedding = HuggingFaceEmbeddings(
            model_name=embd_model,
            dimensions=embd_dims,
        )
        self.llm = ChatOpenAI(
            model=llm_model,
            api_key=api_key,
            temperature=llm_temp,
        )


class OtherConfig:
    def __init__(self):
        self.chunk_size = 500
        self.chunk_overlap = 50
        self.default_urls = [
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
        ]
