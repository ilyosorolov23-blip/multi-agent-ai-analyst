"""
Shared LLM/embeddings factory. Every agent used to build its own
ChatGoogleGenerativeAI pointed straight at Gemini; now they all go
through this one factory, which points OpenAI-compatible LangChain
classes at the LiteLLM proxy (PROXY_BASE_URL / PROXY_API_KEY in .env).
"""
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import settings


def get_chat_llm(temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=settings.proxy_base_url,
        api_key=settings.proxy_api_key,
        model=settings.proxy_chat_model,
        temperature=temperature,
    )


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        base_url=settings.proxy_base_url,
        api_key=settings.proxy_api_key,
        model=settings.proxy_embedding_model,
    )
