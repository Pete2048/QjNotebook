import os
from typing import List
from pydantic import BaseModel

class AppConfig(BaseModel):
    """应用配置，支持 LangChain 组件配置"""
    app_env: str = os.getenv("APP_ENV", "dev")
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))

    # 向量存储
    vector_store: str = os.getenv("VECTOR_STORE", "chroma")  # milvus | chroma
    chroma_dir: str = os.getenv("CHROMA_DIR", "app/storage/chroma")
    milvus_host: str = os.getenv("MILVUS_HOST", "localhost")
    milvus_port: str = os.getenv("MILVUS_PORT", "19530")

    # 文档处理配置
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # 检索配置
    retriever_type: str = os.getenv("RETRIEVER_TYPE", "hybrid")  # vector, bm25, hybrid
    hybrid_alpha: float = float(os.getenv("HYBRID_ALPHA", "0.6"))  # 向量检索权重
    reranker_type: str = os.getenv("RERANKER_TYPE", "mmr")  # basic, mmr
    mmr_lambda: float = float(os.getenv("MMR_LAMBDA", "0.7"))  # 多样性参数
    top_k_retrieval: int = int(os.getenv("TOP_K_RETRIEVAL", "10"))  # 检索数量

    # Embeddings
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "openai")  # openai | hf
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    openai_embed_base_url: str = os.getenv("OPENAI_EMBED_BASE_URL", "https://api.openai.com/v1")
    openai_embed_api_key: str = os.getenv("OPENAI_EMBED_API_KEY", os.getenv("LLM_API_KEY", ""))  # 复用密钥
    
    # HuggingFace Embeddings
    hf_embedding_model: str = os.getenv("HF_EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
    
    # LLM 多提供方
    # 全局默认
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # 提供方列表（优先级顺序），例如 "openai,deepseek,gemini,doubao"
    providers_raw: str = os.getenv("LLM_PROVIDERS", "openai")
    llm_providers: List[str] = []

    # OpenAI
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"))
    openai_api_key: str = os.getenv("OPENAI_API_KEY", os.getenv("LLM_API_KEY", ""))
    openai_model: str = os.getenv("OPENAI_MODEL", os.getenv("LLM_MODEL", "gpt-4o-mini"))

    # DeepSeek (OpenAI 兼容)
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    # Gemini (Google Generative Language API)
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

    # Doubao (ByteDance/Volcano Ark, OpenAI 兼容)
    doubao_base_url: str = os.getenv("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    doubao_api_key: str = os.getenv("DOUBAO_API_KEY", "")
    doubao_model: str = os.getenv("DOUBAO_MODEL", "ep-20250609121409-9882n")

    # 知识图谱
    enable_kg: bool = os.getenv("ENABLE_KG", "false").lower() == "true"
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password")

def load_config() -> AppConfig:
    """加载配置"""
    cfg = AppConfig()
    # 解析 LLM_PROVIDERS 环境变量
    providers_str = os.getenv("LLM_PROVIDERS", "openai,deepseek,gemini,doubao")
    cfg.llm_providers = [p.strip().lower() for p in providers_str.split(",") if p.strip()]
    return cfg