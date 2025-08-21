from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from ..config import AppConfig

class EmbeddingFactory:
    """嵌入模型工厂类"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self._embeddings: Optional[Embeddings] = None
    
    def get_embeddings(self, provider: Optional[str] = None) -> Embeddings:
        """获取嵌入模型实例"""
        provider = provider or self.config.embedding_provider
        
        if self._embeddings is None:
            self._embeddings = self._create_embeddings(provider)
        
        return self._embeddings
    
    def _create_embeddings(self, provider: str) -> Embeddings:
        """创建嵌入模型实例"""
        if provider == "openai":
            return OpenAIEmbeddings(
                model=self.config.embedding_model,
                openai_api_key=self.config.openai_embed_api_key,
                openai_api_base=self.config.openai_embed_base_url,
                chunk_size=1000,
                max_retries=3
            )
        
        elif provider == "hf" or provider == "huggingface":
            return HuggingFaceEmbeddings(
                model_name=self.config.hf_embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
    
    def get_available_providers(self) -> list[str]:
        """获取可用的嵌入提供方列表"""
        available = []
        
        # 检查 OpenAI 配置
        if self.config.openai_embed_api_key:
            available.append("openai")
        
        # HuggingFace 总是可用（本地模型）
        available.append("huggingface")
        
        return available
    
    def test_embeddings(self, provider: str) -> bool:
        """测试嵌入模型是否可用"""
        try:
            embeddings = self._create_embeddings(provider)
            # 简单测试
            test_result = embeddings.embed_query("test")
            return len(test_result) > 0
        except Exception as e:
            print(f"Error testing embeddings for provider {provider}: {e}")
            return False
    
    def get_embedding_dimension(self, provider: Optional[str] = None) -> int:
        """获取嵌入向量维度"""
        provider = provider or self.config.embedding_provider
        
        # 常见模型的维度
        dimensions = {
            "text-embedding-3-large": 3072,
            "text-embedding-3-small": 1536,
            "text-embedding-ada-002": 1536,
            "BAAI/bge-large-zh-v1.5": 1024,
            "BAAI/bge-base-zh-v1.5": 768,
            "sentence-transformers/all-MiniLM-L6-v2": 384,
        }
        
        if provider == "openai":
            return dimensions.get(self.config.embedding_model, 1536)
        elif provider in ["hf", "huggingface"]:
            return dimensions.get(self.config.hf_embedding_model, 768)
        
        return 768  # 默认维度