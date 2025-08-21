from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
import openai
import requests
import json
from langchain_core.embeddings import Embeddings

class BaseEmbeddingProvider(Embeddings, ABC):
    """嵌入提供者基类"""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档列表"""
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """嵌入查询文本"""
        pass

class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI 嵌入提供者"""
    
    def __init__(self, api_key: str, model_name: str = "text-embedding-ada-002", **kwargs):
        super().__init__(model_name, **kwargs)
        self.client = openai.OpenAI(api_key=api_key)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档列表"""
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"OpenAI embedding error: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入查询文本"""
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=[text]
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"OpenAI embedding error: {e}")
            raise

class DeepSeekEmbeddingProvider(BaseEmbeddingProvider):
    """DeepSeek 嵌入提供者"""
    
    def __init__(self, api_key: str, model_name: str = "deepseek-embedding", **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_key = api_key
        self.base_url = kwargs.get('base_url', 'https://api.deepseek.com/v1')
    
    def _make_request(self, texts: List[str]) -> List[List[float]]:
        """发送嵌入请求"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': self.model_name,
            'input': texts
        }
        
        try:
            response = requests.post(
                f'{self.base_url}/embeddings',
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return [item['embedding'] for item in result['data']]
        except Exception as e:
            print(f"DeepSeek embedding error: {e}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档列表"""
        return self._make_request(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入查询文本"""
        result = self._make_request([text])
        return result[0]

class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):
    """HuggingFace 嵌入提供者"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", **kwargs):
        super().__init__(model_name, **kwargs)
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except ImportError:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档列表"""
        try:
            embeddings = self.model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            print(f"HuggingFace embedding error: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入查询文本"""
        try:
            embedding = self.model.encode([text])
            return embedding[0].tolist()
        except Exception as e:
            print(f"HuggingFace embedding error: {e}")
            raise

class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """本地嵌入提供者（使用预训练模型）"""
    
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, **kwargs)
        self.model_path = model_path
        self._load_model()
    
    def _load_model(self):
        """加载本地模型"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_path)
        except ImportError:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")
        except Exception as e:
            print(f"Failed to load local model: {e}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档列表"""
        try:
            embeddings = self.model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            print(f"Local embedding error: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入查询文本"""
        try:
            embedding = self.model.encode([text])
            return embedding[0].tolist()
        except Exception as e:
            print(f"Local embedding error: {e}")
            raise

class MockEmbeddingProvider(BaseEmbeddingProvider):
    """模拟嵌入提供者（用于测试）"""
    
    def __init__(self, dimension: int = 768, **kwargs):
        super().__init__("mock-embedding", **kwargs)
        self.dimension = dimension
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """生成模拟嵌入"""
        import random
        embeddings = []
        for text in texts:
            # 基于文本内容生成确定性的随机嵌入
            random.seed(hash(text) % (2**32))
            embedding = [random.random() for _ in range(self.dimension)]
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """生成模拟查询嵌入"""
        import random
        random.seed(hash(text) % (2**32))
        return [random.random() for _ in range(self.dimension)]