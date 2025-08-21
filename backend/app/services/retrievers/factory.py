from typing import Optional, Dict, Any
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers import (
    BM25Retriever,
    EnsembleRetriever,
    MultiQueryRetriever,
    ContextualCompressionRetriever
)
from langchain.retrievers.document_compressors import LLMChainExtractor
from ..config import AppConfig
from .reranker import MMRReranker, BasicReranker

class RetrieverFactory:
    """检索器工厂类"""
    
    def __init__(self, config: AppConfig):
        self.config = config
    
    def create_retriever(
        self, 
        vector_store: VectorStore,
        retriever_type: Optional[str] = None,
        search_kwargs: Dict[str, Any] = None
    ) -> BaseRetriever:
        """创建检索器"""
        retriever_type = retriever_type or self.config.retriever_type
        search_kwargs = search_kwargs or {"k": self.config.top_k_retrieval}
        
        if retriever_type == "vector":
            return self._create_vector_retriever(vector_store, search_kwargs)
        elif retriever_type == "bm25":
            return self._create_bm25_retriever(vector_store, search_kwargs)
        elif retriever_type == "hybrid":
            return self._create_hybrid_retriever(vector_store, search_kwargs)
        elif retriever_type == "multi_query":
            return self._create_multi_query_retriever(vector_store, search_kwargs)
        elif retriever_type == "contextual_compression":
            return self._create_contextual_compression_retriever(vector_store, search_kwargs)
        else:
            return self._create_vector_retriever(vector_store, search_kwargs)
    
    def _create_vector_retriever(
        self, 
        vector_store: VectorStore, 
        search_kwargs: Dict[str, Any]
    ) -> BaseRetriever:
        """创建向量检索器"""
        retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
        
        # 应用重排序
        if self.config.reranker_type == "mmr":
            reranker = MMRReranker(lambda_mult=self.config.mmr_lambda)
            return reranker.rerank_retriever(retriever)
        elif self.config.reranker_type == "basic":
            reranker = BasicReranker()
            return reranker.rerank_retriever(retriever)
        
        return retriever
    
    def _create_bm25_retriever(
        self, 
        vector_store: VectorStore, 
        search_kwargs: Dict[str, Any]
    ) -> BaseRetriever:
        """创建 BM25 检索器"""
        # 获取所有文档用于 BM25 索引
        try:
            # 尝试获取所有文档
            all_docs = vector_store.similarity_search("", k=10000)  # 获取大量文档
            if not all_docs:
                # 如果没有文档，回退到向量检索
                return self._create_vector_retriever(vector_store, search_kwargs)
            
            # 提取文档内容
            texts = [doc.page_content for doc in all_docs]
            
            # 创建 BM25 检索器
            bm25_retriever = BM25Retriever.from_texts(
                texts, 
                metadatas=[doc.metadata for doc in all_docs]
            )
            bm25_retriever.k = search_kwargs.get("k", self.config.top_k_retrieval)
            
            return bm25_retriever
        
        except Exception as e:
            print(f"Failed to create BM25 retriever: {e}")
            # 回退到向量检索
            return self._create_vector_retriever(vector_store, search_kwargs)
    
    def _create_hybrid_retriever(
        self, 
        vector_store: VectorStore, 
        search_kwargs: Dict[str, Any]
    ) -> BaseRetriever:
        """创建混合检索器（向量 + BM25）"""
        try:
            # 创建向量检索器
            vector_retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
            
            # 创建 BM25 检索器
            bm25_retriever = self._create_bm25_retriever(vector_store, search_kwargs)
            
            # 如果 BM25 创建失败，返回向量检索器
            if isinstance(bm25_retriever, type(vector_retriever)):
                return vector_retriever
            
            # 创建集成检索器
            ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[self.config.hybrid_alpha, 1 - self.config.hybrid_alpha]
            )
            
            return ensemble_retriever
        
        except Exception as e:
            print(f"Failed to create hybrid retriever: {e}")
            # 回退到向量检索
            return self._create_vector_retriever(vector_store, search_kwargs)
    
    def _create_multi_query_retriever(
        self, 
        vector_store: VectorStore, 
        search_kwargs: Dict[str, Any]
    ) -> BaseRetriever:
        """创建多查询检索器"""
        try:
            from ..llms.factory import LLMFactory
            
            # 获取 LLM
            llm_factory = LLMFactory(self.config)
            llm = llm_factory.get_llm()
            
            # 创建基础检索器
            base_retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
            
            # 创建多查询检索器
            multi_query_retriever = MultiQueryRetriever.from_llm(
                retriever=base_retriever,
                llm=llm
            )
            
            return multi_query_retriever
        
        except Exception as e:
            print(f"Failed to create multi-query retriever: {e}")
            # 回退到向量检索
            return self._create_vector_retriever(vector_store, search_kwargs)
    
    def _create_contextual_compression_retriever(
        self, 
        vector_store: VectorStore, 
        search_kwargs: Dict[str, Any]
    ) -> BaseRetriever:
        """创建上下文压缩检索器"""
        try:
            from ..llms.factory import LLMFactory
            
            # 获取 LLM
            llm_factory = LLMFactory(self.config)
            llm = llm_factory.get_llm()
            
            # 创建基础检索器
            base_retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
            
            # 创建压缩器
            compressor = LLMChainExtractor.from_llm(llm)
            
            # 创建上下文压缩检索器
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )
            
            return compression_retriever
        
        except Exception as e:
            print(f"Failed to create contextual compression retriever: {e}")
            # 回退到向量检索
            return self._create_vector_retriever(vector_store, search_kwargs)
    
    def get_available_types(self) -> list[str]:
        """获取可用的检索器类型"""
        return [
            "vector",
            "bm25", 
            "hybrid",
            "multi_query",
            "contextual_compression"
        ]
    
    def test_retriever(self, retriever: BaseRetriever, query: str = "test") -> bool:
        """测试检索器是否正常工作"""
        try:
            results = retriever.get_relevant_documents(query)
            return True
        except Exception as e:
            print(f"Retriever test failed: {e}")
            return False