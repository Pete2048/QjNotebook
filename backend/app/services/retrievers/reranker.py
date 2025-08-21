from typing import List, Dict, Any, Optional, Callable
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class BaseReranker:
    """基础重排序器"""
    
    def __init__(self):
        pass
    
    def rerank_documents(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: Optional[int] = None
    ) -> List[Document]:
        """重排序文档"""
        raise NotImplementedError
    
    def rerank_retriever(self, retriever: BaseRetriever) -> BaseRetriever:
        """包装检索器以添加重排序功能"""
        return RerankedRetriever(retriever, self)

class BasicReranker(BaseReranker):
    """基础重排序器（按相似度分数排序）"""
    
    def rerank_documents(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: Optional[int] = None
    ) -> List[Document]:
        """按现有分数重排序"""
        # 按元数据中的分数排序（如果存在）
        scored_docs = []
        for doc in documents:
            score = doc.metadata.get('score', 0.0)
            scored_docs.append((doc, score))
        
        # 按分数降序排序
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # 返回排序后的文档
        reranked = [doc for doc, score in scored_docs]
        
        if top_k:
            reranked = reranked[:top_k]
        
        return reranked

class MMRReranker(BaseReranker):
    """最大边际相关性（MMR）重排序器"""
    
    def __init__(self, lambda_mult: float = 0.5, k1: int = 20):
        super().__init__()
        self.lambda_mult = lambda_mult  # 相关性与多样性的权衡参数
        self.k1 = k1  # 初始检索数量
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    
    def rerank_documents(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: Optional[int] = None
    ) -> List[Document]:
        """使用 MMR 算法重排序"""
        if not documents:
            return documents
        
        if len(documents) <= 1:
            return documents
        
        try:
            # 准备文本数据
            doc_texts = [doc.page_content for doc in documents]
            all_texts = [query] + doc_texts
            
            # 计算 TF-IDF 向量
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            query_vec = tfidf_matrix[0:1]  # 查询向量
            doc_vecs = tfidf_matrix[1:]     # 文档向量
            
            # 计算查询与文档的相似度
            query_similarities = cosine_similarity(query_vec, doc_vecs).flatten()
            
            # MMR 算法
            selected_indices = []
            remaining_indices = list(range(len(documents)))
            
            # 选择第一个文档（相似度最高的）
            first_idx = np.argmax(query_similarities)
            selected_indices.append(first_idx)
            remaining_indices.remove(first_idx)
            
            # 迭代选择剩余文档
            target_count = min(top_k or len(documents), len(documents))
            
            while len(selected_indices) < target_count and remaining_indices:
                mmr_scores = []
                
                for idx in remaining_indices:
                    # 计算与查询的相似度
                    query_sim = query_similarities[idx]
                    
                    # 计算与已选择文档的最大相似度
                    max_sim_to_selected = 0.0
                    if selected_indices:
                        selected_vecs = doc_vecs[selected_indices]
                        current_vec = doc_vecs[idx:idx+1]
                        similarities_to_selected = cosine_similarity(current_vec, selected_vecs).flatten()
                        max_sim_to_selected = np.max(similarities_to_selected)
                    
                    # 计算 MMR 分数
                    mmr_score = (self.lambda_mult * query_sim - 
                               (1 - self.lambda_mult) * max_sim_to_selected)
                    mmr_scores.append((idx, mmr_score))
                
                # 选择 MMR 分数最高的文档
                best_idx, best_score = max(mmr_scores, key=lambda x: x[1])
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            
            # 返回重排序的文档
            reranked_docs = [documents[i] for i in selected_indices]
            
            # 更新文档的分数元数据
            for i, doc in enumerate(reranked_docs):
                doc.metadata['mmr_rank'] = i + 1
                doc.metadata['original_score'] = query_similarities[selected_indices[i]]
            
            return reranked_docs
        
        except Exception as e:
            print(f"MMR reranking failed: {e}")
            # 回退到基础排序
            return BasicReranker().rerank_documents(query, documents, top_k)

class RerankedRetriever(BaseRetriever):
    """带重排序功能的检索器包装器"""
    
    def __init__(self, base_retriever: BaseRetriever, reranker: BaseReranker):
        super().__init__()
        self.base_retriever = base_retriever
        self.reranker = reranker
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """获取相关文档并重排序"""
        # 使用基础检索器获取文档
        documents = self.base_retriever.get_relevant_documents(
            query, callbacks=run_manager.get_child()
        )
        
        # 使用重排序器重排序
        reranked_documents = self.reranker.rerank_documents(query, documents)
        
        return reranked_documents