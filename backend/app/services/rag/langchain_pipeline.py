from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from ..llms.factory import LLMFactory
from ..embeddings.factory import EmbeddingFactory
from ..vectorstores.factory import VectorStoreFactory
from ..retrievers.factory import RetrieverFactory
from ..config import AppConfig
from .prompts import RAGPrompts
import time

class LangChainRAGPipeline:
    """基于 LangChain 的 RAG 管道"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.llm_factory = LLMFactory(config)
        self.embedding_factory = EmbeddingFactory(config)
        self.vector_store_factory = VectorStoreFactory(config)
        self.retriever_factory = RetrieverFactory(config)
        self.prompts = RAGPrompts()
        
        # 当前活跃的提供方
        self.current_provider = config.llm_provider
    
    def query(
        self, 
        question: str, 
        notebook_id: str, 
        top_k: int = 5,
        provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """执行 RAG 查询"""
        start_time = time.time()
        
        try:
            # 使用指定的提供方或当前活跃的提供方
            llm_provider = provider or self.current_provider
            
            # 获取组件
            llm = self.llm_factory.get_llm(llm_provider)
            embeddings = self.embedding_factory.get_embeddings()
            vector_store = self.vector_store_factory.get_vector_store(
                collection_name=f"notebook_{notebook_id}",
                embeddings=embeddings
            )
            
            # 创建检索器
            retriever = self.retriever_factory.create_retriever(
                vector_store=vector_store,
                search_kwargs={"k": top_k}
            )
            
            # 检索相关文档
            relevant_docs = retriever.get_relevant_documents(question)
            
            if not relevant_docs:
                return {
                    "question": question,
                    "answer": "抱歉，我在知识库中没有找到相关信息来回答您的问题。请尝试上传相关文档或重新表述您的问题。",
                    "sources": [],
                    "meta": {
                        "provider": llm_provider,
                        "retrieval_time": time.time() - start_time,
                        "total_time": time.time() - start_time,
                        "retrieved_docs": 0
                    }
                }
            
            # 构建上下文
            context = self._format_context(relevant_docs)
            
            # 创建提示模板
            prompt = self.prompts.get_qa_prompt()
            
            # 构建 RAG 链
            rag_chain = (
                {
                    "context": lambda x: context,
                    "question": RunnablePassthrough()
                }
                | prompt
                | llm
                | StrOutputParser()
            )
            
            # 执行查询
            retrieval_time = time.time() - start_time
            answer = rag_chain.invoke(question)
            total_time = time.time() - start_time
            
            # 格式化源文档
            sources = self._format_sources(relevant_docs)
            
            return {
                "question": question,
                "answer": answer,
                "sources": sources,
                "meta": {
                    "provider": llm_provider,
                    "retrieval_time": retrieval_time,
                    "generation_time": total_time - retrieval_time,
                    "total_time": total_time,
                    "retrieved_docs": len(relevant_docs),
                    "context_length": len(context)
                }
            }
            
        except Exception as e:
            return {
                "question": question,
                "answer": f"查询过程中发生错误: {str(e)}",
                "sources": [],
                "meta": {
                    "provider": provider or self.current_provider,
                    "error": str(e),
                    "total_time": time.time() - start_time
                }
            }
    
    def query_with_history(
        self,
        question: str,
        notebook_id: str,
        chat_history: List[Dict[str, str]] = None,
        top_k: int = 5,
        provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """带历史记录的查询"""
        start_time = time.time()
        
        try:
            llm_provider = provider or self.current_provider
            llm = self.llm_factory.get_llm(llm_provider)
            embeddings = self.embedding_factory.get_embeddings()
            vector_store = self.vector_store_factory.get_vector_store(
                collection_name=f"notebook_{notebook_id}",
                embeddings=embeddings
            )
            
            # 创建检索器
            retriever = self.retriever_factory.create_retriever(
                vector_store=vector_store,
                search_kwargs={"k": top_k}
            )
            
            # 如果有历史记录，重写问题以包含上下文
            if chat_history:
                question = self._rewrite_question_with_history(question, chat_history, llm)
            
            # 检索相关文档
            relevant_docs = retriever.get_relevant_documents(question)
            
            if not relevant_docs:
                return self._empty_response(question, llm_provider, start_time)
            
            # 构建上下文
            context = self._format_context(relevant_docs)
            
            # 创建带历史记录的提示模板
            prompt = self.prompts.get_conversational_qa_prompt()
            
            # 格式化历史记录
            history_text = self._format_chat_history(chat_history or [])
            
            # 构建 RAG 链
            rag_chain = (
                {
                    "context": lambda x: context,
                    "chat_history": lambda x: history_text,
                    "question": RunnablePassthrough()
                }
                | prompt
                | llm
                | StrOutputParser()
            )
            
            # 执行查询
            retrieval_time = time.time() - start_time
            answer = rag_chain.invoke(question)
            total_time = time.time() - start_time
            
            # 格式化源文档
            sources = self._format_sources(relevant_docs)
            
            return {
                "question": question,
                "answer": answer,
                "sources": sources,
                "meta": {
                    "provider": llm_provider,
                    "retrieval_time": retrieval_time,
                    "generation_time": total_time - retrieval_time,
                    "total_time": total_time,
                    "retrieved_docs": len(relevant_docs),
                    "context_length": len(context),
                    "has_history": bool(chat_history)
                }
            }
            
        except Exception as e:
            return {
                "question": question,
                "answer": f"查询过程中发生错误: {str(e)}",
                "sources": [],
                "meta": {
                    "provider": provider or self.current_provider,
                    "error": str(e),
                    "total_time": time.time() - start_time
                }
            }
    
    def set_provider(self, provider: str) -> bool:
        """设置当前 LLM 提供方"""
        try:
            # 测试提供方是否可用
            llm = self.llm_factory.get_llm(provider)
            # 简单测试
            test_response = llm.invoke("Hello")
            if test_response.content:
                self.current_provider = provider
                return True
            return False
        except Exception:
            return False
    
    def get_available_providers(self) -> List[str]:
        """获取可用的 LLM 提供方"""
        return self.llm_factory.get_available_providers()
    
    def _format_context(self, documents: List[Document]) -> str:
        """格式化上下文文档"""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', 'Unknown')
            content = doc.page_content.strip()
            context_parts.append(f"[文档{i}] 来源: {source}\n{content}")
        
        return "\n\n".join(context_parts)
    
    def _format_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """格式化源文档信息"""
        sources = []
        for doc in documents:
            source_info = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": doc.metadata.get('score', 0.0)
            }
            sources.append(source_info)
        
        return sources
    
    def _format_chat_history(self, chat_history: List[Dict[str, str]]) -> str:
        """格式化聊天历史"""
        if not chat_history:
            return ""
        
        history_parts = []
        for entry in chat_history[-5:]:  # 只保留最近5轮对话
            if "question" in entry and "answer" in entry:
                history_parts.append(f"用户: {entry['question']}")
                history_parts.append(f"助手: {entry['answer']}")
        
        return "\n".join(history_parts)
    
    def _rewrite_question_with_history(
        self, 
        question: str, 
        chat_history: List[Dict[str, str]], 
        llm
    ) -> str:
        """基于历史记录重写问题"""
        if not chat_history:
            return question
        
        try:
            history_text = self._format_chat_history(chat_history)
            rewrite_prompt = self.prompts.get_question_rewrite_prompt()
            
            rewrite_chain = (
                {
                    "chat_history": lambda x: history_text,
                    "question": RunnablePassthrough()
                }
                | rewrite_prompt
                | llm
                | StrOutputParser()
            )
            
            rewritten = rewrite_chain.invoke(question)
            return rewritten.strip()
        
        except Exception:
            # 如果重写失败，返回原问题
            return question
    
    def _empty_response(self, question: str, provider: str, start_time: float) -> Dict[str, Any]:
        """空响应"""
        return {
            "question": question,
            "answer": "抱歉，我在知识库中没有找到相关信息来回答您的问题。请尝试上传相关文档或重新表述您的问题。",
            "sources": [],
            "meta": {
                "provider": provider,
                "total_time": time.time() - start_time,
                "retrieved_docs": 0
            }
        }