from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import os
from pathlib import Path
from ..vectorstores.factory import VectorStoreFactory
from ..embeddings.factory import EmbeddingFactory
from ..document_processor.loader import DocumentLoader
from ..document_processor.chunker import TextChunker
from .langchain_pipeline import LangChainRAGPipeline
from ..config import AppConfig

class NotebookManager:
    """笔记本管理器"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.storage_dir = Path("app/storage/notebooks")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self.embedding_factory = EmbeddingFactory(config)
        self.vector_store_factory = VectorStoreFactory(config)
        self.document_loader = DocumentLoader()
        self.text_chunker = TextChunker(config)
        self.rag_pipeline = LangChainRAGPipeline(config)
    
    def create_notebook(self, name: str) -> Dict[str, Any]:
        """创建新笔记本"""
        notebook_id = self._generate_id()
        notebook = {
            "id": notebook_id,
            "name": name,
            "created_at": int(datetime.now().timestamp()),
            "updated_at": int(datetime.now().timestamp()),
            "document_count": 0,
            "metadata": {}
        }
        
        # 保存笔记本元数据
        self._save_notebook_metadata(notebook)
        
        # 初始化向量存储
        self._initialize_vector_store(notebook_id)
        
        return notebook
    
    def list_notebooks(self) -> List[Dict[str, Any]]:
        """列出所有笔记本"""
        notebooks = []
        for notebook_file in self.storage_dir.glob("*/metadata.json"):
            try:
                with open(notebook_file, 'r', encoding='utf-8') as f:
                    notebook = json.load(f)
                    notebooks.append(notebook)
            except Exception as e:
                print(f"Error loading notebook {notebook_file}: {e}")
        
        # 按创建时间倒序排列
        notebooks.sort(key=lambda x: x.get('created_at', 0), reverse=True)
        return notebooks
    
    def get_notebook(self, notebook_id: str) -> Optional[Dict[str, Any]]:
        """获取笔记本信息"""
        metadata_file = self.storage_dir / notebook_id / "metadata.json"
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    
    def delete_notebook(self, notebook_id: str) -> bool:
        """删除笔记本"""
        notebook_dir = self.storage_dir / notebook_id
        if not notebook_dir.exists():
            return False
        
        try:
            # 删除向量存储
            self._cleanup_vector_store(notebook_id)
            
            # 删除文件夹
            import shutil
            shutil.rmtree(notebook_dir)
            return True
        except Exception as e:
            print(f"Error deleting notebook {notebook_id}: {e}")
            return False
    
    def add_documents(self, notebook_id: str, texts: List[str], metadata: Dict[str, Any] = None) -> bool:
        """向笔记本添加文档"""
        notebook = self.get_notebook(notebook_id)
        if not notebook:
            return False
        
        try:
            # 文本分块
            chunks = []
            for i, text in enumerate(texts):
                text_chunks = self.text_chunker.chunk_text(text)
                for j, chunk in enumerate(text_chunks):
                    chunk_metadata = {
                        "source": metadata.get("source", f"document_{i}") if metadata else f"document_{i}",
                        "chunk_id": f"{i}_{j}",
                        "notebook_id": notebook_id,
                        **(metadata or {})
                    }
                    chunks.append((chunk, chunk_metadata))
            
            # 添加到向量存储
            vector_store = self._get_vector_store(notebook_id)
            texts_to_add = [chunk[0] for chunk in chunks]
            metadatas_to_add = [chunk[1] for chunk in chunks]
            
            vector_store.add_texts(texts_to_add, metadatas=metadatas_to_add)
            
            # 更新笔记本元数据
            notebook["document_count"] += len(texts)
            notebook["updated_at"] = int(datetime.now().timestamp())
            self._save_notebook_metadata(notebook)
            
            return True
        except Exception as e:
            print(f"Error adding documents to notebook {notebook_id}: {e}")
            return False
    
    def query_notebook(self, notebook_id: str, question: str, top_k: int = 5) -> Dict[str, Any]:
        """查询笔记本"""
        notebook = self.get_notebook(notebook_id)
        if not notebook:
            return {"error": "Notebook not found"}
        
        try:
            # 使用 RAG 管道进行查询
            result = self.rag_pipeline.query(
                question=question,
                notebook_id=notebook_id,
                top_k=top_k
            )
            return result
        except Exception as e:
            print(f"Error querying notebook {notebook_id}: {e}")
            return {"error": str(e)}
    
    def _generate_id(self) -> str:
        """生成唯一ID"""
        import uuid
        return str(uuid.uuid4())
    
    def _save_notebook_metadata(self, notebook: Dict[str, Any]):
        """保存笔记本元数据"""
        notebook_dir = self.storage_dir / notebook["id"]
        notebook_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_file = notebook_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, ensure_ascii=False, indent=2)
    
    def _initialize_vector_store(self, notebook_id: str):
        """初始化向量存储"""
        try:
            vector_store = self._get_vector_store(notebook_id)
            # 向量存储会在第一次添加文档时自动初始化
        except Exception as e:
            print(f"Error initializing vector store for notebook {notebook_id}: {e}")
    
    def _get_vector_store(self, notebook_id: str):
        """获取笔记本的向量存储"""
        embeddings = self.embedding_factory.get_embeddings()
        return self.vector_store_factory.get_vector_store(
            collection_name=f"notebook_{notebook_id}",
            embeddings=embeddings
        )
    
    def _cleanup_vector_store(self, notebook_id: str):
        """清理向量存储"""
        try:
            if self.config.vector_store == "chroma":
                # Chroma 会在删除文件夹时自动清理
                pass
            elif self.config.vector_store == "milvus":
                # Milvus 需要显式删除集合
                vector_store = self._get_vector_store(notebook_id)
                if hasattr(vector_store, 'drop_collection'):
                    vector_store.drop_collection()
        except Exception as e:
            print(f"Error cleaning up vector store for notebook {notebook_id}: {e}")