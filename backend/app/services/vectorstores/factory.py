from typing import Optional
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import Milvus
from ..config import AppConfig
import os
from pathlib import Path

class VectorStoreFactory:
    """向量存储工厂类"""
    
    def __init__(self, config: AppConfig):
        self.config = config
    
    def get_vector_store(
        self, 
        collection_name: str, 
        embeddings: Embeddings,
        **kwargs
    ) -> VectorStore:
        """获取向量存储实例"""
        if self.config.vector_store == "chroma":
            return self._create_chroma_store(collection_name, embeddings, **kwargs)
        elif self.config.vector_store == "milvus":
            return self._create_milvus_store(collection_name, embeddings, **kwargs)
        else:
            raise ValueError(f"Unsupported vector store: {self.config.vector_store}")
    
    def _create_chroma_store(
        self, 
        collection_name: str, 
        embeddings: Embeddings,
        **kwargs
    ) -> Chroma:
        """创建 Chroma 向量存储"""
        # 确保存储目录存在
        persist_directory = Path(self.config.chroma_dir) / collection_name
        persist_directory.mkdir(parents=True, exist_ok=True)
        
        return Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(persist_directory),
            **kwargs
        )
    
    def _create_milvus_store(
        self, 
        collection_name: str, 
        embeddings: Embeddings,
        **kwargs
    ) -> Milvus:
        """创建 Milvus 向量存储"""
        connection_args = {
            "host": self.config.milvus_host,
            "port": self.config.milvus_port,
        }
        
        return Milvus(
            embedding_function=embeddings,
            collection_name=collection_name,
            connection_args=connection_args,
            **kwargs
        )
    
    def list_collections(self) -> list[str]:
        """列出所有集合"""
        if self.config.vector_store == "chroma":
            return self._list_chroma_collections()
        elif self.config.vector_store == "milvus":
            return self._list_milvus_collections()
        else:
            return []
    
    def _list_chroma_collections(self) -> list[str]:
        """列出 Chroma 集合"""
        chroma_dir = Path(self.config.chroma_dir)
        if not chroma_dir.exists():
            return []
        
        collections = []
        for item in chroma_dir.iterdir():
            if item.is_dir():
                collections.append(item.name)
        return collections
    
    def _list_milvus_collections(self) -> list[str]:
        """列出 Milvus 集合"""
        try:
            from pymilvus import connections, utility
            
            # 连接到 Milvus
            connections.connect(
                alias="default",
                host=self.config.milvus_host,
                port=self.config.milvus_port
            )
            
            # 获取所有集合
            collections = utility.list_collections()
            return collections
        except Exception as e:
            print(f"Error listing Milvus collections: {e}")
            return []
    
    def delete_collection(self, collection_name: str) -> bool:
        """删除集合"""
        try:
            if self.config.vector_store == "chroma":
                return self._delete_chroma_collection(collection_name)
            elif self.config.vector_store == "milvus":
                return self._delete_milvus_collection(collection_name)
            return False
        except Exception as e:
            print(f"Error deleting collection {collection_name}: {e}")
            return False
    
    def _delete_chroma_collection(self, collection_name: str) -> bool:
        """删除 Chroma 集合"""
        import shutil
        collection_dir = Path(self.config.chroma_dir) / collection_name
        if collection_dir.exists():
            shutil.rmtree(collection_dir)
            return True
        return False
    
    def _delete_milvus_collection(self, collection_name: str) -> bool:
        """删除 Milvus 集合"""
        try:
            from pymilvus import connections, utility
            
            connections.connect(
                alias="default",
                host=self.config.milvus_host,
                port=self.config.milvus_port
            )
            
            if utility.has_collection(collection_name):
                utility.drop_collection(collection_name)
                return True
            return False
        except Exception as e:
            print(f"Error deleting Milvus collection {collection_name}: {e}")
            return False