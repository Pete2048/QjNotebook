from typing import List, Dict, Any, Optional
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter
)
from langchain_core.documents import Document
from ..config import AppConfig
import re

class TextChunker:
    """文本分块器，支持多种分块策略"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
    
    def chunk_text(self, text: str, strategy: str = "recursive") -> List[str]:
        """对文本进行分块"""
        if strategy == "recursive":
            return self._recursive_chunk(text)
        elif strategy == "character":
            return self._character_chunk(text)
        elif strategy == "token":
            return self._token_chunk(text)
        elif strategy == "markdown":
            return self._markdown_chunk(text)
        elif strategy == "semantic":
            return self._semantic_chunk(text)
        else:
            return self._recursive_chunk(text)  # 默认策略
    
    def chunk_documents(self, documents: List[Document], strategy: str = "recursive") -> List[Document]:
        """对文档列表进行分块"""
        chunked_docs = []
        
        for doc in documents:
            chunks = self.chunk_text(doc.page_content, strategy)
            
            for i, chunk in enumerate(chunks):
                # 创建新的文档，保留原始元数据并添加分块信息
                chunk_metadata = doc.metadata.copy()
                chunk_metadata.update({
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_strategy": strategy,
                    "chunk_size": len(chunk)
                })
                
                chunked_docs.append(Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                ))
        
        return chunked_docs
    
    def _recursive_chunk(self, text: str) -> List[str]:
        """递归字符分块（推荐）"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ";", ":", "，", " ", ""]
        )
        return splitter.split_text(text)
    
    def _character_chunk(self, text: str) -> List[str]:
        """字符分块"""
        splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator="\n"
        )
        return splitter.split_text(text)
    
    def _token_chunk(self, text: str) -> List[str]:
        """基于 Token 的分块"""
        splitter = TokenTextSplitter(
            chunk_size=self.chunk_size // 4,  # Token 通常比字符少
            chunk_overlap=self.chunk_overlap // 4
        )
        return splitter.split_text(text)
    
    def _markdown_chunk(self, text: str) -> List[str]:
        """Markdown 结构化分块"""
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]
        
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )
        
        # 先按标题分块
        md_header_splits = markdown_splitter.split_text(text)
        
        # 如果分块太大，再进行递归分块
        final_chunks = []
        for doc in md_header_splits:
            if len(doc.page_content) > self.chunk_size:
                sub_chunks = self._recursive_chunk(doc.page_content)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(doc.page_content)
        
        return final_chunks
    
    def _semantic_chunk(self, text: str) -> List[str]:
        """语义分块（基于句子边界）"""
        # 按句子分割
        sentences = self._split_sentences(text)
        
        chunks = []
        current_chunk = ""
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # 如果添加这个句子会超过限制，先保存当前块
            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # 计算重叠部分
                overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)
                current_chunk = overlap_text + sentence
                current_size = len(current_chunk)
            else:
                current_chunk += sentence
                current_size += sentence_size
        
        # 添加最后一个块
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """分割句子（支持中英文）"""
        # 中文句子结束符
        chinese_endings = r'[。！？；]'
        # 英文句子结束符
        english_endings = r'[.!?]'
        
        # 组合正则表达式
        sentence_pattern = f'({chinese_endings}|{english_endings})\s*'
        
        sentences = re.split(sentence_pattern, text)
        
        # 重新组合句子和标点
        result = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                sentence = sentences[i] + sentences[i + 1]
                if sentence.strip():
                    result.append(sentence)
        
        # 处理最后一个可能没有标点的句子
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            result.append(sentences[-1])
        
        return result
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """获取重叠文本"""
        if len(text) <= overlap_size:
            return text
        
        # 尝试在句子边界处截断
        overlap_text = text[-overlap_size:]
        
        # 找到第一个句子开始的位置
        sentence_start = re.search(r'[。！？；.!?]\s*', overlap_text)
        if sentence_start:
            return overlap_text[sentence_start.end():]
        
        return overlap_text
    
    def get_optimal_chunk_size(self, text: str, target_chunks: int = 10) -> int:
        """根据文本长度推荐最优分块大小"""
        text_length = len(text)
        optimal_size = text_length // target_chunks
        
        # 限制在合理范围内
        min_size = 200
        max_size = 2000
        
        return max(min_size, min(optimal_size, max_size))
    
    def analyze_text_structure(self, text: str) -> Dict[str, Any]:
        """分析文本结构，帮助选择分块策略"""
        analysis = {
            "total_length": len(text),
            "line_count": len(text.split('\n')),
            "paragraph_count": len([p for p in text.split('\n\n') if p.strip()]),
            "has_markdown_headers": bool(re.search(r'^#+\s', text, re.MULTILINE)),
            "has_code_blocks": bool(re.search(r'```', text)),
            "has_lists": bool(re.search(r'^\s*[-*+]\s', text, re.MULTILINE)),
            "sentence_count": len(self._split_sentences(text)),
        }
        
        # 推荐分块策略
        if analysis["has_markdown_headers"]:
            analysis["recommended_strategy"] = "markdown"
        elif analysis["paragraph_count"] > 5:
            analysis["recommended_strategy"] = "recursive"
        elif analysis["sentence_count"] > 10:
            analysis["recommended_strategy"] = "semantic"
        else:
            analysis["recommended_strategy"] = "character"
        
        return analysis