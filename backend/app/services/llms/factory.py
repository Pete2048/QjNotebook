from typing import Optional, Dict, Any
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from ..config import AppConfig

class LLMFactory:
    """LLM 工厂类，支持多提供方"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self._llms: Dict[str, BaseLanguageModel] = {}
    
    def get_llm(self, provider: Optional[str] = None) -> BaseLanguageModel:
        """获取 LLM 实例"""
        provider = provider or self.config.llm_provider
        
        if provider in self._llms:
            return self._llms[provider]
        
        llm = self._create_llm(provider)
        self._llms[provider] = llm
        return llm
    
    def _create_llm(self, provider: str) -> BaseLanguageModel:
        """创建 LLM 实例"""
        if provider == "openai":
            return ChatOpenAI(
                base_url=self.config.openai_base_url,
                api_key=self.config.openai_api_key,
                model=self.config.openai_model,
                temperature=0.1,
                max_tokens=4000
            )
        
        elif provider == "deepseek":
            return ChatOpenAI(
                base_url=self.config.deepseek_base_url,
                api_key=self.config.deepseek_api_key,
                model=self.config.deepseek_model,
                temperature=0.1,
                max_tokens=4000
            )
        
        elif provider == "gemini":
            return ChatGoogleGenerativeAI(
                google_api_key=self.config.gemini_api_key,
                model=self.config.gemini_model,
                temperature=0.1,
                max_output_tokens=4000
            )
        
        elif provider == "doubao":
            return ChatOpenAI(
                base_url=self.config.doubao_base_url,
                api_key=self.config.doubao_api_key,
                model=self.config.doubao_model,
                temperature=0.1,
                max_tokens=4000
            )
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    def get_available_providers(self) -> list[str]:
        """获取可用的提供方列表"""
        available = []
        
        # 检查各提供方的配置
        if self.config.openai_api_key:
            available.append("openai")
        if self.config.deepseek_api_key:
            available.append("deepseek")
        if self.config.gemini_api_key:
            available.append("gemini")
        if self.config.doubao_api_key:
            available.append("doubao")
        
        return available
    
    def test_provider(self, provider: str) -> bool:
        """测试提供方是否可用"""
        try:
            llm = self.get_llm(provider)
            # 简单测试调用
            response = llm.invoke("Hello")
            return bool(response.content)
        except Exception:
            return False