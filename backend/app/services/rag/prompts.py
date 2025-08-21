from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from typing import Dict, Any

class RAGPrompts:
    """RAG 系统提示模板集合"""
    
    def __init__(self):
        self.templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[str, Any]:
        """初始化所有提示模板"""
        return {
            "qa": self._create_qa_template(),
            "conversational_qa": self._create_conversational_qa_template(),
            "question_rewrite": self._create_question_rewrite_template(),
            "summarization": self._create_summarization_template(),
            "extraction": self._create_extraction_template(),
        }
    
    def get_qa_prompt(self) -> ChatPromptTemplate:
        """获取基础问答提示模板"""
        return self.templates["qa"]
    
    def get_conversational_qa_prompt(self) -> ChatPromptTemplate:
        """获取对话式问答提示模板"""
        return self.templates["conversational_qa"]
    
    def get_question_rewrite_prompt(self) -> ChatPromptTemplate:
        """获取问题重写提示模板"""
        return self.templates["question_rewrite"]
    
    def get_summarization_prompt(self) -> ChatPromptTemplate:
        """获取摘要提示模板"""
        return self.templates["summarization"]
    
    def get_extraction_prompt(self) -> ChatPromptTemplate:
        """获取信息提取提示模板"""
        return self.templates["extraction"]
    
    def _create_qa_template(self) -> ChatPromptTemplate:
        """创建基础问答模板"""
        system_message = """你是一个专业的知识库助手，能够基于提供的上下文信息准确回答用户问题。

请遵循以下原则：
1. 仅基于提供的上下文信息回答问题
2. 如果上下文中没有相关信息，请明确说明
3. 回答要准确、简洁、有条理
4. 可以适当引用上下文中的具体内容
5. 使用中文回答

上下文信息：
{context}"""
        
        human_message = "问题：{question}"
        
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_message)
        ])
    
    def _create_conversational_qa_template(self) -> ChatPromptTemplate:
        """创建对话式问答模板"""
        system_message = """你是一个专业的知识库助手，能够基于提供的上下文信息和对话历史准确回答用户问题。

请遵循以下原则：
1. 仅基于提供的上下文信息回答问题
2. 考虑对话历史，保持上下文连贯性
3. 如果上下文中没有相关信息，请明确说明
4. 回答要准确、简洁、有条理
5. 可以适当引用上下文中的具体内容
6. 使用中文回答

对话历史：
{chat_history}

上下文信息：
{context}"""
        
        human_message = "问题：{question}"
        
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_message)
        ])
    
    def _create_question_rewrite_template(self) -> ChatPromptTemplate:
        """创建问题重写模板"""
        system_message = """你是一个问题重写专家。基于对话历史，将用户的当前问题重写为一个独立、完整的问题。

重写原则：
1. 保持问题的核心意图不变
2. 补充必要的上下文信息
3. 确保重写后的问题可以独立理解
4. 使用中文
5. 如果当前问题已经很完整，可以保持不变

对话历史：
{chat_history}"""
        
        human_message = "当前问题：{question}\n\n请重写这个问题："
        
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_message)
        ])
    
    def _create_summarization_template(self) -> ChatPromptTemplate:
        """创建摘要模板"""
        system_message = """你是一个专业的文档摘要专家。请为提供的文档内容生成简洁、准确的摘要。

摘要要求：
1. 提取文档的核心信息和要点
2. 保持逻辑清晰，结构合理
3. 长度适中，通常在200-500字之间
4. 使用中文
5. 突出重要信息和关键概念"""
        
        human_message = "请为以下文档生成摘要：\n\n{content}"
        
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_message)
        ])
    
    def _create_extraction_template(self) -> ChatPromptTemplate:
        """创建信息提取模板"""
        system_message = """你是一个信息提取专家。请从提供的文档中提取指定类型的信息。

提取要求：
1. 准确识别和提取目标信息
2. 保持信息的完整性和准确性
3. 按照指定格式输出
4. 如果没有找到相关信息，请明确说明
5. 使用中文"""
        
        human_message = """文档内容：
{content}

提取要求：{extraction_instruction}

请提取相关信息："""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_message)
        ])