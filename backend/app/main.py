"""
NotebookLM-like RAG API
基于 LangChain 实现的企业级知识库 RAG 系统
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import json
import os
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import time
from dotenv import load_dotenv, find_dotenv
import pickle

from app.config import load_config
from app.services.rag.langchain_pipeline import LangChainRAGPipeline

# 创建 FastAPI 应用
app = FastAPI(title="企业级知识库 RAG API", version="1.0.0")

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建静态目录（如果不存在）
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)

# 挂载静态文件
try:
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
except Exception as e:
    print(f"警告: 无法挂载静态文件: {e}")

# 确保所有响应使用 UTF-8 编码
@app.middleware("http")
async def ensure_utf8_encoding(request, call_next):
    response = await call_next(request)
    if response.headers.get("content-type", "").startswith("application/json"):
        response.headers["content-type"] = "application/json; charset=utf-8"
    return response

# 加载配置
# 优先加载项目根/父目录的 .env-local，其次加载 .env
load_dotenv(find_dotenv(filename=".env-local"))
load_dotenv(find_dotenv())
cfg = load_config()
pipeline: Optional[LangChainRAGPipeline] = None

# 笔记本持久化存储路径
NOTEBOOKS_STORAGE_PATH = os.path.join(os.path.dirname(__file__), "storage", "notebooks.pkl")

def save_notebooks():
    """保存笔记本到磁盘"""
    try:
        if pipeline and hasattr(pipeline, 'notebooks') and pipeline.notebooks:
            os.makedirs(os.path.dirname(NOTEBOOKS_STORAGE_PATH), exist_ok=True)
            with open(NOTEBOOKS_STORAGE_PATH, 'wb') as f:
                pickle.dump(pipeline.notebooks, f)
            print(f"已保存 {len(pipeline.notebooks)} 个笔记本")
    except Exception as e:
        print(f"保存笔记本失败: {e}")

def load_notebooks():
    """从磁盘加载笔记本"""
    try:
        if os.path.exists(NOTEBOOKS_STORAGE_PATH):
            with open(NOTEBOOKS_STORAGE_PATH, 'rb') as f:
                notebooks = pickle.load(f)
                print(f"成功加载 {len(notebooks)} 个笔记本")
                return notebooks
    except Exception as e:
        print(f"加载笔记本失败: {e}")
    return {}


class CreateNotebookRequest(BaseModel):
    name: str


class UploadRequest(BaseModel):
    file_path: str
    metadata: Optional[Dict[str, Any]] = None


class UploadTextRequest(BaseModel):
    texts: List[str]
    metadata: Optional[Dict[str, Any]] = None


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5


class ProviderRequest(BaseModel):
    # 运行时切换的模型提供方，取值如：openai、deepseek、gemini、doubao
    provider: str


@app.on_event("startup")
async def startup():
    global pipeline
    try:
        pipeline = LangChainRAGPipeline(cfg)
        app.state.started_at = time.time()
        
        # 加载已保存的笔记本
        try:
            saved_notebooks = load_notebooks()
            if saved_notebooks:
                pipeline.notebooks = saved_notebooks
                print(f"已加载 {len(saved_notebooks)} 个笔记本")
        except Exception as load_error:
            print(f"加载笔记本失败: {load_error}")
            
        print("LangChain RAG 流水线初始化成功")
    except Exception as e:
        print(f"初始化 LangChain RAG 流水线时出错: {e}")
        # 仍然设置启动时间，以便健康检查端点正常工作
        app.state.started_at = time.time()


@app.get("/")
async def root():
    return {"message": "欢迎使用企业级知识库 RAG API。访问 /docs 获取 API 文档。"}


@app.get("/health")
async def health():
    vector_store = getattr(cfg, "vector_store", "unknown")
    kg_enabled = getattr(cfg, "enable_kg", False)
    
    return {
        "status": "ok",
        "uptime_sec": round(time.time() - getattr(app.state, "started_at", time.time()), 2),
        "vector_store": vector_store,
        "kg_enabled": kg_enabled,
        "pipeline_ready": pipeline is not None
    }


@app.get("/api/settings")
async def get_settings():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="流水线未就绪")
    try:
        return pipeline.get_settings()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/settings/provider")
async def set_provider(req: ProviderRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="流水线未就绪")
    try:
        return pipeline.set_provider(req.provider)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/notebooks")
async def create_notebook(req: CreateNotebookRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="流水线未就绪")
    try:
        nb = pipeline.create_notebook(req.name)
        # 保存笔记本到磁盘
        save_notebooks()
        # 确保响应使用 UTF-8 编码
        return JSONResponse(
            content=nb,
            headers={"Content-Type": "application/json; charset=utf-8"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/notebooks")
async def list_notebooks():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="流水线未就绪")
    notebooks = list(pipeline.notebooks.values())
    return JSONResponse(
        content=notebooks,
        headers={"Content-Type": "application/json; charset=utf-8"}
    )


@app.delete("/api/notebooks/{notebook_id}")
async def delete_notebook(notebook_id: str):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="流水线未就绪")
    if notebook_id not in pipeline.notebooks:
        raise HTTPException(status_code=404, detail="笔记本不存在")
    try:
        pipeline.delete_notebook(notebook_id)
        # 保存笔记本到磁盘
        save_notebooks()
        return {"message": "笔记本删除成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/notebooks/{notebook_id}/upload")
async def upload_to_notebook(notebook_id: str, req: UploadRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="流水线未就绪")
    if notebook_id not in pipeline.notebooks:
        raise HTTPException(status_code=404, detail="笔记本不存在")
    try:
        stats = pipeline.ingest_paths(notebook_id, [req.file_path], req.metadata or {})
        return {"message": "ok", "ingested": stats}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/notebooks/{notebook_id}/uploadText")
async def upload_text_to_notebook(notebook_id: str, req: UploadTextRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="流水线未就绪")
    if notebook_id not in pipeline.notebooks:
        raise HTTPException(status_code=404, detail="笔记本不存在")
    try:
        print(f"[MAIN] 收到 uploadText 请求，笔记本 ID: {notebook_id}")
        print(f"[MAIN] 请求元数据: {req.metadata}")
        print(f"[MAIN] 文本数量: {len(req.texts)}")
        print(f"[MAIN] 第一个文本长度: {len(req.texts[0]) if req.texts else 0}")
        
        stats = pipeline.ingest_texts(notebook_id, req.texts, req.metadata or {})
        
        print(f"[MAIN] 摄入完成: {stats}")
        return {"message": "ok", "ingested": stats}
    except Exception as e:
        print(f"[MAIN] uploadText 出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/notebooks/{notebook_id}/query")
async def query_notebook(notebook_id: str, req: QueryRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="流水线未就绪")
    if notebook_id not in pipeline.notebooks:
        raise HTTPException(status_code=404, detail="笔记本不存在")
    try:
        result = pipeline.query(notebook_id, req.question, top_k=req.top_k or 5)
        # 使源文档可序列化，并确保 UTF-8 编码
        serializable_sources: List[Dict[str, Any]] = []
        for s in result.get("sources", []):
            content = s.get("content", "")
            # 确保内容正确编码为 UTF-8 字符串
            if isinstance(content, bytes):
                content = content.decode('utf-8', errors='ignore')
            elif not isinstance(content, str):
                content = str(content)
            
            serializable_sources.append(
                {
                    "content": content,
                    "metadata": s.get("metadata", {}),
                    "score": s.get("score", 0.0),
                }
            )
        
        # 确保答案正确编码
        answer = result.get("answer", "")
        if isinstance(answer, bytes):
            answer = answer.decode('utf-8', errors='ignore')
        elif not isinstance(answer, str):
            answer = str(answer)
            
        response_data = {
            "question": result.get("question"),
            "answer": answer,
            "sources": serializable_sources,
            "meta": result.get("meta", {}),
        }
        
        # 返回时明确指定 UTF-8 编码
        return JSONResponse(
            content=response_data,
            headers={"Content-Type": "application/json; charset=utf-8"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))