import React, { useEffect, useMemo, useState } from "react";
import {
  getHealth,
  listNotebooks,
  createNotebook,
  deleteNotebook,
  uploadText,
  uploadPath,
  queryNotebook,
  getSettings,
  setProvider,
} from "./api";
import "./styles.css";

type Notebook = {
  id: string;
  name: string;
  created_at: number;
};

type Source = {
  content: string;
  metadata: Record<string, any>;
  score: number;
};

type QueryResult = {
  question: string;
  answer: string;
  sources: Source[];
  meta: Record<string, any>;
};

type ChatMessage = {
  id: string;
  question: string;
  answer: string;
  sources: Source[];
  timestamp: number;
};

export default function App() {
  const [health, setHealth] = useState<string>("检查中...");
  const [notebooks, setNotebooks] = useState<Notebook[]>([]);
  const [selected, setSelected] = useState<string>("");
  const [newNotebookName, setNewNotebookName] = useState<string>("");
  const [ingestText, setIngestText] = useState<string>("");
  const [uploadedFiles, setUploadedFiles] = useState<any[]>([]);
  const [isDragOver, setIsDragOver] = useState<boolean>(false);
  const [question, setQuestion] = useState<string>("");
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [uploading, setUploading] = useState<boolean>(false);
  const [uploadProgress, setUploadProgress] = useState<string>("");

  // 模型列表与当前模型
  const [providers, setProviders] = useState<string[]>([]);
  const [activeProvider, setActiveProvider] = useState<string>("");

  useEffect(() => {
    (async () => {
      try {
        const h = await getHealth();
        setHealth(`正常 (向量库=${h.vector_store}, KG=${h.kg_enabled})`);
      } catch (e: any) {
        setHealth(`后端异常: ${e?.message || "无法连接后端"}`);
      }
      try {
        const s = await getSettings();
        setProviders(s.providers || []);
        setActiveProvider(s.active_provider || "");
      } catch {}
      const nbs = await listNotebooks();
      setNotebooks(nbs);
      if (nbs.length > 0) setSelected(nbs[0].id);
    })();
  }, []);

  const selectedNotebook = useMemo(
    () => notebooks.find((n) => n.id === selected),
    [selected, notebooks]
  );

  async function handleCreateNotebook() {
    if (!newNotebookName.trim()) return;
    try {
      const nb = await createNotebook(newNotebookName.trim());
      setNotebooks((prev) => [nb, ...prev]);
      setSelected(nb.id);
      setNewNotebookName("");
    } catch (e: any) {
      alert(`创建失败: ${e?.message || "未知错误"}`);
    }
  }

  async function handleIngestText() {
    if (!selected || !ingestText.trim()) return;
    await uploadText(selected, [ingestText.trim()], {});
    setIngestText("");
    alert("已导入文本。");
  }

  async function handleFileUpload(files: FileList) {
    if (!selected || !files.length) return;
    
    setUploading(true);
    setUploadProgress(`准备上传 ${files.length} 个文件...`);
    
    let successCount = 0;
    
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      setUploadProgress(`正在处理文件 ${i + 1}/${files.length}: ${file.name}`);
      
      try {
        let text = "";
        
        // Handle different file types
        if (file.type === "application/pdf" || file.name.toLowerCase().endsWith('.pdf')) {
          // Check file size limit (10MB max)
          if (file.size > 10 * 1024 * 1024) {
            throw new Error("PDF文件过大，请选择小于10MB的文件");
          }
          
          setUploadProgress(`正在读取PDF文件: ${file.name}...`);
          const arrayBuffer = await file.arrayBuffer();
          const uint8Array = new Uint8Array(arrayBuffer);
          
          setUploadProgress(`正在转换文件格式: ${file.name}...`);
          // Convert to base64 in chunks to avoid "Maximum call stack size exceeded"
          // Convert to base64 in chunks to avoid "Maximum call stack size exceeded"
          let base64 = '';
          const chunkSize = 8192; // Process 8KB at a time
          
          // Use a more reliable base64 encoding approach
          try {
          // First try the modern approach if available
            if (typeof window !== 'undefined' && window.btoa) {
              // Browser environment - chunk processing to avoid stack overflow
              for (let j = 0; j < uint8Array.length; j += chunkSize) {
                const chunk = uint8Array.slice(j, j + chunkSize);
                const chunkString = String.fromCharCode.apply(null, Array.from(chunk));
                base64 += window.btoa(chunkString);
                
                // Update progress for large files
                if (j % (chunkSize * 10) === 0) {
                  const progress = Math.round((j / uint8Array.length) * 100);
                  setUploadProgress(`正在转换文件格式: ${file.name} (${progress}%)`);
                }
              }
            } else {
              throw new Error("No base64 encoding method available");
            }
            
            // Validate the base64 string
            if (!base64 || base64.length < 100) {
              throw new Error("Base64 conversion failed - result too short");
            }
            
            // Log the first few characters to verify it's valid base64
            console.log(`Base64 prefix (first 20 chars): ${base64.substring(0, 20)}`);
            
          } catch (encodeError) {
            console.error("Base64 encoding error:", encodeError);
            throw new Error(`PDF编码失败: ${encodeError.message}`);
          }
          
          setUploadProgress(`正在上传到服务器: ${file.name}...`);
          // Send to backend with special PDF handling flag
          await uploadText(selected, [base64], { 
            source: file.name,
            type: file.type,
            size: file.size,
            encoding: "base64",
            isPDF: true
          });
        } else {
          setUploadProgress(`正在读取文本文件: ${file.name}...`);
          // For text files, read with proper encoding
          text = await file.text();
          
          // Validate text is readable
          if (text && text.trim()) {
            setUploadProgress(`正在上传到服务器: ${file.name}...`);
            await uploadText(selected, [text], { 
              source: file.name,
              type: file.type,
              size: file.size,
              encoding: "utf-8"
            });
          } else {
            throw new Error("文件内容为空或无法读取");
          }
        }
        
        // Add to uploaded files list
        setUploadedFiles(prev => [...prev, {
          id: Date.now() + i,
          name: file.name,
          type: file.type,
          size: file.size,
          uploadedAt: new Date().toISOString()
        }]);
        
        successCount++;
        setUploadProgress(`已完成 ${successCount}/${files.length} 个文件`);
        
      } catch (e: any) {
        console.error(`Upload failed for ${file.name}:`, e);
        setUploadProgress(`文件 ${file.name} 上传失败: ${e?.message || "未知错误"}`);
        // Continue with next file instead of stopping
      }
    }
    
    setUploading(false);
    setUploadProgress("");
    
    // 移除弹窗提示，改为静默完成
    console.log(`Upload completed: ${successCount}/${files.length} files successful`);
  }

  function handleDragOver(e: React.DragEvent) {
    e.preventDefault();
    setIsDragOver(true);
  }

  function handleDragLeave(e: React.DragEvent) {
    e.preventDefault();
    setIsDragOver(false);
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setIsDragOver(false);
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileUpload(files);
    }
  }

  function handleFileInputChange(e: React.ChangeEvent<HTMLInputElement>) {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFileUpload(files);
    }
  }

  async function handleAsk() {
    if (!selected || !question.trim()) return;
    setLoading(true);
    const currentQuestion = question.trim();
    setQuestion(""); // Clear input immediately
    
    try {
      const res: QueryResult = await queryNotebook(selected, currentQuestion, 5);
      const answer = !res.answer || res.answer.trim() === "" 
        ? "暂无相关内容可以回答您的问题。请先上传一些文档或文本到知识库中。"
        : res.answer;
      
      // Add to chat history
      const newMessage: ChatMessage = {
        id: Date.now().toString(),
        question: currentQuestion,
        answer: answer,
        sources: res.sources || [],
        timestamp: Date.now()
      };
      
      setChatHistory(prev => [newMessage, ...prev]);
    } catch (e: any) {
      const errorMessage = `查询失败: ${e?.message || "未知错误"}`;
      const newMessage: ChatMessage = {
        id: Date.now().toString(),
        question: currentQuestion,
        answer: errorMessage,
        sources: [],
        timestamp: Date.now()
      };
      setChatHistory(prev => [newMessage, ...prev]);
    } finally {
      setLoading(false);
    }
  }

  async function handleProviderChange(p: string) {
    try {
      await setProvider(p);
      setActiveProvider(p);
    } catch (e: any) {
      alert(`切换模型失败: ${e?.message || "未知错误"}`);
    }
  }

  function clearChatHistory() {
    if (confirm("确定要清空聊天记录吗？")) {
      setChatHistory([]);
    }
  }

  async function handleDeleteNotebook(notebookId: string) {
    if (!confirm("确定要删除这个笔记本吗？")) return;
    try {
      await deleteNotebook(notebookId);
      const updatedNotebooks = notebooks.filter((nb) => nb.id !== notebookId);
      setNotebooks(updatedNotebooks);
      if (selected === notebookId) {
        setSelected(updatedNotebooks.length > 0 ? updatedNotebooks[0].id : "");
      }
      alert("笔记本已删除。");
    } catch (e: any) {
      alert(`删除失败: ${e?.message || "未知错误"}`);
    }
  }

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand">NotebookLM 知识库</div>
        <div className="health">后端状态：{health}</div>

        <div className="section-title">模型切换</div>
        <div className="provider-row">
          <select
            value={activeProvider || ""}
            onChange={(e) => handleProviderChange(e.target.value)}
          >
            <option value="" disabled>选择模型提供方</option>
            {providers.map((p) => (
              <option key={p} value={p}>
                {p === "doubao" ? "豆包 (Volcengine Ark)" :
                 p === "openai" ? "OpenAI" :
                 p === "deepseek" ? "DeepSeek" :
                 p === "gemini" ? "Gemini" : p}
              </option>
            ))}
          </select>
        </div>

        <div className="section-title">笔记本</div>
        <div className="new-notebook">
          <input
            placeholder="新笔记本名称"
            value={newNotebookName}
            onChange={(e) => setNewNotebookName(e.target.value)}
          />
          <button onClick={handleCreateNotebook}>创建</button>
        </div>
        <ul className="notebook-list">
          {notebooks.map((nb) => (
            <li
              key={nb.id}
              className={selected === nb.id ? "active" : ""}
              title={new Date(nb.created_at * 1000).toLocaleString()}
              style={{display: 'flex', alignItems: 'center'}}
            >
              <span onClick={() => setSelected(nb.id)} style={{flex: 1, cursor: 'pointer'}}>
                {nb.name}
              </span>
              <button 
                onClick={(e) => {
                  e.stopPropagation();
                  handleDeleteNotebook(nb.id);
                }}
                style={{
                  marginLeft: '8px',
                  padding: '2px 6px',
                  fontSize: '12px',
                  background: '#ff4444',
                  color: 'white',
                  border: 'none',
                  borderRadius: '3px',
                  cursor: 'pointer'
                }}
              >
                删除
              </button>
            </li>
          ))}
        </ul>
      </aside>

      <main className="main">
        <section className="sources">
          <div className="section-title">数据源</div>
          {!selectedNotebook ? (
            <div className="empty">请先创建或选择左侧笔记本，然后导入数据。</div>
          ) : (
            <>
              {/* File Upload Area */}
              <div 
                className={`file-upload-area ${isDragOver ? 'drag-over' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => document.getElementById('file-input')?.click()}
              >
                <div className="upload-icon">📁</div>
                <div className="upload-text">
                  <div>点击上传文件或拖拽到此处</div>
                  <div className="upload-hint">支持 PDF, TXT, MD, DOC 等格式</div>
                </div>
                <input
                  id="file-input"
                  type="file"
                  multiple
                  accept=".pdf,.txt,.md,.doc,.docx,.csv"
                  onChange={handleFileInputChange}
                  style={{ display: 'none' }}
                />
              </div>

              {/* Text Input Area */}
              <div className="text-input-area">
                <textarea
                  placeholder="或在此粘贴文本内容..."
                  value={ingestText}
                  onChange={(e) => setIngestText(e.target.value)}
                  rows={4}
                />
                <button onClick={handleIngestText} disabled={!ingestText.trim()}>
                  导入文本
                </button>
              </div>

              {/* Uploaded Files List */}
              {uploadedFiles.length > 0 && (
                <div className="uploaded-files">
                  <div className="files-title">已上传文档 ({uploadedFiles.length})</div>
                  <div className="files-list">
                    {uploadedFiles.map((file) => (
                      <div key={file.id} className="file-item">
                        <div className="file-icon">📄</div>
                        <div className="file-info">
                          <div className="file-name">{file.name}</div>
                          <div className="file-meta">
                            {(file.size / 1024).toFixed(1)} KB • {new Date(file.uploadedAt).toLocaleString()}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </section>

        <section className="chat">
          <div className="chat-header">
            <div className="section-title">聊天</div>
            {chatHistory.length > 0 && (
              <button 
                onClick={clearChatHistory}
                className="clear-history-btn"
                style={{
                  padding: '4px 8px',
                  fontSize: '12px',
                  background: 'var(--danger)',
                  marginLeft: 'auto'
                }}
              >
                清空记录
              </button>
            )}
          </div>
          <div className="chat-input">
            <input
              placeholder="请输入问题..."
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") handleAsk();
              }}
            />
            <button onClick={handleAsk} disabled={!question.trim() || !selected || loading}>
              {loading ? "思考中..." : "提问"}
            </button>
          </div>
          
          <div className="chat-history">
            {chatHistory.length === 0 ? (
              <div className="empty">暂无对话记录。</div>
            ) : (
              chatHistory.map((message) => (
                <div key={message.id} className="chat-message">
                  <div className="question-block">
                    <div className="question-label">问题:</div>
                    <div className="question-text">{message.question}</div>
                    <div className="timestamp">{new Date(message.timestamp).toLocaleString()}</div>
                  </div>
                  <div className="answer-block">
                    <div className="answer-label">回答:</div>
                    <div className="answer-text">{message.answer}</div>
                    {message.sources.length > 0 && (
                      <div className="sources-section">
                        <div className="sources-title">引用来源:</div>
                        <div className="sources-list">
                          {message.sources.map((s, i) => (
                            <div className="citation" key={i}>
                              <div className="meta">
                                <span className="badge">[{i + 1}]</span>
                                <span className="src">{s.metadata?.source || "unknown"}</span>
                                <span className="score">score={s.score?.toFixed(3) ?? "0.000"}</span>
                              </div>
                              <div className="snippet">
                                {s.content.length > 200 ? s.content.slice(0, 200) + " ..." : s.content}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}
          </div>
        </section>
      </main>
    </div>
  );
}