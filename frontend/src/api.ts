const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export interface Notebook {
  id: string;
  name: string;
  created_at: number;
  updated_at?: number;
  document_count?: number;
}

export interface QueryResult {
  question: string;
  answer: string;
  sources: Array<{
    content: string;
    metadata: Record<string, any>;
    score: number;
  }>;
  meta: Record<string, any>;
}

export interface HealthStatus {
  status: string;
  vector_store: string;
  kg_enabled: boolean;
  timestamp: number;
}

export interface Settings {
  providers: string[];
  active_provider: string;
}

class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  
  const config: RequestInit = {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  };

  try {
    const response = await fetch(url, config);
    
    if (!response.ok) {
      let errorMessage = `HTTP ${response.status}`;
      try {
        const errorData = await response.json();
        errorMessage = errorData.detail || errorData.message || errorMessage;
      } catch {
        errorMessage = response.statusText || errorMessage;
      }
      throw new ApiError(response.status, errorMessage);
    }

    // Handle empty responses
    const contentType = response.headers.get('content-type');
    if (contentType && contentType.includes('application/json')) {
      return await response.json();
    } else {
      return {} as T;
    }
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    
    // Network or other errors
    if (error instanceof TypeError && error.message.includes('fetch')) {
      throw new ApiError(0, '无法连接到服务器，请检查后端是否正常运行');
    }
    
    throw new ApiError(0, error instanceof Error ? error.message : '未知错误');
  }
}

// Health check
export async function getHealth(): Promise<HealthStatus> {
  return request<HealthStatus>('/health');
}

// Settings
export async function getSettings(): Promise<Settings> {
  return request<Settings>('/settings');
}

export async function setProvider(provider: string): Promise<void> {
  return request<void>('/settings/provider', {
    method: 'POST',
    body: JSON.stringify({ provider }),
  });
}

// Notebook management
export async function listNotebooks(): Promise<Notebook[]> {
  return request<Notebook[]>('/notebooks');
}

export async function createNotebook(name: string): Promise<Notebook> {
  return request<Notebook>('/notebooks', {
    method: 'POST',
    body: JSON.stringify({ name }),
  });
}

export async function deleteNotebook(notebookId: string): Promise<void> {
  return request<void>(`/notebooks/${notebookId}`, {
    method: 'DELETE',
  });
}

export async function getNotebook(notebookId: string): Promise<Notebook> {
  return request<Notebook>(`/notebooks/${notebookId}`);
}

// Document management
export async function uploadText(
  notebookId: string,
  texts: string[],
  metadata: Record<string, any> = {}
): Promise<void> {
  return request<void>(`/notebooks/${notebookId}/documents`, {
    method: 'POST',
    body: JSON.stringify({ texts, metadata }),
  });
}

export async function uploadPath(
  notebookId: string,
  path: string,
  metadata: Record<string, any> = {}
): Promise<void> {
  return request<void>(`/notebooks/${notebookId}/documents/path`, {
    method: 'POST',
    body: JSON.stringify({ path, metadata }),
  });
}

// Query
export async function queryNotebook(
  notebookId: string,
  question: string,
  topK: number = 5
): Promise<QueryResult> {
  return request<QueryResult>(`/notebooks/${notebookId}/query`, {
    method: 'POST',
    body: JSON.stringify({ question, top_k: topK }),
  });
}

// File upload helper
export async function uploadFile(
  notebookId: string,
  file: File,
  onProgress?: (progress: number) => void
): Promise<void> {
  return new Promise((resolve, reject) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('metadata', JSON.stringify({ source: file.name }));

    const xhr = new XMLHttpRequest();
    
    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable && onProgress) {
        const progress = (e.loaded / e.total) * 100;
        onProgress(progress);
      }
    });

    xhr.addEventListener('load', () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve();
      } else {
        let errorMessage = `HTTP ${xhr.status}`;
        try {
          const errorData = JSON.parse(xhr.responseText);
          errorMessage = errorData.detail || errorData.message || errorMessage;
        } catch {
          errorMessage = xhr.statusText || errorMessage;
        }
        reject(new ApiError(xhr.status, errorMessage));
      }
    });

    xhr.addEventListener('error', () => {
      reject(new ApiError(0, '文件上传失败'));
    });

    xhr.open('POST', `${API_BASE_URL}/notebooks/${notebookId}/documents/upload`);
    xhr.send(formData);
  });
}