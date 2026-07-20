export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

export type Dataset = { id: string; name: string; source_file: string; row_count: number; uploaded_at: string }
export type Flow = { id: string; raw_features_json: Record<string, unknown>; predicted_label: number; predicted_attack_cat?: string | null; confidence: number; true_label?: number | null }
export type ModelMetrics = { accuracy: number; classes: string[]; confusion_matrix: number[][]; classification_report: Record<string, { precision: number; recall: number; 'f1-score': number; support: number }>; feature_importance: [string, number][] }

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, init)
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    throw new Error(body.detail || `Request failed (${response.status})`)
  }
  return response.json() as Promise<T>
}

export const api = {
  datasets: () => request<Dataset[]>('/api/datasets'),
  dataset: (id: string) => request<Dataset>(`/api/datasets/${id}`),
  summary: (id: string) => request<any>(`/api/datasets/${id}/summary`),
  flows: (id: string, page = 1, options: Record<string, string | number> = {}) => request<any>(`/api/datasets/${id}/flows?${new URLSearchParams({ page: String(page), limit: '50', ...Object.fromEntries(Object.entries(options).map(([k, v]) => [k, String(v)])) })}`),
  flow: (id: string) => request<Flow>(`/api/flows/${id}`),
  explanation: (id: string) => request<any[]>(`/api/flows/${id}/explanation`),
  activeModel: (lane: string) => request<{ version_label: string; dataset_source: string; metrics: ModelMetrics }>(`/api/models/${lane}/active`),
  matrix: (lane: string) => request<{ classes: string[]; matrix: number[][] }>(`/api/models/${lane}/confusion-matrix`),
  upload: (file: File) => { const form = new FormData(); form.append('file', file); return request<{ job_id: string; status: string; max_analysis_rows: number }>('/api/datasets/upload', { method: 'POST', body: form }) },
  job: (id: string) => request<any>(`/api/jobs/${id}`),
  trainCicids: () => request<any>('/api/models/cicids2017/train', { method: 'POST' }),
}
