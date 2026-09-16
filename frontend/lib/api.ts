const getApiUrl = () => {
  const url = process.env.NEXT_PUBLIC_API_URL || process.env.NEXT_PUBLIC_API_BASE_URL || "https://breast-cancer-ai-backend.onrender.com";
  return url.endsWith("/") ? url.slice(0, -1) : url;
};
const API_URL = getApiUrl();

export interface ModelMetadata {
  name: string;
  version: string;
}

export async function checkHealth() {
  try {
    const res = await fetch(`${API_URL}/api/health`, { cache: 'no-store' });
    if (!res.ok) throw new Error("Backend offline");
    return await res.json();
  } catch (error) {
    console.error("Health check failed:", error);
    return { status: "unhealthy", model_loaded: false };
  }
}

export async function getModelMetadata() {
  const res = await fetch(`${API_URL}/api/model/metadata`, { cache: 'no-store' });
  if (!res.ok) throw new Error("Failed to fetch model metadata");
  return res.json();
}// ── Ultrasound Image AI Prediction API ──

export interface ImageExplanation {
  available: boolean;
  type?: string;
  heatmap?: string;
  overlay?: string;
  disclaimer?: string;
}

export interface ImagePredictionResponse {
  predicted_class: string;
  prediction?: string;
  confidence: number;
  probabilities: Record<string, number>;
  status?: string;
  message?: string;
  image_quality?: string;
  quality_warnings?: string[];
  model_version?: string;
  explanation?: ImageExplanation;
}

export async function predictImage(file: File, model: string = "v5b"): Promise<ImagePredictionResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const endpoint = model && model.toLowerCase() === "v14"
    ? `${API_URL}/api/image-predict?model=v14`
    : `${API_URL}/api/image-predict`;

  const res = await fetch(endpoint, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.detail || "Image prediction failed");
  }

  return res.json();
}

export interface ImageModelStatusResponse {
  status: "ready" | "unavailable";
  model_loaded: boolean;
  classes?: string[];
}

export async function getImageModelStatus(): Promise<ImageModelStatusResponse> {
  const res = await fetch(`${API_URL}/api/image-model/status`, { cache: 'no-store' });
  if (!res.ok) {
    throw new Error("Failed to fetch image model status");
  }
  return res.json();
}


