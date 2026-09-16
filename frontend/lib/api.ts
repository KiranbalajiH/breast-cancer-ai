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
}

// ── Experimental Image Analysis API ──

export interface CompatibilityFeatureReport {
  name: string;
  extracted: number | null;
  z_score: number | null;
  verdict: string;
  training_range: [number, number];
  training_mean: number;
}

export interface CompatibilityReport {
  per_feature: CompatibilityFeatureReport[];
  num_compatible: number;
  num_marginal: number;
  num_incompatible: number;
  overall_verdict: string;
  prediction_allowed: boolean;
  message: string;
}

export interface ImageAnalysisResponse {
  success: boolean;
  num_nuclei: number;
  num_measured?: number;
  message: string;
  diagnostic_images: {
    original: string;
    preprocessed: string;
    binary_mask: string;
    nuclei_overlay: string;
  } | null;
  features: Record<string, number> | null;
  compatibility: CompatibilityReport | null;
  prediction?: {
    prediction: string;
    prediction_code: string;
    confidence: number;
    probabilities: Record<string, number>;
    model: { name: string; version: string };
  } | null;
  prediction_blocked?: boolean;
  block_reason?: string | null;
}

export async function extractImageFeatures(file: File): Promise<ImageAnalysisResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_URL}/api/image-analysis/extract`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.detail || "Image feature extraction failed");
  }

  return res.json();
}

export async function predictFromImage(file: File): Promise<ImageAnalysisResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_URL}/api/image-analysis/predict`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.detail || "Image prediction failed");
  }

  return res.json();
}

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


