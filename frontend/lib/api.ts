const API_URL = process.env.NEXT_PUBLIC_API_URL || process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";

export interface BreastCancerFeatures {
  "mean radius": number;
  "mean texture": number;
  "mean perimeter": number;
  "mean area": number;
  "mean smoothness": number;
  "mean compactness": number;
  "mean concavity": number;
  "mean concave points": number;
  "mean symmetry": number;
  "mean fractal dimension": number;
  "radius error": number;
  "texture error": number;
  "perimeter error": number;
  "area error": number;
  "smoothness error": number;
  "compactness error": number;
  "concavity error": number;
  "concave points error": number;
  "symmetry error": number;
  "fractal dimension error": number;
  "worst radius": number;
  "worst texture": number;
  "worst perimeter": number;
  "worst area": number;
  "worst smoothness": number;
  "worst compactness": number;
  "worst concavity": number;
  "worst concave points": number;
  "worst symmetry": number;
  "worst fractal dimension": number;
}

export interface ModelMetadata {
  name: string;
  version: string;
}

export interface PredictionResponse {
  prediction: string;
  prediction_code: string;
  confidence: number;
  probabilities: Record<string, number>;
  model: ModelMetadata;
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

export async function predictCancer(features: BreastCancerFeatures): Promise<PredictionResponse> {
  const res = await fetch(`${API_URL}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(features),
  });
  
  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.detail || "Prediction failed");
  }
  
  return res.json();
}

export async function getModelMetadata() {
  const res = await fetch(`${API_URL}/api/model/metadata`, { cache: 'no-store' });
  if (!res.ok) throw new Error("Failed to fetch model metadata");
  return res.json();
}

export async function getModelComparison() {
  const res = await fetch(`${API_URL}/api/model/comparison`, { cache: 'no-store' });
  if (!res.ok) throw new Error("Failed to fetch model comparison");
  return res.json();
}

export async function getModelFeatures() {
  const res = await fetch(`${API_URL}/api/model/features`, { cache: 'no-store' });
  if (!res.ok) throw new Error("Failed to fetch features");
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

export interface ImagePredictionResponse {
  predicted_class: string;
  confidence: number;
  probabilities: Record<string, number>;
}

export async function predictImage(file: File): Promise<ImagePredictionResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_URL}/api/image-predict`, {
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


