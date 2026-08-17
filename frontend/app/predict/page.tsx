/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";

import React, { useState, useCallback, useMemo } from "react";
import { predictCancer, BreastCancerFeatures, PredictionResponse, predictImage, ImagePredictionResponse } from "@/lib/api";
import { AlertCircle, PlayCircle, RefreshCw, Dna, Info, HelpCircle, Upload, ImageIcon, Loader2, Activity } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

const CORE_DESCRIPTIONS: Record<string, string> = {
  "Radius": "Distance from center to points on the perimeter.",
  "Texture": "Standard deviation of gray-scale values.",
  "Perimeter": "Size of the cell nucleus perimeter.",
  "Area": "Size of the cell nucleus area.",
  "Smoothness": "Local variation in radius lengths.",
  "Compactness": "Computed as Perimeter^2 / Area - 1.0.",
  "Concavity": "Severity of concave portions of the contour.",
  "Concave Points": "Number of concave portions of the contour.",
  "Symmetry": "Symmetry alignment of the nucleus structure.",
  "Fractal Dimension": "Irregularity of the nucleus boundary (coastline approximation)."
};

const DATASET_STATS: Record<string, { min: number; max: number }> = {
  "mean radius": { "min": 6.981, "max": 28.11 },
  "mean texture": { "min": 9.71, "max": 39.28 },
  "mean perimeter": { "min": 43.79, "max": 188.5 },
  "mean area": { "min": 143.5, "max": 2501.0 },
  "mean smoothness": { "min": 0.05263, "max": 0.1634 },
  "mean compactness": { "min": 0.01938, "max": 0.3454 },
  "mean concavity": { "min": 0.0, "max": 0.4268 },
  "mean concave points": { "min": 0.0, "max": 0.2012 },
  "mean symmetry": { "min": 0.106, "max": 0.304 },
  "mean fractal dimension": { "min": 0.04996, "max": 0.09744 },
  "radius error": { "min": 0.1115, "max": 2.873 },
  "texture error": { "min": 0.3602, "max": 4.885 },
  "perimeter error": { "min": 0.757, "max": 21.98 },
  "area error": { "min": 6.802, "max": 542.2 },
  "smoothness error": { "min": 0.001713, "max": 0.03113 },
  "compactness error": { "min": 0.002252, "max": 0.1354 },
  "concavity error": { "min": 0.0, "max": 0.396 },
  "concave points error": { "min": 0.0, "max": 0.05279 },
  "symmetry error": { "min": 0.007882, "max": 0.07895 },
  "fractal dimension error": { "min": 0.0008948, "max": 0.02984 },
  "worst radius": { "min": 7.93, "max": 36.04 },
  "worst texture": { "min": 12.02, "max": 49.54 },
  "worst perimeter": { "min": 50.41, "max": 251.2 },
  "worst area": { "min": 185.2, "max": 4254.0 },
  "worst smoothness": { "min": 0.07117, "max": 0.2226 },
  "worst compactness": { "min": 0.02729, "max": 1.058 },
  "worst concavity": { "min": 0.0, "max": 1.252 },
  "worst concave points": { "min": 0.0, "max": 0.291 },
  "worst symmetry": { "min": 0.1565, "max": 0.6638 },
  "worst fractal dimension": { "min": 0.05504, "max": 0.2075 }
};

const DEFAULT_FEATURES: Record<string, string> = {
  "mean radius": "", "mean texture": "", "mean perimeter": "", "mean area": "", "mean smoothness": "",
  "mean compactness": "", "mean concavity": "", "mean concave points": "", "mean symmetry": "", "mean fractal dimension": "",
  "radius error": "", "texture error": "", "perimeter error": "", "area error": "", "smoothness error": "",
  "compactness error": "", "concavity error": "", "concave points error": "", "symmetry error": "", "fractal dimension error": "",
  "worst radius": "", "worst texture": "", "worst perimeter": "", "worst area": "", "worst smoothness": "",
  "worst compactness": "", "worst concavity": "", "worst concave points": "", "worst symmetry": "", "worst fractal dimension": ""
};

const SAMPLE_BENIGN: Record<string, number> = {
  "mean radius": 13.54, "mean texture": 14.36, "mean perimeter": 87.46, "mean area": 566.3, 
  "mean smoothness": 0.09779, "mean compactness": 0.08129, "mean concavity": 0.06664, 
  "mean concave points": 0.04781, "mean symmetry": 0.1885, "mean fractal dimension": 0.05766, 
  "radius error": 0.2699, "texture error": 0.7886, "perimeter error": 2.058, "area error": 23.56, 
  "smoothness error": 0.008462, "compactness error": 0.0146, "concavity error": 0.02387, 
  "concave points error": 0.01315, "symmetry error": 0.0198, "fractal dimension error": 0.0023, 
  "worst radius": 15.11, "worst texture": 19.26, "worst perimeter": 99.7, "worst area": 711.2, 
  "worst smoothness": 0.144, "worst compactness": 0.1773, "worst concavity": 0.239, 
  "worst concave points": 0.1288, "worst symmetry": 0.2977, "worst fractal dimension": 0.07259
};

const SAMPLE_MALIGNANT: Record<string, number> = {
  "mean radius": 20.57, "mean texture": 17.77, "mean perimeter": 132.9, "mean area": 1326.0, 
  "mean smoothness": 0.08474, "mean compactness": 0.07864, "mean concavity": 0.0869, 
  "mean concave points": 0.07017, "mean symmetry": 0.1812, "mean fractal dimension": 0.05667, 
  "radius error": 0.5435, "texture error": 0.7339, "perimeter error": 3.398, "area error": 74.08, 
  "smoothness error": 0.005225, "compactness error": 0.01308, "concavity error": 0.0186, 
  "concave points error": 0.0134, "symmetry error": 0.01389, "fractal dimension error": 0.003532, 
  "worst radius": 24.99, "worst texture": 23.41, "worst perimeter": 158.8, "worst area": 1956.0, 
  "worst smoothness": 0.1238, "worst compactness": 0.1866, "worst concavity": 0.2416, 
  "worst concave points": 0.186, "worst symmetry": 0.275, "worst fractal dimension": 0.08902
};

type ProcessingState = "idle" | "validating" | "preparing" | "running" | "complete" | "error";

interface LoadedSampleInfo {
  type: "Benign" | "Malignant";
  rowIndex: number;
}

export default function Predict() {
  const [activeTab, setActiveTab] = useState<"image" | "tabular">("image");

  // Tabular State
  const [features, setFeatures] = useState<Record<string, string | number>>(DEFAULT_FEATURES);
  const [processState, setProcessState] = useState<ProcessingState>("idle");
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [sampleInfo, setSampleInfo] = useState<LoadedSampleInfo | null>(null);

  // Image State
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [imageProcessState, setImageProcessState] = useState<"idle" | "running" | "complete" | "error">("idle");
  const [imageResult, setImageResult] = useState<ImagePredictionResponse | null>(null);
  const [imageErrorMsg, setImageErrorMsg] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);

  // Validations
  const getFieldError = useCallback((key: string, val: string | number): string | null => {
    const strVal = String(val).trim();
    if (strVal === "") return "Required";
    const num = Number(strVal);
    if (isNaN(num) || !isFinite(num)) return "Invalid";
    return null;
  }, []);

  const getFieldWarning = useCallback((key: string, val: string | number): string | null => {
    const strVal = String(val).trim();
    if (strVal === "") return null;
    const num = Number(strVal);
    if (isNaN(num) || !isFinite(num)) return null;
    const range = DATASET_STATS[key];
    if (range) {
      if (num < range.min || num > range.max) {
        return `[${range.min} - ${range.max}]`;
      }
    }
    return null;
  }, []);

  const handleChange = useCallback((key: string, value: string) => {
    setFeatures((prev) => ({ ...prev, [key]: value }));
    setSampleInfo(null);
  }, []);

  const hasValidationErrors = useMemo(() => {
    return Object.keys(features).some((key) => getFieldError(key, features[key]) !== null);
  }, [features, getFieldError]);

  const handlePredict = async () => {
    if (hasValidationErrors) {
      setErrorMsg("Please correct all validation errors before running analysis.");
      return;
    }
    
    setErrorMsg(null);
    setProcessState("validating");
    
    setTimeout(() => setProcessState("preparing"), 400);
    setTimeout(() => setProcessState("running"), 800);
    
    try {
      const numericFeatures = Object.fromEntries(
        Object.entries(features).map(([k, v]) => [k, Number(v)])
      ) as unknown as BreastCancerFeatures;
      
      const res = await predictCancer(numericFeatures);
      
      setTimeout(() => {
        setResult(res);
        setProcessState("complete");
      }, 1400);
      
    } catch (err: any) {
      setTimeout(() => {
        setErrorMsg(err.message || "Failed to run prediction analysis.");
        setProcessState("error");
      }, 1400);
    }
  };

  const handleFileSelection = (f: File) => {
    if (!f.type.startsWith("image/")) {
      setImageErrorMsg("Please upload a valid image file (JPEG or PNG).");
      return;
    }
    if (f.size > 10 * 1024 * 1024) {
      setImageErrorMsg("File too large. Maximum size is 10 MB.");
      return;
    }
    setImageFile(f);
    setImagePreview(URL.createObjectURL(f));
    setImageErrorMsg(null);
    setImageResult(null);
    setImageProcessState("idle");
  };

  const handleImagePredict = async () => {
    if (!imageFile) return;
    
    setImageProcessState("running");
    setImageErrorMsg(null);
    setImageResult(null);

    try {
      const res = await predictImage(imageFile);
      setImageResult(res);
      setImageProcessState("complete");
    } catch (err: any) {
      setImageErrorMsg(err.message || "Unable to analyze the image.");
      setImageProcessState("error");
    }
  };

  return (
    <div className="max-w-[1400px] mx-auto px-4 sm:px-6 lg:px-8 py-8 lg:py-12">
      
      {/* Header and Tab Switcher */}
      <div className="flex flex-col md:flex-row md:items-end justify-between mb-8 gap-4 border-b border-slate-200 pb-4">
        <div>
          <h1 className="text-4xl font-extrabold text-slate-900 tracking-tight">AI Analysis Workspace</h1>
          <p className="text-slate-500 mt-2 text-base font-medium">
            Select an analysis method to classify tumor risk probabilities.
          </p>
        </div>
        
        <div className="flex bg-slate-100 p-1.5 rounded-xl shadow-sm border border-slate-200">
          <button
            onClick={() => setActiveTab("image")}
            className={`py-2 px-6 font-bold text-sm rounded-lg transition-all ${
              activeTab === "image"
                ? "bg-white text-medical-700 shadow-sm border border-slate-200/50"
                : "text-slate-500 hover:text-slate-700 hover:bg-slate-50"
            }`}
          >
            Image Classification
          </button>
          <button
            onClick={() => setActiveTab("tabular")}
            className={`py-2 px-6 font-bold text-sm rounded-lg transition-all ${
              activeTab === "tabular"
                ? "bg-white text-medical-700 shadow-sm border border-slate-200/50"
                : "text-slate-500 hover:text-slate-700 hover:bg-slate-50"
            }`}
          >
            Tabular SVM
          </button>
        </div>
      </div>

      {activeTab === "image" ? (
        /* Image Analysis Workspace */
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column: Upload & Preview */}
          <div className="space-y-6">
            <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm flex flex-col h-full">
              <h2 className="text-xl font-bold text-slate-900 mb-4 flex items-center">
                <ImageIcon className="w-5 h-5 mr-2 text-medical-600" /> Image Input
              </h2>
              
              {!imageFile ? (
                <div 
                  className={`flex-grow border-2 border-dashed rounded-xl p-12 flex flex-col items-center justify-center transition-colors cursor-pointer min-h-[300px] ${
                    dragActive ? "border-medical-400 bg-medical-50" : "border-slate-300 hover:border-medical-300 hover:bg-slate-50"
                  }`}
                  onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
                  onDragLeave={() => setDragActive(false)}
                  onDrop={(e) => { e.preventDefault(); setDragActive(false); if (e.dataTransfer.files?.[0]) handleFileSelection(e.dataTransfer.files[0]); }}
                  onClick={() => document.getElementById("image-file-input")?.click()}
                >
                  <Upload className="w-12 h-12 text-slate-400 mb-4" />
                  <p className="text-slate-700 font-bold text-lg mb-1">Drag and drop ultrasound image</p>
                  <p className="text-slate-400 text-sm font-medium">or click to browse files</p>
                  <input 
                    id="image-file-input" 
                    type="file" 
                    accept="image/jpeg,image/png,image/jpg" 
                    className="hidden"
                    onChange={(e) => e.target.files?.[0] && handleFileSelection(e.target.files[0])}
                  />
                </div>
              ) : (
                <div className="flex-grow flex flex-col space-y-4">
                  <div className="bg-slate-50 rounded-xl p-4 flex items-center justify-between border border-slate-100">
                    <div className="flex items-center gap-3">
                      <div className="bg-white p-2 rounded-lg shadow-sm">
                        <ImageIcon className="w-5 h-5 text-medical-600" />
                      </div>
                      <div>
                        <p className="text-sm font-bold text-slate-800">{imageFile.name}</p>
                        <p className="text-xs text-slate-500 font-medium">{(imageFile.size / 1024).toFixed(1)} KB</p>
                      </div>
                    </div>
                    <button 
                      onClick={() => { setImageFile(null); setImagePreview(null); setImageResult(null); setImageProcessState("idle"); }} 
                      className="text-xs font-bold text-red-600 hover:text-red-700 bg-red-50 hover:bg-red-100 px-3 py-1.5 rounded-lg transition"
                    >
                      Remove
                    </button>
                  </div>

                  <div className="flex-grow bg-slate-50 border border-slate-100 rounded-xl p-2 flex items-center justify-center min-h-[300px]">
                    {imagePreview && <img src={imagePreview} alt="Preview" className="max-h-[300px] object-contain rounded-lg shadow-sm" />}
                  </div>

                  <button
                    onClick={handleImagePredict}
                    disabled={imageProcessState === "running"}
                    className="w-full py-4 bg-medical-600 text-white rounded-xl font-bold text-lg shadow-md hover:bg-medical-700 disabled:opacity-50 transition flex items-center justify-center"
                  >
                    {imageProcessState === "running" ? <><Loader2 className="w-6 h-6 animate-spin mr-2" /> Analyzing...</> : <><PlayCircle className="w-6 h-6 mr-2" /> Analyze Image</>}
                  </button>
                </div>
              )}

              {imageErrorMsg && (
                <div className="mt-4 p-4 bg-red-50 border border-red-200 text-red-700 font-medium rounded-xl flex items-start">
                  <AlertCircle className="w-5 h-5 mr-3 flex-shrink-0 mt-0.5" />
                  <span>{imageErrorMsg}</span>
                </div>
              )}
            </div>
          </div>

          {/* Right Column: Prediction Results */}
          <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm flex flex-col min-h-[500px]">
            <h2 className="text-xl font-bold text-slate-900 mb-6 border-b border-slate-100 pb-4 flex items-center">
              <Activity className="w-5 h-5 mr-2 text-medical-600" /> Prediction Results
            </h2>
            
            {imageProcessState === "idle" && (
              <div className="flex-grow flex flex-col items-center justify-center text-center">
                <div className="w-20 h-20 bg-slate-50 rounded-full flex items-center justify-center mb-6">
                  <ImageIcon className="w-10 h-10 text-slate-300" />
                </div>
                <h3 className="text-lg font-bold text-slate-700 mb-2">Ready for Analysis</h3>
                <p className="text-sm text-slate-500 max-w-sm">Upload a breast ultrasound image and click &quot;Analyze Image&quot; to view the predicted class and confidence distribution.</p>
              </div>
            )}

            {imageProcessState === "running" && (
              <div className="flex-grow flex flex-col items-center justify-center text-center">
                <Loader2 className="w-12 h-12 text-medical-500 animate-spin mb-6" />
                <h3 className="text-lg font-bold text-slate-700 mb-1">Applying MobileNetV2...</h3>
                <p className="text-sm text-slate-500">Extracting deep visual features</p>
              </div>
            )}

            {imageProcessState === "complete" && imageResult && (
              <AnimatePresence>
                <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="flex-grow flex flex-col">
                  <div className={`p-8 rounded-2xl text-center mb-8 border shadow-sm ${
                    imageResult.predicted_class.toLowerCase() === "malignant" ? "bg-red-50 border-red-100" : "bg-teal-50 border-teal-100"
                  }`}>
                    <p className="text-sm font-bold uppercase tracking-wider text-slate-500 mb-2">Primary Prediction</p>
                    <h3 className={`text-4xl md:text-5xl font-black tracking-tight ${
                      imageResult.predicted_class.toLowerCase() === "malignant" ? "text-red-700" : "text-teal-700"
                    }`}>
                      {imageResult.predicted_class.toUpperCase()}
                    </h3>
                  </div>

                  <div className="space-y-5 mb-8 flex-grow">
                    <p className="text-sm font-bold text-slate-800 uppercase tracking-wide border-b border-slate-100 pb-2">Class Probabilities</p>
                    {["benign", "malignant", "normal"].map((cls) => {
                      const prob = imageResult.probabilities[cls] || 0;
                      const isMalignant = cls === "malignant";
                      const colorClass = isMalignant ? "bg-red-500" : cls === "benign" ? "bg-teal-500" : "bg-medical-500";
                      const textClass = isMalignant ? "text-red-700 font-bold" : cls === "benign" ? "text-teal-700 font-bold" : "text-medical-700 font-bold";

                      return (
                        <div key={cls}>
                          <div className="flex justify-between text-sm mb-2">
                            <span className={`capitalize ${textClass}`}>{cls}</span>
                            <span className={`${textClass}`}>{(prob * 100).toFixed(2)}%</span>
                          </div>
                          <div className="w-full bg-slate-100 rounded-full h-2.5">
                            <motion.div initial={{ width: 0 }} animate={{ width: `${prob * 100}%` }} transition={{ duration: 0.8 }} className={`${colorClass} h-2.5 rounded-full`} />
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  <div className="p-4 bg-slate-50 border border-slate-200 rounded-xl text-xs text-slate-500 leading-relaxed font-medium mt-auto">
                    <strong>DISCLAIMER:</strong> This AI prediction is not a medical diagnosis. Please consult a qualified healthcare professional.
                  </div>
                </motion.div>
              </AnimatePresence>
            )}
          </div>
        </div>
      ) : (
        /* Tabular Analysis Workspace */
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          
          {/* Form Area - Spans 3 columns on large screens */}
          <div className="lg:col-span-3 bg-white p-6 md:p-8 rounded-2xl border border-slate-200 shadow-sm space-y-8">
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 border-b border-slate-100 pb-4">
              <h2 className="text-xl font-bold text-slate-900 flex items-center">
                <Dna className="w-5 h-5 mr-2 text-medical-600" /> Nuclei Measurement Inputs
              </h2>
              <div className="flex space-x-2">
                <button 
                  onClick={() => { setFeatures(DEFAULT_FEATURES); setSampleInfo(null); setResult(null); setProcessState("idle"); }}
                  className="px-3 py-1.5 text-sm font-bold text-slate-600 bg-slate-100 hover:bg-slate-200 rounded-lg transition"
                >
                  Clear Form
                </button>
                <button 
                  onClick={() => { setFeatures(SAMPLE_BENIGN); setSampleInfo({ type: "Benign", rowIndex: 19 }); }}
                  className="px-3 py-1.5 text-sm font-bold text-teal-700 bg-teal-50 hover:bg-teal-100 border border-teal-200 rounded-lg transition"
                >
                  Load Benign
                </button>
                <button 
                  onClick={() => { setFeatures(SAMPLE_MALIGNANT); setSampleInfo({ type: "Malignant", rowIndex: 1 }); }}
                  className="px-3 py-1.5 text-sm font-bold text-red-700 bg-red-50 hover:bg-red-100 border border-red-200 rounded-lg transition"
                >
                  Load Malignant
                </button>
              </div>
            </div>

            {sampleInfo && (
              <div className="p-4 bg-medical-50 border border-medical-100 rounded-xl flex items-start space-x-3">
                <Info className="w-5 h-5 text-medical-600 shrink-0 mt-0.5" />
                <div className="text-sm text-medical-800 font-medium">
                  <strong>Loaded Reference Data:</strong> {sampleInfo.type} (Row {sampleInfo.rowIndex})
                </div>
              </div>
            )}

            <div className="space-y-10">
              <MemoizedFeatureGroup title="Mean Values" features={features} prefix="mean " onChange={handleChange} getFieldError={getFieldError} getFieldWarning={getFieldWarning} />
              <MemoizedFeatureGroup title="Standard Errors" features={features} prefix="" suffix=" error" onChange={handleChange} getFieldError={getFieldError} getFieldWarning={getFieldWarning} />
              <MemoizedFeatureGroup title="Worst Values" features={features} prefix="worst " onChange={handleChange} getFieldError={getFieldError} getFieldWarning={getFieldWarning} />
            </div>

            {errorMsg && (
              <div className="p-4 bg-red-50 border border-red-200 text-red-700 font-bold rounded-xl flex items-start">
                <AlertCircle className="w-5 h-5 mr-3 flex-shrink-0 mt-0.5" />
                <span>{errorMsg}</span>
              </div>
            )}

            <button
              onClick={handlePredict}
              disabled={hasValidationErrors || (processState !== "idle" && processState !== "complete" && processState !== "error")}
              className="w-full py-4 bg-slate-900 text-white rounded-xl font-bold text-lg shadow-md hover:bg-slate-800 disabled:opacity-50 transition flex items-center justify-center mt-8"
            >
              {processState !== "idle" && processState !== "complete" && processState !== "error" ? <RefreshCw className="w-6 h-6 animate-spin mr-2" /> : <PlayCircle className="w-6 h-6 mr-2" />}
              Execute Tabular SVM Prediction
            </button>
          </div>

          {/* Results Area - Spans 1 column */}
          <div className="lg:col-span-1 bg-white p-6 rounded-2xl border border-slate-200 shadow-sm flex flex-col min-h-[500px] sticky top-24">
            <h2 className="text-xl font-bold text-slate-900 border-b border-slate-100 pb-4 mb-6">SVM Results</h2>

            {processState === "idle" && (
               <div className="flex-grow flex flex-col items-center justify-center text-center">
                 <Dna className="w-16 h-16 mb-4 opacity-20 text-slate-400" />
                 <h3 className="text-lg font-bold text-slate-700 mb-2">Awaiting Data</h3>
                 <p className="text-sm text-slate-500 font-medium">Fill the form and execute analysis.</p>
               </div>
            )}

            {(processState === "validating" || processState === "preparing" || processState === "running") && (
              <div className="flex-grow flex flex-col justify-center space-y-6">
                <ProcessingStep text="Validating inputs..." active={processState === "validating"} done={processState === "preparing" || processState === "running"} />
                <ProcessingStep text="Scaling features..." active={processState === "preparing"} done={processState === "running"} />
                <ProcessingStep text="Running RBF Kernel..." active={processState === "running"} done={false} />
              </div>
            )}

            {processState === "complete" && result && (
              <AnimatePresence>
                <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="flex-grow flex flex-col">
                  <div className={`py-6 rounded-xl text-center mb-6 border shadow-sm ${result.prediction_code === "M" ? "bg-red-50 border-red-100" : "bg-teal-50 border-teal-100"}`}>
                    <p className="text-xs font-bold uppercase tracking-wider text-slate-500 mb-1">Prediction</p>
                    <h3 className={`text-3xl font-black tracking-tight ${result.prediction_code === "M" ? "text-red-700" : "text-teal-700"}`}>
                      {result.prediction.toUpperCase()}
                    </h3>
                  </div>

                  <div className="space-y-4 mb-6 flex-grow">
                    <p className="text-xs font-bold text-slate-800 uppercase tracking-wide border-b border-slate-100 pb-2">Probabilities</p>
                    <div>
                      <div className="flex justify-between text-sm mb-1.5">
                        <span className="text-teal-700 font-bold">Benign</span>
                        <span className="text-teal-700 font-bold">{(result.probabilities["benign"] * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-slate-100 rounded-full h-2">
                        <motion.div initial={{ width: 0 }} animate={{ width: `${result.probabilities["benign"] * 100}%` }} className="bg-teal-500 h-2 rounded-full" />
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between text-sm mb-1.5 mt-4">
                        <span className="text-red-700 font-bold">Malignant</span>
                        <span className="text-red-700 font-bold">{(result.probabilities["malignant"] * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-slate-100 rounded-full h-2">
                        <motion.div initial={{ width: 0 }} animate={{ width: `${result.probabilities["malignant"] * 100}%` }} className="bg-red-500 h-2 rounded-full" />
                      </div>
                    </div>
                  </div>

                  {sampleInfo && (
                    <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg text-xs font-medium text-slate-600 mb-4">
                      <p className="font-bold text-slate-800 mb-1">Validation Match</p>
                      <p>Expected: <span className="uppercase">{sampleInfo.type}</span></p>
                      <p>Predicted: <span className="uppercase">{result.prediction}</span></p>
                    </div>
                  )}

                  <div className="mt-auto pt-4 border-t border-slate-100 text-xs text-slate-400 font-medium text-center">
                    {result.model.name} v{result.model.version}
                  </div>
                </motion.div>
              </AnimatePresence>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function ProcessingStep({ text, active, done }: { text: string, active: boolean, done: boolean }) {
  return (
    <div className={`flex items-center space-x-3 transition-opacity duration-300 ${active ? "opacity-100" : done ? "opacity-50" : "opacity-30"}`}>
      <div className={`w-6 h-6 rounded-full flex items-center justify-center ${active ? "bg-medical-100" : done ? "bg-teal-100" : "bg-slate-100"}`}>
        {active && <div className="w-2.5 h-2.5 bg-medical-600 rounded-full animate-ping" />}
        {done && <div className="w-2.5 h-2.5 bg-teal-600 rounded-full" />}
        {!active && !done && <div className="w-2.5 h-2.5 bg-slate-300 rounded-full" />}
      </div>
      <span className={`font-bold text-sm ${active ? "text-medical-700" : done ? "text-teal-700" : "text-slate-500"}`}>{text}</span>
    </div>
  );
}

const MemoizedFeatureGroup = React.memo(FeatureGroup);

function FeatureGroup({ 
  title, features, prefix, suffix = "", onChange, getFieldError, getFieldWarning 
}: { 
  title: string; features: any; prefix: string; suffix?: string; 
  onChange: (k: string, v: string) => void;
  getFieldError: (key: string, val: string | number) => string | null;
  getFieldWarning: (key: string, val: string | number) => string | null;
}) {
  const baseKeys = [
    "radius", "texture", "perimeter", "area", "smoothness", 
    "compactness", "concavity", "concave points", "symmetry", "fractal dimension"
  ];
  
  return (
    <div>
      <h3 className="text-lg font-bold text-slate-800 mb-4">{title}</h3>
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 xl:grid-cols-5 gap-3">
        {baseKeys.map((base) => {
          const key = `${prefix}${base}${suffix}`;
          const err = getFieldError(key, features[key]);
          const warning = getFieldWarning(key, features[key]);
          const desc = CORE_DESCRIPTIONS[base.replace(/^\w/, (c) => c.toUpperCase())] || "";

          return (
            <div key={key} className="flex flex-col relative group">
              <label className="text-xs font-bold text-slate-600 mb-1.5 capitalize flex items-center gap-1 cursor-help" title={desc}>
                {base} <HelpCircle className="w-3 h-3 text-slate-400 group-hover:text-medical-500 transition-colors" />
              </label>
              <input 
                type="number" step="any" value={features[key]} onChange={(e) => onChange(key, e.target.value)}
                className={`w-full px-3 py-2 border rounded-lg text-sm transition-colors focus:outline-none focus:ring-2 font-medium ${
                  err ? "border-red-300 focus:ring-red-500 bg-red-50/50 text-red-900" : warning ? "border-amber-300 focus:ring-amber-500 bg-amber-50/20 text-amber-900" : "border-slate-300 focus:ring-medical-500 focus:border-medical-500 bg-slate-50 focus:bg-white text-slate-900"
                }`}
                placeholder="0.0"
              />
              {err ? <span className="absolute -bottom-4 left-0 text-[10px] text-red-600 font-bold truncate w-full">{err}</span> : 
               warning ? <span className="absolute -bottom-4 left-0 text-[10px] text-amber-600 font-bold truncate w-full">{warning}</span> : null}
            </div>
          );
        })}
      </div>
    </div>
  );
}
