/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";

import React, { useState } from "react";
import { predictImage, ImagePredictionResponse } from "@/lib/api";
import { AlertCircle, PlayCircle, Upload, ImageIcon, Loader2, Activity } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

export default function PredictPage() {
  // Image State
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [selectedImageModel, setSelectedImageModel] = useState<"v5b" | "v14">("v5b");
  const [imageProcessState, setImageProcessState] = useState<"idle" | "running" | "complete" | "error">("idle");
  const [imageResult, setImageResult] = useState<ImagePredictionResponse | null>(null);
  const [imageErrorMsg, setImageErrorMsg] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [viewOverlay, setViewOverlay] = useState<boolean>(false);

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
    setViewOverlay(false);
  };

  const handleImagePredict = async () => {
    if (!imageFile) return;
    
    setImageProcessState("running");
    setImageErrorMsg(null);
    setImageResult(null);
    setViewOverlay(false);

    try {
      const res = await predictImage(imageFile, selectedImageModel);
      setImageResult(res);
      setImageProcessState("complete");
    } catch (err: any) {
      setImageErrorMsg(err.message || "Unable to analyze the image.");
      setImageProcessState("error");
    }
  };

  const handleLoadSample = async (samplePath: string, fileName: string) => {
    try {
      const response = await fetch(samplePath);
      const blob = await response.blob();
      const file = new File([blob], fileName, { type: "image/png" });
      handleFileSelection(file);
    } catch {
      setImageErrorMsg("Failed to load sample medical image.");
    }
  };

  const handleModelChange = (m: "v5b" | "v14") => {
    setSelectedImageModel(m);
    setImageResult(null);
    setImageProcessState("idle");
    setViewOverlay(false);
  };

  return (
    <div className="max-w-[1400px] mx-auto px-4 sm:px-6 lg:px-8 py-8 lg:py-12">
      
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between mb-8 gap-4 border-b border-slate-200 pb-4">
        <div>
          <h1 className="text-4xl font-extrabold text-slate-900 tracking-tight">Ultrasound Image AI Workspace</h1>
          <p className="text-slate-500 mt-2 text-base font-medium">
            Upload a breast ultrasound scan for AI-assisted diagnostic triage and visual Grad-CAM analysis.
          </p>
        </div>
      </div>

      {/* Image Analysis Workspace */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Left Column: Upload & Preview */}
        <div className="space-y-6">
          <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm flex flex-col h-full">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-4 gap-2">
              <h2 className="text-xl font-bold text-slate-900 flex items-center">
                <ImageIcon className="w-5 h-5 mr-2 text-medical-600" /> Ultrasound Scan Input
              </h2>

              {/* Research Model Selector Toggle */}
              <div className="flex bg-slate-100 p-1 rounded-lg border border-slate-200 text-xs">
                <button
                  onClick={() => handleModelChange("v5b")}
                  className={`px-3 py-1 font-bold rounded-md transition ${
                    selectedImageModel === "v5b"
                      ? "bg-white text-slate-900 shadow-sm"
                      : "text-slate-500 hover:text-slate-800"
                  }`}
                >
                  V5-B (Production)
                </button>
                <button
                  onClick={() => handleModelChange("v14")}
                  className={`px-3 py-1 font-bold rounded-md transition ${
                    selectedImageModel === "v14"
                      ? "bg-white text-medical-700 shadow-sm"
                      : "text-slate-500 hover:text-slate-800"
                  }`}
                >
                  V14 Ensemble (Research)
                </button>
              </div>
            </div>
            
            {!imageFile ? (
              <div 
                className={`flex-grow border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center transition-colors cursor-pointer min-h-[300px] ${
                  dragActive ? "border-medical-400 bg-medical-50" : "border-slate-300 hover:border-medical-300 hover:bg-slate-50"
                }`}
                onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
                onDragLeave={() => setDragActive(false)}
                onDrop={(e) => { e.preventDefault(); setDragActive(false); if (e.dataTransfer.files?.[0]) handleFileSelection(e.dataTransfer.files[0]); }}
                onClick={() => document.getElementById("image-file-input")?.click()}
              >
                <Upload className="w-12 h-12 text-slate-400 mb-4" />
                <p className="text-slate-700 font-bold text-lg mb-1">Drag and drop ultrasound image</p>
                <p className="text-slate-400 text-sm font-medium mb-4">or click to browse files</p>
                <input 
                  id="image-file-input" 
                  type="file" 
                  accept="image/jpeg,image/png,image/jpg" 
                  className="hidden"
                  onChange={(e) => e.target.files?.[0] && handleFileSelection(e.target.files[0])}
                />

                {/* Sample Medical Ultrasound Loaders */}
                <div className="pt-4 border-t border-slate-200 w-full flex flex-col items-center">
                  <p className="text-xs text-slate-500 font-bold uppercase tracking-wider mb-2">Or test with real clinical ultrasound samples:</p>
                  <div className="flex flex-wrap gap-2 justify-center">
                    <button
                      type="button"
                      onClick={(e) => { e.stopPropagation(); handleLoadSample("/samples/benign_sample.png", "benign_scan_sample.png"); }}
                      className="px-3 py-1.5 bg-emerald-50 hover:bg-emerald-100 border border-emerald-200 text-emerald-800 text-xs font-bold rounded-lg transition"
                    >
                      + Sample Benign Scan
                    </button>
                    <button
                      type="button"
                      onClick={(e) => { e.stopPropagation(); handleLoadSample("/samples/malignant_sample.png", "malignant_scan_sample.png"); }}
                      className="px-3 py-1.5 bg-rose-50 hover:bg-rose-100 border border-rose-200 text-rose-800 text-xs font-bold rounded-lg transition"
                    >
                      + Sample Malignant Scan
                    </button>
                    <button
                      type="button"
                      onClick={(e) => { e.stopPropagation(); handleLoadSample("/samples/normal_sample.png", "normal_scan_sample.png"); }}
                      className="px-3 py-1.5 bg-sky-50 hover:bg-sky-100 border border-sky-200 text-sky-800 text-xs font-bold rounded-lg transition"
                    >
                      + Sample Normal Scan
                    </button>
                  </div>
                </div>
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
                    onClick={() => { setImageFile(null); setImagePreview(null); setImageResult(null); setImageProcessState("idle"); setViewOverlay(false); }} 
                    className="text-xs font-bold text-red-600 hover:text-red-700 bg-red-50 hover:bg-red-100 px-3 py-1.5 rounded-lg transition"
                  >
                    Remove
                  </button>
                </div>

                <div className="flex-grow bg-slate-50 border border-slate-100 rounded-xl p-2 flex flex-col items-center justify-center min-h-[300px] relative">
                  {imageResult?.explanation?.available && (
                    <div className="absolute top-3 right-3 z-10 bg-white/90 backdrop-blur-sm p-1 rounded-lg shadow-sm border border-slate-200 flex gap-1">
                      <button
                        onClick={() => setViewOverlay(false)}
                        className={`px-2.5 py-1 text-xs font-bold rounded-md transition ${!viewOverlay ? "bg-slate-900 text-white" : "text-slate-600 hover:bg-slate-100"}`}
                      >
                        Original
                      </button>
                      <button
                        onClick={() => setViewOverlay(true)}
                        className={`px-2.5 py-1 text-xs font-bold rounded-md transition ${viewOverlay ? "bg-medical-600 text-white" : "text-slate-600 hover:bg-slate-100"}`}
                      >
                        Grad-CAM
                      </button>
                    </div>
                  )}
                  {imagePreview && (
                    <img 
                      src={viewOverlay && imageResult?.explanation?.overlay ? imageResult.explanation.overlay : imagePreview} 
                      alt="Ultrasound Preview" 
                      className="max-h-[350px] w-auto object-contain rounded-lg shadow-sm"
                    />
                  )}
                </div>

                <button
                  onClick={handleImagePredict}
                  disabled={imageProcessState === "running"}
                  className="w-full py-4 bg-medical-600 hover:bg-medical-700 text-white font-bold rounded-xl shadow-md hover:shadow-lg transition flex items-center justify-center gap-2 disabled:opacity-50"
                >
                  {imageProcessState === "running" ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      Analyzing Ultrasound Scan...
                    </>
                  ) : (
                    <>
                      <PlayCircle className="w-5 h-5" />
                      Run Image Classification ({selectedImageModel === "v5b" ? "V5-B Production" : "V14 Research Ensemble"})
                    </>
                  )}
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Image Results */}
        <div className="space-y-6">
          <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm min-h-[400px] flex flex-col">
            <h2 className="text-xl font-bold text-slate-900 mb-6 flex items-center border-b border-slate-100 pb-4">
              <Activity className="w-5 h-5 mr-2 text-medical-600" /> Image Verdict & Triage
            </h2>

            <AnimatePresence mode="wait">
              {imageProcessState === "idle" && (
                <motion.div 
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  className="flex-grow flex flex-col items-center justify-center text-center p-8 border-2 border-dashed border-slate-100 rounded-xl bg-slate-50/50"
                >
                  <Activity className="w-12 h-12 text-slate-300 mb-3" />
                  <p className="text-slate-600 font-bold mb-1">Awaiting Image Input</p>
                  <p className="text-slate-400 text-sm max-w-xs font-medium">Upload a breast ultrasound scan on the left and click Run Image Classification.</p>
                </motion.div>
              )}

              {imageProcessState === "error" && (
                <motion.div 
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  className="p-6 bg-red-50 border border-red-200 rounded-xl"
                >
                  <div className="flex items-center gap-3 text-red-800 font-bold mb-2">
                    <AlertCircle className="w-5 h-5" /> Image Inference Error
                  </div>
                  <p className="text-red-700 text-sm font-medium">{imageErrorMsg}</p>
                </motion.div>
              )}

              {imageProcessState === "complete" && imageResult && (
                <motion.div 
                  initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                  className="space-y-6"
                >
                  {/* Verdict Badge */}
                  <div className={`p-6 rounded-2xl border flex items-center justify-between shadow-sm ${
                    imageResult.predicted_class === "malignant" 
                      ? "bg-red-50 border-red-200 text-red-900" 
                      : imageResult.predicted_class === "benign"
                      ? "bg-teal-50 border-teal-200 text-teal-900"
                      : "bg-slate-50 border-slate-200 text-slate-900"
                  }`}>
                    <div>
                      <span className="text-xs font-bold uppercase tracking-wider opacity-75">Diagnostic Triage Verdict</span>
                      <h3 className="text-3xl font-extrabold capitalize mt-1 tracking-tight">{imageResult.predicted_class}</h3>
                    </div>
                    <div className="text-right">
                      <span className="text-xs font-bold uppercase tracking-wider opacity-75">Confidence</span>
                      <p className="text-2xl font-black mt-1">{(imageResult.confidence * 100).toFixed(1)}%</p>
                    </div>
                  </div>

                  {/* Quality Notices */}
                  {imageResult.quality_warnings && imageResult.quality_warnings.length > 0 && (
                    <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 text-xs text-amber-900 space-y-1">
                      <p className="font-bold flex items-center gap-1.5">
                        <AlertCircle className="w-4 h-4 text-amber-600" /> Image Quality Notices:
                      </p>
                      <ul className="list-disc list-inside font-medium pl-1">
                        {imageResult.quality_warnings.map((w, i) => (
                          <li key={i}>{w}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Probability Bars */}
                  <div className="bg-slate-50 rounded-xl p-6 border border-slate-100 space-y-4">
                    <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500">Class Probabilities</h4>
                    <div className="space-y-3">
                      {Object.entries(imageResult.probabilities).map(([cls, prob]) => (
                        <div key={cls} className="space-y-1">
                          <div className="flex justify-between text-sm font-bold text-slate-700 capitalize">
                            <span>{cls}</span>
                            <span>{(prob * 100).toFixed(1)}%</span>
                          </div>
                          <div className="w-full bg-slate-200 rounded-full h-2.5 overflow-hidden">
                            <div 
                              className={`h-2.5 rounded-full transition-all duration-500 ${
                                cls === "malignant" ? "bg-red-500" : cls === "benign" ? "bg-teal-500" : "bg-slate-400"
                              }`}
                              style={{ width: `${(prob * 100).toFixed(1)}%` }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Disclaimer */}
                  <p className="text-xs text-slate-400 font-medium leading-relaxed">
                    Evaluated model version: <span className="font-bold text-slate-600">{imageResult.model_version || selectedImageModel}</span>. Grad-CAM overlays highlight regions of interest. Not intended as a standalone clinical diagnosis.
                  </p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </div>
  );
}
