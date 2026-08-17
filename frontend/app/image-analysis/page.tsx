/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";

import { useState, useCallback } from "react";
import { extractImageFeatures, predictFromImage, ImageAnalysisResponse } from "@/lib/api";
import { 
  Upload, AlertTriangle, CheckCircle2, XCircle, Microscope, 
  FlaskConical, ShieldAlert, Loader2, ImageIcon, Info
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

type Stage = "idle" | "uploading" | "processing" | "done" | "error";

export default function ImageAnalysisPage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [stage, setStage] = useState<Stage>("idle");
  const [result, setResult] = useState<ImageAnalysisResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);

  const handleFile = useCallback((f: File) => {
    if (!f.type.startsWith("image/")) {
      setErrorMsg("Please upload a valid image file (JPEG or PNG).");
      return;
    }
    if (f.size > 10 * 1024 * 1024) {
      setErrorMsg("File too large. Maximum size is 10 MB.");
      return;
    }
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setErrorMsg(null);
    setStage("idle");
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    if (e.dataTransfer.files?.[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  }, [handleFile]);

  const handleExtract = async () => {
    if (!file) return;
    setStage("uploading");
    setErrorMsg(null);
    setResult(null);

    setTimeout(() => setStage("processing"), 400);

    try {
      const res = await extractImageFeatures(file);
      setResult(res);
      setStage("done");
    } catch (err: any) {
      setErrorMsg(err.message || "Feature extraction failed.");
      setStage("error");
    }
  };

  const handlePredict = async () => {
    if (!file) return;
    setStage("uploading");
    setErrorMsg(null);
    setResult(null);

    setTimeout(() => setStage("processing"), 400);

    try {
      const res = await predictFromImage(file);
      setResult(res);
      setStage("done");
    } catch (err: any) {
      setErrorMsg(err.message || "Prediction failed.");
      setStage("error");
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setErrorMsg(null);
    setStage("idle");
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <Microscope className="w-8 h-8 text-purple-600" />
          <h1 className="text-3xl font-extrabold text-gray-900 tracking-tight">
            Image Feature Extraction
          </h1>
          <span className="px-2.5 py-1 bg-amber-100 text-amber-800 text-[10px] font-bold rounded-full uppercase tracking-wider border border-amber-200">
            Experimental
          </span>
        </div>
        <p className="text-gray-500 text-sm leading-relaxed max-w-3xl">
          This experimental workflow extracts morphological and texture measurements from cell nuclei 
          in microscopy images. Extracted features are validated against the training dataset distribution 
          before any prediction is attempted.
        </p>
      </div>

      {/* Guidance Panel */}
      <div className="mb-8 p-4 bg-blue-50 border border-blue-200 rounded-xl">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-blue-600 shrink-0 mt-0.5" />
          <div className="text-xs text-blue-800 space-y-1">
            <p className="font-bold text-sm">Image Requirements</p>
            <p>This experimental workflow is intended for suitable microscopic or digitized cell nuclei imagery. 
               It is <strong>not compatible</strong> with ordinary photographs, mammograms, or general X-ray images.</p>
            <p>Supported formats: <strong>JPEG, PNG</strong> — Maximum size: <strong>10 MB</strong></p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left: Upload + Diagnostics */}
        <div className="lg:col-span-2 space-y-6">
          
          {/* Upload Zone */}
          {!file ? (
            <div 
              className={`border-2 border-dashed rounded-2xl p-12 text-center transition-colors cursor-pointer ${
                dragActive ? "border-purple-400 bg-purple-50" : "border-gray-300 hover:border-purple-300 hover:bg-gray-50"
              }`}
              onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
              onDragLeave={() => setDragActive(false)}
              onDrop={handleDrop}
              onClick={() => document.getElementById("file-input")?.click()}
            >
              <Upload className="w-12 h-12 mx-auto text-gray-400 mb-4" />
              <p className="text-gray-600 font-medium mb-1">Drop a cell microscopy image here</p>
              <p className="text-gray-400 text-sm">or click to browse files</p>
              <input 
                id="file-input" 
                type="file" 
                accept="image/jpeg,image/png" 
                className="hidden"
                onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
              />
            </div>
          ) : (
            <div className="space-y-4">
              {/* Image Preview */}
              <div className="bg-white border rounded-2xl p-4 shadow-sm">
                <div className="flex justify-between items-center mb-3">
                  <div className="flex items-center gap-2">
                    <ImageIcon className="w-4 h-4 text-gray-500" />
                    <span className="text-sm font-medium text-gray-700">{file.name}</span>
                    <span className="text-xs text-gray-400">({(file.size / 1024).toFixed(1)} KB)</span>
                  </div>
                  <button onClick={reset} className="text-xs text-gray-500 hover:text-red-600 transition">
                    Remove
                  </button>
                </div>
                {preview && (
                  <img 
                    src={preview} 
                    alt="Uploaded preview" 
                    className="max-h-64 rounded-lg mx-auto border"
                  />
                )}
              </div>

              {/* Action Buttons */}
              <div className="flex gap-3">
                <button
                  onClick={handleExtract}
                  disabled={stage === "uploading" || stage === "processing"}
                  className="flex-1 py-3 bg-purple-600 text-white rounded-xl font-bold shadow-md hover:bg-purple-700 disabled:opacity-50 transition flex items-center justify-center gap-2"
                >
                  {(stage === "uploading" || stage === "processing") ? (
                    <Loader2 className="w-5 h-5 animate-spin" />
                  ) : (
                    <FlaskConical className="w-5 h-5" />
                  )}
                  Extract Features
                </button>
                <button
                  onClick={handlePredict}
                  disabled={stage === "uploading" || stage === "processing"}
                  className="flex-1 py-3 bg-blue-600 text-white rounded-xl font-bold shadow-md hover:bg-blue-700 disabled:opacity-50 transition flex items-center justify-center gap-2"
                >
                  {(stage === "uploading" || stage === "processing") ? (
                    <Loader2 className="w-5 h-5 animate-spin" />
                  ) : (
                    <Microscope className="w-5 h-5" />
                  )}
                  Extract & Predict
                </button>
              </div>
            </div>
          )}

          {errorMsg && (
            <div className="p-4 bg-red-50 border border-red-200 text-red-700 rounded-xl flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 shrink-0 mt-0.5" />
              <span className="text-sm">{errorMsg}</span>
            </div>
          )}

          {/* Diagnostic Images */}
          {result?.diagnostic_images && (
            <div className="bg-white border rounded-2xl p-6 shadow-sm">
              <h3 className="text-lg font-bold text-gray-900 mb-4">Segmentation Diagnostics</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {[
                  { key: "original", label: "Original" },
                  { key: "preprocessed", label: "Preprocessed" },
                  { key: "binary_mask", label: "Binary Mask" },
                  { key: "nuclei_overlay", label: "Detected Nuclei" },
                ].map(({ key, label }) => (
                  <div key={key} className="text-center">
                    <img 
                      src={`data:image/png;base64,${(result.diagnostic_images as any)[key]}`}
                      alt={label}
                      className="rounded-lg border w-full aspect-square object-contain bg-gray-50"
                    />
                    <p className="text-xs text-gray-500 mt-1 font-medium">{label}</p>
                  </div>
                ))}
              </div>
              <p className="text-xs text-gray-500 mt-3">
                <strong>Nuclei detected:</strong> {result.num_nuclei} | 
                <strong> Measured:</strong> {result.num_measured ?? 0}
              </p>
            </div>
          )}

          {/* Extracted Features Table */}
          {result?.features && (
            <div className="bg-white border rounded-2xl p-6 shadow-sm">
              <h3 className="text-lg font-bold text-gray-900 mb-4">Extracted Feature Vector (30 Features)</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b text-left text-gray-500">
                      <th className="py-2 pr-3">Feature</th>
                      <th className="py-2 pr-3">Extracted Value</th>
                      <th className="py-2 pr-3">Training Range</th>
                      <th className="py-2">Compatibility</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.compatibility?.per_feature.map((f) => (
                      <tr key={f.name} className="border-b border-gray-50 hover:bg-gray-50">
                        <td className="py-1.5 pr-3 font-medium text-gray-700 capitalize">{f.name}</td>
                        <td className="py-1.5 pr-3 font-mono">
                          {f.extracted !== null ? f.extracted.toFixed(5) : "—"}
                        </td>
                        <td className="py-1.5 pr-3 text-gray-400 font-mono">
                          [{f.training_range[0].toFixed(4)}, {f.training_range[1].toFixed(4)}]
                        </td>
                        <td className="py-1.5">
                          <VerdictBadge verdict={f.verdict} zScore={f.z_score} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Medical Disclaimer */}
          <div className="bg-amber-50 border border-amber-200 p-4 rounded-xl text-xs text-amber-800 leading-relaxed">
            <div className="flex items-start gap-2">
              <ShieldAlert className="w-4 h-4 shrink-0 mt-0.5" />
              <p>
                <strong>EXPERIMENTAL DISCLAIMER:</strong> This experimental feature extraction workflow has not 
                been clinically validated. Extracted image measurements may not be equivalent to the measurements 
                used to train the underlying model. Results must not be used as a substitute for professional 
                medical diagnosis.
              </p>
            </div>
          </div>
        </div>

        {/* Right: Compatibility & Prediction */}
        <div className="lg:col-span-1">
          <div className="bg-white rounded-2xl border shadow-sm p-6 sticky top-24 min-h-[400px] flex flex-col">
            <h2 className="text-xl font-bold text-gray-900 border-b pb-4 mb-6">Analysis Results</h2>

            {stage === "idle" && !result && (
              <div className="flex-grow flex flex-col items-center justify-center text-gray-400 text-center">
                <Microscope className="w-16 h-16 mb-4 opacity-30" />
                <p className="text-sm">Upload a cell microscopy image to begin experimental analysis.</p>
              </div>
            )}

            {(stage === "uploading" || stage === "processing") && (
              <div className="flex-grow flex flex-col items-center justify-center">
                <Loader2 className="w-12 h-12 text-purple-500 animate-spin mb-4" />
                <p className="text-sm text-gray-500 font-medium">
                  {stage === "uploading" ? "Uploading image..." : "Analyzing image..."}
                </p>
                <p className="text-xs text-gray-400 mt-1">
                  Preprocessing → Segmentation → Feature Extraction → Validation
                </p>
              </div>
            )}

            {stage === "done" && result && (
              <AnimatePresence>
                <motion.div 
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex-grow flex flex-col"
                >
                  {/* Pipeline Status */}
                  <div className={`p-4 rounded-xl mb-4 border ${
                    result.success ? "bg-emerald-50 border-emerald-200" : "bg-red-50 border-red-200"
                  }`}>
                    <div className="flex items-center gap-2 mb-1">
                      {result.success ? (
                        <CheckCircle2 className="w-5 h-5 text-emerald-600" />
                      ) : (
                        <XCircle className="w-5 h-5 text-red-600" />
                      )}
                      <span className={`font-bold text-sm ${result.success ? "text-emerald-800" : "text-red-800"}`}>
                        {result.success ? "Features Extracted" : "Extraction Failed"}
                      </span>
                    </div>
                    <p className="text-xs text-gray-600 leading-relaxed">{result.message}</p>
                  </div>

                  {/* Compatibility Report */}
                  {result.compatibility && (
                    <div className="mb-4">
                      <h3 className="text-sm font-bold text-gray-700 mb-2">Distribution Compatibility</h3>
                      <div className={`p-3 rounded-lg border ${
                        result.compatibility.overall_verdict === "Compatible" 
                          ? "bg-emerald-50 border-emerald-200"
                          : result.compatibility.overall_verdict === "Potentially Incompatible"
                            ? "bg-amber-50 border-amber-200"
                            : "bg-red-50 border-red-200"
                      }`}>
                        <div className="flex items-center gap-2 mb-2">
                          {result.compatibility.overall_verdict === "Compatible" ? (
                            <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                          ) : result.compatibility.overall_verdict === "Potentially Incompatible" ? (
                            <AlertTriangle className="w-4 h-4 text-amber-600" />
                          ) : (
                            <XCircle className="w-4 h-4 text-red-600" />
                          )}
                          <span className="font-bold text-xs uppercase tracking-wider">
                            {result.compatibility.overall_verdict}
                          </span>
                        </div>
                        <p className="text-[11px] text-gray-600 leading-relaxed">{result.compatibility.message}</p>
                      </div>
                      <div className="grid grid-cols-3 gap-2 mt-3 text-center">
                        <div className="bg-emerald-50 border border-emerald-200 rounded-lg py-2">
                          <p className="text-lg font-bold text-emerald-700">{result.compatibility.num_compatible}</p>
                          <p className="text-[10px] text-emerald-600 font-medium">Compatible</p>
                        </div>
                        <div className="bg-amber-50 border border-amber-200 rounded-lg py-2">
                          <p className="text-lg font-bold text-amber-700">{result.compatibility.num_marginal}</p>
                          <p className="text-[10px] text-amber-600 font-medium">Marginal</p>
                        </div>
                        <div className="bg-red-50 border border-red-200 rounded-lg py-2">
                          <p className="text-lg font-bold text-red-700">{result.compatibility.num_incompatible}</p>
                          <p className="text-[10px] text-red-600 font-medium">Incompatible</p>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Prediction Result */}
                  {result.prediction && !result.prediction_blocked && (
                    <div className="mb-4">
                      <h3 className="text-sm font-bold text-gray-700 mb-2">Experimental Prediction</h3>
                      <div className={`p-4 rounded-xl text-center border ${
                        result.prediction.prediction_code === "M" 
                          ? "bg-red-50 border-red-200" 
                          : "bg-emerald-50 border-emerald-200"
                      }`}>
                        <p className={`text-2xl font-black tracking-widest ${
                          result.prediction.prediction_code === "M" ? "text-red-700" : "text-emerald-700"
                        }`}>
                          {result.prediction.prediction.toUpperCase()}
                        </p>
                        <p className="text-xs text-gray-500 mt-1">
                          Confidence: {(result.prediction.confidence * 100).toFixed(1)}%
                        </p>
                      </div>
                      <div className="mt-3 space-y-2">
                        {Object.entries(result.prediction.probabilities).map(([cls, prob]) => (
                          <div key={cls}>
                            <div className="flex justify-between text-xs mb-0.5">
                              <span className="font-medium capitalize">{cls}</span>
                              <span className="font-bold">{(prob * 100).toFixed(1)}%</span>
                            </div>
                            <div className="w-full bg-gray-100 rounded-full h-1.5">
                              <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${prob * 100}%` }}
                                transition={{ duration: 0.8 }}
                                className={`h-1.5 rounded-full ${cls === "malignant" ? "bg-red-500" : "bg-emerald-500"}`}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                      <p className="text-[10px] text-gray-400 mt-2 italic">
                        ⚠ Experimental result from image-extracted features. Not clinically validated.
                      </p>
                    </div>
                  )}

                  {result.prediction_blocked && result.block_reason && (
                    <div className="p-3 bg-gray-50 border border-gray-200 rounded-lg mb-4">
                      <div className="flex items-center gap-2 mb-1">
                        <ShieldAlert className="w-4 h-4 text-gray-500" />
                        <span className="text-xs font-bold text-gray-700">Prediction Blocked</span>
                      </div>
                      <p className="text-[11px] text-gray-500 leading-relaxed">{result.block_reason}</p>
                    </div>
                  )}
                </motion.div>
              </AnimatePresence>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function VerdictBadge({ verdict, zScore }: { verdict: string; zScore: number | null }) {
  const config: Record<string, { bg: string; text: string; icon: React.ReactNode }> = {
    "Compatible": { bg: "bg-emerald-100", text: "text-emerald-700", icon: <CheckCircle2 className="w-3 h-3" /> },
    "Marginal": { bg: "bg-amber-100", text: "text-amber-700", icon: <AlertTriangle className="w-3 h-3" /> },
    "Incompatible": { bg: "bg-red-100", text: "text-red-700", icon: <XCircle className="w-3 h-3" /> },
    "Missing": { bg: "bg-gray-100", text: "text-gray-600", icon: <XCircle className="w-3 h-3" /> },
  };
  const c = config[verdict] || config["Missing"];
  
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold ${c.bg} ${c.text}`}>
      {c.icon}
      {verdict}
      {zScore !== null && <span className="opacity-60">(z={zScore})</span>}
    </span>
  );
}
