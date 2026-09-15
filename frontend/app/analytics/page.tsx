/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";

import { Activity } from "lucide-react";

export default function Analytics() {
  return (
    <div className="max-w-[1400px] mx-auto px-4 sm:px-6 lg:px-8 py-8 lg:py-12">
      <div className="mb-8 border-b border-slate-200 pb-6">
        <h1 className="text-4xl font-extrabold text-slate-900 tracking-tight mb-2">Ultrasound AI Model Analytics</h1>
        <p className="text-slate-500 font-medium text-base">
          Performance audit for production and research breast ultrasound image classification models.
        </p>
      </div>

      {/* Production Model Overview: V5-B */}
      <div className="bg-white p-8 rounded-2xl border border-slate-200 shadow-sm mb-8">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-6 border-b border-slate-100 pb-4">
          <div>
            <span className="text-xs font-bold uppercase tracking-wider text-medical-600 bg-medical-50 px-3 py-1 rounded-full border border-medical-100">
              Active Production Model
            </span>
            <h2 className="text-2xl font-extrabold text-slate-900 mt-2">V5-B MobileNetV2 (Aspect-Ratio Letterboxed)</h2>
          </div>
          <div className="mt-2 sm:mt-0 text-sm font-bold text-slate-500">
            SHA256: <code className="text-xs bg-slate-100 px-2 py-1 rounded font-mono text-slate-800">93b3c106...26b32e03c</code>
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <MetricCard title="Historical Accuracy" value="80.3%" subtitle="117-Image Test Set" />
          <MetricCard title="Historical Malignant Recall" value="87.5%" subtitle="Sensitivity (4 FN)" />
          <MetricCard title="Grouped Accuracy" value="70.1%" subtitle="107-Image Test Set" />
          <MetricCard title="Grouped Malignant Recall" value="83.9%" subtitle="Zero Cluster Overlap" />
        </div>
      </div>

      {/* Research Model Overview: V14 Ensemble */}
      <div className="bg-white p-8 rounded-2xl border border-slate-200 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-6 border-b border-slate-100 pb-4">
          <div>
            <span className="text-xs font-bold uppercase tracking-wider text-purple-600 bg-purple-50 px-3 py-1 rounded-full border border-purple-100">
              Research Champion (Opt-in)
            </span>
            <h2 className="text-2xl font-extrabold text-slate-900 mt-2">V14 Ensemble (V11-B MobileNetV2 + V13-B0 EfficientNetB0)</h2>
          </div>
          <div className="mt-2 sm:mt-0 text-sm font-bold text-slate-500">
            Decision Threshold: <code className="text-xs bg-slate-100 px-2 py-1 rounded font-mono text-slate-800">t* = 0.58</code>
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <MetricCard title="Historical Accuracy" value="86.3%" subtitle="117-Image Test Set" />
          <MetricCard title="Historical Malignant Recall" value="84.4%" subtitle="Sensitivity (5 FN)" />
          <MetricCard title="Grouped Accuracy" value="80.4%" subtitle="107-Image Test Set" />
          <MetricCard title="Grouped Malignant Precision" value="82.8%" subtitle="High Specificity" />
        </div>
      </div>
    </div>
  );
}

function MetricCard({ title, value, subtitle }: { title: string, value: string, subtitle?: string }) {
  return (
    <div className="bg-slate-50 p-5 rounded-2xl border border-slate-200 shadow-sm flex flex-col items-center justify-center text-center">
      <span className="text-xs font-bold text-slate-500 mb-1 uppercase tracking-wider">{title}</span>
      <span className="text-3xl font-black text-slate-900 tracking-tight my-1">{value}</span>
      {subtitle && <span className="text-[11px] font-medium text-slate-400">{subtitle}</span>}
    </div>
  );
}
