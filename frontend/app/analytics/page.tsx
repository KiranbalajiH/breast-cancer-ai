/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";

import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { getModelMetadata, getModelComparison } from "@/lib/api";
import { Activity, RefreshCcw } from "lucide-react";

const ComparisonChart = dynamic(() => import('@/components/analytics/Charts').then(mod => mod.ComparisonChart), { 
  ssr: false, 
  loading: () => <div className="w-full h-full flex items-center justify-center text-slate-400 text-sm font-medium">Loading Chart...</div> 
});

const FeatureChart = dynamic(() => import('@/components/analytics/Charts').then(mod => mod.FeatureChart), { 
  ssr: false,
  loading: () => <div className="w-full h-full flex items-center justify-center text-slate-400 text-sm font-medium">Loading Chart...</div> 
});

export default function Analytics() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [metadata, setMetadata] = useState<any>(null);
  const [comparison, setComparison] = useState<Record<string, Record<string, number>> | null>(null);
  const [metricFilter, setMetricFilter] = useState("f1_score");

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [meta, comp] = await Promise.all([
        getModelMetadata(),
        getModelComparison()
      ]);
      setMetadata(meta);
      setComparison(comp);
    } catch {
      setError("Unable to connect to the AI model backend. Check that the FastAPI server is running.");
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh]">
        <Activity className="w-12 h-12 text-medical-500 animate-pulse mb-4" />
        <p className="text-slate-500 font-bold">Loading Analytics Dashboard...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] px-4">
        <div className="bg-red-50 p-6 rounded-2xl border border-red-100 max-w-md text-center shadow-sm">
          <p className="text-red-700 font-medium mb-4">{error}</p>
          <button 
            onClick={fetchData}
            className="flex items-center mx-auto px-5 py-2.5 bg-red-600 text-white font-bold rounded-xl hover:bg-red-700 transition shadow-sm"
          >
            <RefreshCcw className="w-4 h-4 mr-2" /> Retry Connection
          </button>
        </div>
      </div>
    );
  }

  const comparisonData = comparison ? Object.keys(comparison).map(modelName => ({
    name: modelName,
    accuracy: comparison[modelName].accuracy * 100,
    precision: comparison[modelName].precision * 100,
    recall: comparison[modelName].recall * 100,
    specificity: comparison[modelName].specificity * 100,
    f1_score: comparison[modelName].f1_score * 100,
    roc_auc: comparison[modelName].roc_auc * 100,
  })) : [];

  const cm = (metadata?.metrics as any)?.confusion_matrix;

  return (
    <div className="max-w-[1400px] mx-auto px-4 sm:px-6 lg:px-8 py-8 lg:py-12">
      <div className="mb-8">
        <h1 className="text-4xl font-extrabold text-slate-900 tracking-tight mb-2">Model Analytics</h1>
        <p className="text-slate-500 font-medium text-base">
          Currently deployed tabular model: <span className="font-bold text-medical-700 bg-medical-50 px-2 py-0.5 rounded-lg border border-medical-100">{metadata?.model_name} (v{metadata?.model_version})</span>
        </p>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-8">
        <MetricCard title="Accuracy" value={(metadata?.metrics?.accuracy * 100).toFixed(1) + "%"} />
        <MetricCard title="Precision" value={(metadata?.metrics?.precision * 100).toFixed(1) + "%"} />
        <MetricCard title="Recall" value={(metadata?.metrics?.recall * 100).toFixed(1) + "%"} />
        <MetricCard title="Specificity" value={(metadata?.metrics?.specificity * 100).toFixed(1) + "%"} />
        <MetricCard title="F1 Score" value={(metadata?.metrics?.f1_score * 100).toFixed(1) + "%"} />
        <MetricCard title="ROC-AUC" value={(metadata?.metrics?.roc_auc * 100).toFixed(1) + "%"} />
      </div>

      {/* Middle Row: Comparison Chart & Confusion Matrix */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm flex flex-col">
          <div className="flex justify-between items-center mb-6 border-b border-slate-100 pb-4">
            <h2 className="text-xl font-bold text-slate-900">Model Comparison</h2>
            <select 
              value={metricFilter}
              onChange={(e) => setMetricFilter(e.target.value)}
              className="text-sm font-bold border-slate-200 rounded-lg shadow-sm focus:ring-medical-500 focus:border-medical-500 px-4 py-2 bg-slate-50 text-slate-700"
            >
              <option value="accuracy">Accuracy</option>
              <option value="f1_score">F1 Score</option>
              <option value="recall">Recall</option>
              <option value="precision">Precision</option>
              <option value="roc_auc">ROC-AUC</option>
            </select>
          </div>
          <div className="h-80 flex-grow">
            <ComparisonChart data={comparisonData} metricFilter={metricFilter} />
          </div>
        </div>

        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm flex flex-col">
          <div className="border-b border-slate-100 pb-4 mb-6">
            <h2 className="text-xl font-bold text-slate-900 mb-1">Out-of-Fold Confusion Matrix</h2>
            <p className="text-xs font-medium text-slate-500">Dataset Size: 569 samples ({metadata?.model_name})</p>
          </div>
          {cm ? (
            <div className="grid grid-cols-2 gap-4 h-80 flex-grow">
              <div className="bg-teal-50 rounded-2xl p-6 flex flex-col justify-center items-center border border-teal-100 shadow-sm">
                <span className="text-teal-900 font-black text-5xl mb-2 tracking-tight">{cm.tn}</span>
                <span className="text-teal-700 text-sm font-bold uppercase tracking-wider">True Negative</span>
              </div>
              <div className="bg-red-50 rounded-2xl p-6 flex flex-col justify-center items-center border border-red-100 shadow-sm">
                <span className="text-red-900 font-black text-5xl mb-2 tracking-tight">{cm.fp}</span>
                <span className="text-red-700 text-sm font-bold uppercase tracking-wider">False Positive</span>
              </div>
              <div className="bg-red-50 rounded-2xl p-6 flex flex-col justify-center items-center border border-red-100 shadow-sm">
                <span className="text-red-900 font-black text-5xl mb-2 tracking-tight">{cm.fn}</span>
                <span className="text-red-700 text-sm font-bold uppercase tracking-wider">False Negative</span>
              </div>
              <div className="bg-teal-50 rounded-2xl p-6 flex flex-col justify-center items-center border border-teal-100 shadow-sm">
                <span className="text-teal-900 font-black text-5xl mb-2 tracking-tight">{cm.tp}</span>
                <span className="text-teal-700 text-sm font-bold uppercase tracking-wider">True Positive</span>
              </div>
            </div>
          ) : (
            <div className="h-80 flex-grow flex items-center justify-center text-slate-400 bg-slate-50 rounded-xl border border-slate-100 font-medium">
              Confusion Matrix not available.
            </div>
          )}
        </div>
      </div>

      {/* Bottom Row: Feature Importance */}
      {(metadata?.feature_importance as any[]) && (
        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
          <div className="border-b border-slate-100 pb-4 mb-6">
            <h2 className="text-xl font-bold text-slate-900 mb-1">Feature Importance (Top 10)</h2>
            <p className="text-sm font-medium text-slate-500">Represents statistical relationships (permutation importance), not proven medical causation.</p>
          </div>
          <div className="h-80 w-full">
            <FeatureChart data={(metadata.feature_importance as any[]).slice(0, 10)} />
          </div>
        </div>
      )}
    </div>
  );
}

function MetricCard({ title, value }: { title: string, value: string }) {
  return (
    <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm flex flex-col items-center justify-center text-center group hover:border-medical-200 hover:shadow-md transition-all duration-300">
      <span className="text-xs font-bold text-slate-500 mb-2 uppercase tracking-wider">{title}</span>
      <span className="text-4xl font-black text-slate-900 tracking-tight group-hover:text-medical-700 transition-colors">{value}</span>
    </div>
  );
}
