import { Database, Layers, ShieldAlert, CheckCircle2 } from "lucide-react";

export default function About() {
  return (
    <div className="max-w-[1400px] mx-auto px-4 sm:px-6 lg:px-8 py-12 lg:py-16">
      <div className="mb-12 border-b border-slate-200 pb-8">
        <h1 className="text-4xl lg:text-5xl font-extrabold text-slate-900 tracking-tight mb-4">About OncoAI</h1>
        <p className="text-lg text-slate-500 max-w-3xl font-medium leading-relaxed">
          OncoAI is a production-grade deep learning decision support application designed to classify breast ultrasound scans. The system provides rapid, data-driven insights utilizing fine-tuned computer vision architectures and visual Grad-CAM explanations to support diagnostic evaluation.
        </p>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-12">
        {/* Left Column */}
        <div className="space-y-8 lg:space-y-12">
          
          <section className="bg-white p-8 rounded-2xl border border-slate-200 shadow-sm">
            <h2 className="text-2xl font-bold text-slate-900 mb-6 flex items-center">
              <div className="bg-medical-50 w-10 h-10 rounded-xl flex items-center justify-center mr-4 text-medical-600">
                <Database className="w-5 h-5" />
              </div>
              The Image Dataset
            </h2>
            <div className="space-y-6 text-slate-600 leading-relaxed font-medium">
              <div className="flex gap-4">
                <CheckCircle2 className="w-6 h-6 text-teal-500 shrink-0" />
                <div>
                  <strong className="text-slate-900 block mb-1">Breast Ultrasound Images (BUSI)</strong>
                  Trained on cleaned breast ultrasound scans categorized into benign, malignant, and normal diagnostic classes, providing direct visual pattern recognition across unique patient lesion clusters.
                </div>
              </div>
            </div>
          </section>

          <section className="bg-white p-8 rounded-2xl border border-slate-200 shadow-sm">
            <h2 className="text-2xl font-bold text-slate-900 mb-6 flex items-center">
              <div className="bg-medical-50 w-10 h-10 rounded-xl flex items-center justify-center mr-4 text-medical-600">
                <Layers className="w-5 h-5" />
              </div>
              Methodology
            </h2>
            <div className="space-y-6 text-slate-600 leading-relaxed font-medium">
              <div className="flex gap-4">
                <CheckCircle2 className="w-6 h-6 text-medical-500 shrink-0" />
                <div>
                  <strong className="text-slate-900 block mb-1">Production V5-B Model</strong>
                  A MobileNetV2 architecture with aspect-ratio letterbox preprocessing ($224\times224\times3$), optimized for high sensitivity (87.50% historical recall) while preserving clinical safety.
                </div>
              </div>
              <div className="flex gap-4">
                <CheckCircle2 className="w-6 h-6 text-medical-500 shrink-0" />
                <div>
                  <strong className="text-slate-900 block mb-1">V14 Research Ensemble</strong>
                  A 50/50 probability ensemble combining MobileNetV2 (V11-B) and EfficientNetB0 (V13-B0) operating at threshold $t^*=0.58$ for advanced research evaluations.
                </div>
              </div>
            </div>
          </section>

        </div>

        {/* Right Column */}
        <div className="space-y-8 lg:space-y-12">
          
          <section className="bg-slate-50 border border-slate-200 rounded-2xl p-8 relative overflow-hidden h-full">
            <div className="absolute top-0 left-0 w-1.5 h-full bg-amber-400"></div>
            <h2 className="text-2xl font-bold text-slate-900 mb-8 flex items-center">
              <div className="bg-amber-100 w-10 h-10 rounded-xl flex items-center justify-center mr-4 text-amber-600">
                <ShieldAlert className="w-5 h-5" />
              </div>
              Important Limitations & Disclaimer
            </h2>
            <ul className="space-y-5 text-slate-700 text-base font-medium">
              <li className="flex items-start bg-white p-4 rounded-xl border border-slate-100 shadow-sm">
                <span className="mr-3 text-amber-500 text-lg leading-none">•</span> 
                <span>This system is trained strictly on historical breast ultrasound imaging datasets.</span>
              </li>
              <li className="flex items-start bg-white p-4 rounded-xl border border-slate-100 shadow-sm">
                <span className="mr-3 text-amber-500 text-lg leading-none">•</span> 
                <span>Results may not necessarily generalize to all clinical populations or imaging devices.</span>
              </li>
              <li className="flex items-start bg-white p-4 rounded-xl border border-slate-100 shadow-sm">
                <span className="mr-3 text-amber-500 text-lg leading-none">•</span> 
                <span>The model is <strong>not validated</strong> for real-world clinical deployment.</span>
              </li>
              <li className="flex items-start bg-white p-4 rounded-xl border border-slate-100 shadow-sm">
                <span className="mr-3 text-amber-500 text-lg leading-none">•</span> 
                <span>A high confidence score or Grad-CAM heatmap does <strong>not</strong> equal clinical certainty.</span>
              </li>
              <li className="flex items-start bg-amber-50 p-4 rounded-xl border border-amber-200 shadow-sm">
                <span className="mr-3 text-amber-600 text-lg leading-none mt-1"><ShieldAlert className="w-5 h-5" /></span> 
                <span className="text-amber-900 font-bold">This application does NOT replace professional healthcare evaluation, diagnosis, or treatment. It is for educational and research purposes only.</span>
              </li>
            </ul>
          </section>

        </div>
      </div>
    </div>
  );
}
