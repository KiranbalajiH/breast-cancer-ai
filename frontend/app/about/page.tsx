import { Info, Database, Layers, ShieldAlert, CheckCircle2 } from "lucide-react";

export default function About() {
  return (
    <div className="max-w-[1400px] mx-auto px-4 sm:px-6 lg:px-8 py-12 lg:py-16">
      <div className="mb-12 border-b border-slate-200 pb-8">
        <h1 className="text-4xl lg:text-5xl font-extrabold text-slate-900 tracking-tight mb-4">About OncoAI</h1>
        <p className="text-lg text-slate-500 max-w-3xl font-medium leading-relaxed">
          OncoAI is a modern machine learning application designed to classify breast cancer tumors based on cellular diagnostic measurements and medical ultrasound imaging. The system provides rapid, data-driven insights utilizing highly optimized AI models to support research and preliminary diagnostic evaluation.
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
              The Datasets
            </h2>
            <div className="space-y-6 text-slate-600 leading-relaxed font-medium">
              <div className="flex gap-4">
                <CheckCircle2 className="w-6 h-6 text-teal-500 shrink-0" />
                <div>
                  <strong className="text-slate-900 block mb-1">Tabular Data</strong>
                  Uses the established Breast Cancer Wisconsin Diagnostic dataset. It contains 569 instances of diagnostic measurements, representing features computed from digitized images of a fine needle aspirate (FNA) of a breast mass.
                </div>
              </div>
              <div className="flex gap-4">
                <CheckCircle2 className="w-6 h-6 text-teal-500 shrink-0" />
                <div>
                  <strong className="text-slate-900 block mb-1">Image Data</strong>
                  Uses the Breast Ultrasound Images Dataset (BUSI), containing images categorized into normal, benign, and malignant classes, allowing for direct visual pattern recognition.
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
                  <strong className="text-slate-900 block mb-1">Tabular SVM Analysis</strong>
                  The selected model is a Support Vector Machine (SVM) with an RBF kernel and SMOTE oversampling, chosen for its superior ability to maximize recall (minimizing false negatives) while maintaining a high F1 score and ROC-AUC.
                </div>
              </div>
              <div className="flex gap-4">
                <CheckCircle2 className="w-6 h-6 text-medical-500 shrink-0" />
                <div>
                  <strong className="text-slate-900 block mb-1">Image Classification</strong>
                  A MobileNetV2 Transfer Learning architecture is employed to extract deep visual features, providing a fast and efficient classification suitable for web deployment.
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
                <span>This system is trained using specific historical datasets.</span>
              </li>
              <li className="flex items-start bg-white p-4 rounded-xl border border-slate-100 shadow-sm">
                <span className="mr-3 text-amber-500 text-lg leading-none">•</span> 
                <span>Results may not necessarily generalize to all clinical populations.</span>
              </li>
              <li className="flex items-start bg-white p-4 rounded-xl border border-slate-100 shadow-sm">
                <span className="mr-3 text-amber-500 text-lg leading-none">•</span> 
                <span>The model is <strong>not validated</strong> for real-world clinical deployment.</span>
              </li>
              <li className="flex items-start bg-white p-4 rounded-xl border border-slate-100 shadow-sm">
                <span className="mr-3 text-amber-500 text-lg leading-none">•</span> 
                <span>A high confidence score does <strong>not</strong> equal clinical or medical certainty.</span>
              </li>
              <li className="flex items-start bg-amber-50 p-4 rounded-xl border border-amber-200 shadow-sm">
                <span className="mr-3 text-amber-600 text-lg leading-none mt-1"><Info className="w-5 h-5" /></span> 
                <span className="text-amber-900 font-bold">This application does NOT replace professional healthcare evaluation, diagnosis, or treatment. It is for educational and research purposes only.</span>
              </li>
            </ul>
          </section>

        </div>
      </div>
    </div>
  );
}
