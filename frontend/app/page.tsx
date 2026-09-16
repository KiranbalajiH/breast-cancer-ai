import Link from "next/link";
import { Activity, ChevronRight } from "lucide-react";

export default function Home() {
  return (
    <div className="flex flex-col min-h-[calc(100vh-80px)] bg-white overflow-hidden">
      <div className="max-w-[1400px] mx-auto px-4 sm:px-6 lg:px-8 w-full pt-16 pb-12 lg:pt-20 lg:pb-16 flex flex-col lg:flex-row items-center gap-16 relative z-10">
        
        {/* Left Hero Content */}
        <div className="flex-1 text-left relative">
          <div className="inline-flex items-center px-4 py-1.5 rounded-full bg-medical-50 border border-medical-100 text-medical-700 text-sm font-bold mb-6 shadow-sm">
            <span className="flex w-2.5 h-2.5 rounded-full bg-medical-500 mr-2.5 animate-pulse"></span>
            Research & Diagnostic Support Tool
          </div>
          <h1 className="text-5xl lg:text-7xl font-extrabold text-slate-900 tracking-tight mb-8 leading-[1.1]">
            Advanced Breast Ultrasound <br className="hidden lg:block" />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-medical-600 to-medical-400">Diagnostic Intelligence</span>
          </h1>
          <p className="text-xl text-slate-600 mb-10 leading-relaxed max-w-2xl text-balance font-medium">
            Enhance diagnostic capabilities with our deep learning vision models fine-tuned on clinical ultrasound imaging. Obtain rapid, data-driven insights through intelligent visual analysis and Grad-CAM explainability.
          </p>
          <div className="flex flex-col sm:flex-row items-center gap-4">
            <Link 
              href="/predict" 
              className="px-8 py-4 w-full sm:w-auto rounded-xl bg-slate-900 text-white font-bold text-lg shadow-md hover:bg-slate-800 hover:shadow-lg transition-all duration-300 flex items-center justify-center"
            >
              Start Analysis
              <ChevronRight className="w-5 h-5 ml-2" />
            </Link>
            <Link 
              href="/about" 
              className="px-8 py-4 w-full sm:w-auto rounded-xl bg-white text-slate-700 font-bold text-lg shadow-sm border border-slate-200 hover:border-slate-300 hover:bg-slate-50 transition-all duration-300 flex items-center justify-center"
            >
              Learn More
            </Link>
          </div>
        </div>

        {/* Right Medical Ultrasound Visual Showcase */}
        <div className="flex-1 w-full relative">
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[120%] h-[120%] bg-medical-100 opacity-40 blur-3xl rounded-full -z-10" />
          <div className="relative rounded-3xl overflow-hidden border border-slate-800 shadow-2xl bg-slate-950 p-2 group">
            <div className="relative rounded-2xl overflow-hidden aspect-[4/3]">
              <img 
                src="/hero_ultrasound.jpg" 
                alt="Medical Ultrasound Diagnostic Interface"
                className="w-full h-full object-cover object-center transform group-hover:scale-105 transition-transform duration-700"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-slate-950/80 via-transparent to-slate-950/20" />
              
              {/* Overlay Medical Badges */}
              <div className="absolute top-4 left-4 bg-slate-900/90 backdrop-blur-md px-3.5 py-1.5 rounded-xl border border-slate-700 text-xs text-medical-400 font-bold flex items-center shadow-lg">
                <Activity className="w-4 h-4 mr-2 text-medical-400 animate-pulse" />
                ACUSON S2000 Ultrasound AI Feed
              </div>

              <div className="absolute bottom-4 left-4 right-4 bg-slate-900/90 backdrop-blur-md p-4 rounded-xl border border-slate-800 shadow-xl flex items-center justify-between">
                <div>
                  <p className="text-xs text-slate-400 font-medium">Lesion Analysis & Grad-CAM</p>
                  <p className="text-sm font-bold text-white flex items-center">
                    BI-RADS Lesion Detection • V5-B & V14 Ensemble
                  </p>
                </div>
                <div className="bg-medical-500/20 text-medical-300 border border-medical-500/30 px-3 py-1 rounded-lg text-xs font-bold">
                  87.50% Sensitivity
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

