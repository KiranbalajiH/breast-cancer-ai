import Link from "next/link";
import { Activity, ShieldCheck, PieChart, FlaskConical, ChevronRight } from "lucide-react";

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
            Advanced Breast Cancer <br className="hidden lg:block" />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-medical-600 to-medical-400">Diagnostic Intelligence</span>
          </h1>
          <p className="text-xl text-slate-600 mb-10 leading-relaxed max-w-2xl text-balance font-medium">
            Enhance diagnostic capabilities with our machine learning models trained on robust clinical data. Obtain rapid, data-driven insights through intelligent analysis of medical imaging and cellular measurements.
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
              href="/analytics" 
              className="px-8 py-4 w-full sm:w-auto rounded-xl bg-white text-slate-700 font-bold text-lg shadow-sm border border-slate-200 hover:border-slate-300 hover:bg-slate-50 transition-all duration-300 flex items-center justify-center"
            >
              Explore Model Data
            </Link>
          </div>
        </div>

        {/* Right Feature Grid */}
        <div className="flex-1 grid grid-cols-1 sm:grid-cols-2 gap-4 w-full relative">
            {/* Subtle glow effect behind cards */}
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[120%] h-[120%] bg-medical-50 opacity-50 blur-3xl rounded-full -z-10" />
            
            <FeatureCard 
              icon={<Activity className="w-7 h-7 text-medical-600" />}
              title="Image Analysis"
              description="Upload ultrasound imagery for automated feature extraction using MobileNetV2."
            />
            <FeatureCard 
              icon={<FlaskConical className="w-7 h-7 text-medical-600" />}
              title="Cellular Data"
              description="Process 30 distinct numeric lab features through an optimized SVM pipeline."
            />
            <FeatureCard 
              icon={<PieChart className="w-7 h-7 text-medical-600" />}
              title="Probability Confidence"
              description="Receive precise probability distributions rather than simple binary outcomes."
            />
            <FeatureCard 
              icon={<ShieldCheck className="w-7 h-7 text-medical-600" />}
              title="Transparent Analytics"
              description="Full visibility into model performance and validation metrics."
            />
        </div>
      </div>
    </div>
  );
}

function FeatureCard({ icon, title, description }: { icon: React.ReactNode, title: string, description: string }) {
  return (
    <div className="bg-slate-50 rounded-2xl p-8 border border-slate-200 shadow-sm hover:shadow-md hover:border-medical-200 transition-all duration-300 flex flex-col justify-between h-full group">
      <div>
        <div className="bg-white border border-slate-100 w-14 h-14 rounded-2xl flex items-center justify-center mb-6 shadow-sm group-hover:scale-110 group-hover:bg-medical-50 transition-all duration-300">
          {icon}
        </div>
        <h3 className="text-xl font-extrabold text-slate-900 mb-3">{title}</h3>
      </div>
      <p className="text-base text-slate-600 leading-relaxed font-medium">
        {description}
      </p>
    </div>
  );
}
