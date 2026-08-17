"use client";

import { Activity } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { getImageModelStatus } from "@/lib/api";

export function Navbar() {
  const [status, setStatus] = useState<"LOADING" | "READY" | "UNAVAILABLE" | "OFFLINE">("LOADING");
  const pathname = usePathname();

  useEffect(() => {
    let mounted = true;
    const fetchStatus = async () => {
      try {
        const res = await getImageModelStatus();
        if (mounted) {
          if (res.model_loaded && res.status === "ready") {
            setStatus("READY");
          } else {
            setStatus("UNAVAILABLE");
          }
        }
      } catch (error) {
        if (mounted) {
          if (error instanceof TypeError) {
            setStatus("OFFLINE");
          } else {
            setStatus("UNAVAILABLE");
          }
        }
      }
    };
    fetchStatus();
    const interval = setInterval(fetchStatus, 60000); // reduced frequency to 60s
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  const navLinks = [
    { name: "Home", href: "/" },
    { name: "Prediction", href: "/predict" },
    { name: "Analytics", href: "/analytics" },
    { name: "About", href: "/about" },
  ];

  return (
    <nav className="w-full glass sticky top-0 z-50 border-b border-slate-200/60 shadow-sm">
      <div className="max-w-[1400px] mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-20 items-center">
          <div className="flex items-center space-x-12">
            <Link href="/" className="flex items-center space-x-3 group">
              <div className="bg-medical-500/10 p-2 rounded-xl group-hover:bg-medical-500/20 transition-colors">
                <Activity className="h-8 w-8 text-medical-600" />
              </div>
              <span className="font-extrabold text-2xl tracking-tight text-slate-900">OncoAI</span>
            </Link>
            <div className="hidden md:flex space-x-2 text-base font-semibold">
              {navLinks.map((link) => {
                const isActive = pathname === link.href;
                return (
                  <Link 
                    key={link.name} 
                    href={link.href} 
                    className={`px-5 py-2.5 rounded-xl transition-all duration-200 ${
                      isActive 
                        ? "bg-medical-50 text-medical-800 shadow-sm border border-medical-100" 
                        : "text-slate-600 hover:text-slate-900 hover:bg-slate-100 border border-transparent"
                    }`}
                  >
                    {link.name}
                  </Link>
                );
              })}
            </div>
          </div>
          <div className="flex items-center">
            <div className="flex items-center space-x-2.5 bg-white px-4 py-2 rounded-full border border-slate-200 shadow-sm">
              <div className={`w-2.5 h-2.5 rounded-full ${
                status === "READY" ? "bg-medical-500 shadow-[0_0_8px_rgba(20,184,166,0.6)]" :
                status === "LOADING" ? "bg-amber-400 animate-pulse" :
                status === "UNAVAILABLE" ? "bg-amber-500 shadow-[0_0_8px_rgba(245,158,11,0.6)] animate-pulse" :
                "bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.6)]"
              }`} />
              <span className="text-sm font-bold text-slate-700 tracking-wide">
                {status === "LOADING" ? "Connecting..." : 
                 status === "READY" ? "System Online" : 
                 status === "UNAVAILABLE" ? "Model Unavailable" : "System Offline"}
              </span>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
}
