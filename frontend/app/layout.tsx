import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Navbar } from "@/components/layout/navbar";

const inter = Inter({ 
  subsets: ["latin"],
  variable: '--font-inter',
});

export const metadata: Metadata = {
  title: "OncoAI - Breast Cancer Classification",
  description: "Premium AI-Powered Breast Cancer Classification System",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.className} min-h-screen bg-slate-50 text-slate-900 flex flex-col selection:bg-medical-200 selection:text-medical-900`}>
        <Navbar />
        <main className="flex-grow">
          {children}
        </main>
        <footer className="w-full bg-white/50 backdrop-blur-sm border-t border-slate-200 py-8 mt-auto">
          <div className="max-w-[1400px] mx-auto px-4 text-center text-sm font-medium text-slate-500">
            &copy; {new Date().getFullYear()} OncoAI. For educational and research purposes only.
          </div>
        </footer>
      </body>
    </html>
  );
}
