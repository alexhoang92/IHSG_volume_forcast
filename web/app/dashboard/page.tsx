"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { apiGet, apiPost } from "@/lib/api";
import UploadTab from "@/components/UploadTab";
import RunTab from "@/components/RunTab";
import BacktestTab from "@/components/BacktestTab";

type Tab = "upload" | "run" | "backtest";

const TABS: { id: Tab; label: string }[] = [
  { id: "upload", label: "1. Upload Data" },
  { id: "run", label: "2. Run & Results" },
  { id: "backtest", label: "3. Backtest" },
];

export default function DashboardPage() {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState<Tab>("upload");
  const [uploadedFiles, setUploadedFiles] = useState<Set<string>>(new Set());

  // Check which files are already on the server (e.g. from a previous session)
  useEffect(() => {
    apiGet<Record<string, boolean>>("/upload/status")
      .then((status) => {
        const uploaded = new Set<string>(
          Object.entries(status)
            .filter(([, v]) => v)
            .map(([k]) => k),
        );
        setUploadedFiles(uploaded);
      })
      .catch(() => {});
  }, []);

  function handleUploadSuccess(type: string) {
    setUploadedFiles((prev) => new Set(Array.from(prev).concat(type)));
  }

  async function handleLogout() {
    await apiPost("/auth/logout").catch(() => {});
    router.push("/login");
  }

  const volumeReady = uploadedFiles.has("volume");
  const scenariosReady = uploadedFiles.has("scenarios");

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 px-6 py-4 flex items-center justify-between">
        <div>
          <h1 className="text-lg font-bold text-slate-800">IHSG Volume Forecast</h1>
          <p className="text-xs text-slate-500">Indonesia Stock Exchange — Econometric Forecasting</p>
        </div>
        <div className="flex items-center gap-4">
          <div className="hidden sm:flex items-center gap-2 text-xs text-slate-500">
            {volumeReady && (
              <span className="bg-green-100 text-green-700 px-2 py-0.5 rounded-full">Volume ✓</span>
            )}
            {scenariosReady && (
              <span className="bg-green-100 text-green-700 px-2 py-0.5 rounded-full">Scenarios ✓</span>
            )}
          </div>
          <button
            onClick={handleLogout}
            className="text-sm text-slate-500 hover:text-slate-800 transition-colors"
          >
            Sign out
          </button>
        </div>
      </header>

      {/* Tabs */}
      <div className="bg-white border-b border-slate-200 px-6">
        <nav className="flex gap-1">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                activeTab === tab.id
                  ? "border-blue-600 text-blue-600"
                  : "border-transparent text-slate-500 hover:text-slate-800"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Content */}
      <main className="max-w-5xl mx-auto px-6 py-8">
        {activeTab === "upload" && (
          <UploadTab onUploadSuccess={handleUploadSuccess} />
        )}
        {activeTab === "run" && (
          <RunTab volumeReady={volumeReady} scenariosReady={scenariosReady} />
        )}
        {activeTab === "backtest" && <BacktestTab />}
      </main>
    </div>
  );
}
