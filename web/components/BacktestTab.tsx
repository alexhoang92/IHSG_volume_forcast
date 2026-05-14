"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { API_URL, apiGet, downloadUrl } from "@/lib/api";

interface BacktestRow {
  cycle: number;
  week_end_date: string;
  trading_days: number;
  actual_log_volume: number;
  forecast_log_volume: number;
  actual_volume: number;
  forecast_volume: number;
  actual_new_accounts: number;
  forecast_new_accounts: number;
  error_log_vol: number;
  error_vol_pct: number;
  in_ci: number;
}

interface CycleMetrics {
  cycle: number;
  mape: string;
  mae_log: string;
  direction_accuracy: string;
  ci_coverage: string;
  mae_accounts: string;
}

function computeMetrics(rows: BacktestRow[]): CycleMetrics[] {
  const cycles = Array.from(new Set(rows.map((r) => r.cycle))).sort();
  return cycles.map((cycle) => {
    const cr = rows.filter((r) => r.cycle === cycle);
    const mape =
      cr.reduce((s, r) => s + Math.abs(r.error_vol_pct ?? 0), 0) / cr.length;
    const mae_log =
      cr.reduce((s, r) => s + Math.abs(r.error_log_vol ?? 0), 0) / cr.length;
    const dir_acc =
      cr.filter((r) => {
        const actual_d = r.actual_log_volume - (r.actual_log_volume || 0);
        const fc_d = r.forecast_log_volume - (r.forecast_log_volume || 0);
        return Math.sign(actual_d) === Math.sign(fc_d);
      }).length / cr.length;
    const ci_cov = cr.filter((r) => r.in_ci === 1).length / cr.length;
    const mae_acc =
      cr.reduce((s, r) => s + Math.abs((r.actual_new_accounts ?? 0) - (r.forecast_new_accounts ?? 0)), 0) /
      cr.length;

    return {
      cycle,
      mape: `${(mape * 100).toFixed(1)}%`,
      mae_log: mae_log.toFixed(3),
      direction_accuracy: `${(dir_acc * 100).toFixed(0)}%`,
      ci_coverage: `${(ci_cov * 100).toFixed(0)}%`,
      mae_accounts: Math.round(mae_acc).toLocaleString(),
    };
  });
}

const CHART_NAMES = [
  ["backtest_combined.png", "Combined (Volume + New Accounts)"],
  ["backtest_volume_forecast.png", "Volume Forecast (log scale)"],
  ["backtest_volume_levels.png", "Volume Levels (IDR Billion)"],
  ["backtest_new_accounts.png", "New Accounts"],
  ["backtest_error_distribution.png", "Error Distribution"],
  ["ipo_effect_analysis.png", "IPO Effect Analysis"],
] as const;

export default function BacktestTab() {
  const [activeChart, setActiveChart] = useState(0);

  const { data: backtestRows, isLoading, error } = useQuery<BacktestRow[]>({
    queryKey: ["backtestData"],
    queryFn: () => apiGet<BacktestRow[]>("/results/backtest-data"),
    retry: false,
  });

  const { data: summaryData } = useQuery<{ text: string }>({
    queryKey: ["backtestSummary"],
    queryFn: () => apiGet<{ text: string }>("/results/backtest-summary"),
    retry: false,
  });

  const metrics = backtestRows ? computeMetrics(backtestRows) : null;

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-lg font-semibold text-slate-800 mb-1">Backtest Results</h2>
        <p className="text-sm text-slate-500">
          3-cycle rolling backtest — each cycle covers 2 months of weekly predictions.
        </p>
      </div>

      {isLoading && (
        <p className="text-sm text-slate-500">Loading backtest data…</p>
      )}

      {error && (
        <p className="text-sm text-slate-500 bg-slate-100 rounded px-3 py-2">
          No backtest results yet. Run the pipeline first.
        </p>
      )}

      {/* Metrics summary */}
      {metrics && (
        <div className="bg-white border border-slate-200 rounded-xl p-6">
          <h3 className="font-semibold text-slate-800 mb-4">Backtest Metrics by Cycle</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm border-collapse">
              <thead>
                <tr className="bg-slate-50">
                  <th className="text-left px-3 py-2 border border-slate-200 font-medium text-slate-600">Cycle</th>
                  <th className="text-right px-3 py-2 border border-slate-200 font-medium text-slate-600">MAPE</th>
                  <th className="text-right px-3 py-2 border border-slate-200 font-medium text-slate-600">MAE (log)</th>
                  <th className="text-right px-3 py-2 border border-slate-200 font-medium text-slate-600">Direction Acc</th>
                  <th className="text-right px-3 py-2 border border-slate-200 font-medium text-slate-600">95% CI Coverage</th>
                  <th className="text-right px-3 py-2 border border-slate-200 font-medium text-slate-600">MAE (Accounts)</th>
                </tr>
              </thead>
              <tbody>
                {metrics.map((m) => (
                  <tr key={m.cycle} className="hover:bg-slate-50">
                    <td className="px-3 py-2 border border-slate-200 font-medium">Cycle {m.cycle}</td>
                    <td className="px-3 py-2 border border-slate-200 text-right">{m.mape}</td>
                    <td className="px-3 py-2 border border-slate-200 text-right">{m.mae_log}</td>
                    <td className="px-3 py-2 border border-slate-200 text-right">{m.direction_accuracy}</td>
                    <td className="px-3 py-2 border border-slate-200 text-right">{m.ci_coverage}</td>
                    <td className="px-3 py-2 border border-slate-200 text-right">{m.mae_accounts}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Charts */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-4">
        <h3 className="font-semibold text-slate-800">Backtest Charts</h3>

        {/* Chart tabs */}
        <div className="flex flex-wrap gap-2">
          {CHART_NAMES.map(([, label], i) => (
            <button
              key={i}
              onClick={() => setActiveChart(i)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                activeChart === i
                  ? "bg-blue-600 text-white"
                  : "bg-slate-100 text-slate-600 hover:bg-slate-200"
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={`${API_URL}/results/chart/${CHART_NAMES[activeChart][0]}`}
          alt={CHART_NAMES[activeChart][1]}
          className="w-full rounded border border-slate-100"
          onError={(e) => {
            (e.target as HTMLImageElement).style.display = "none";
          }}
        />
      </div>

      {/* Backtest summary text */}
      {summaryData?.text && (
        <div className="bg-white border border-slate-200 rounded-xl p-6">
          <h3 className="font-semibold text-slate-800 mb-3">Backtest Summary Report</h3>
          <pre className="text-xs text-slate-700 bg-slate-50 rounded p-4 overflow-x-auto font-mono whitespace-pre-wrap">
            {summaryData.text}
          </pre>
        </div>
      )}

      {/* Download */}
      <div className="bg-white border border-slate-200 rounded-xl p-6">
        <h3 className="font-semibold text-slate-800 mb-4">Download Backtest Files</h3>
        <div className="flex flex-wrap gap-3">
          <a
            href={downloadUrl("/results/download/csv/backtest_results.csv")}
            className="border border-slate-300 text-slate-700 px-4 py-2 rounded-lg text-sm hover:bg-slate-50 transition-colors"
          >
            backtest_results.csv
          </a>
          <a
            href={downloadUrl("/results/download/csv/ipo_impact_analysis.csv")}
            className="border border-slate-300 text-slate-700 px-4 py-2 rounded-lg text-sm hover:bg-slate-50 transition-colors"
          >
            ipo_impact_analysis.csv
          </a>
          <a
            href={downloadUrl("/results/download/reports/backtest_summary.txt")}
            className="border border-slate-300 text-slate-700 px-4 py-2 rounded-lg text-sm hover:bg-slate-50 transition-colors"
          >
            backtest_summary.txt
          </a>
          <a
            href={downloadUrl("/results/download-all")}
            className="bg-blue-600 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors"
          >
            Download All (ZIP)
          </a>
        </div>
      </div>
    </div>
  );
}
