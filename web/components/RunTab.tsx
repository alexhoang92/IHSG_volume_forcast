"use client";

import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { API_URL, apiGet, apiPost, downloadUrl } from "@/lib/api";

interface JobStatus {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed";
  created_at: string;
  completed_at: string | null;
  log_tail: string[];
  error: string | null;
}

interface ScenarioRow {
  week_end_date: string;
  BASE_vol_idr_bn?: number;
  BASE_vol_lower_95?: number;
  BASE_vol_upper_95?: number;
  BASE_new_accounts?: number;
  BULL_vol_idr_bn?: number;
  BULL_new_accounts?: number;
  BEAR_vol_idr_bn?: number;
  BEAR_new_accounts?: number;
}

function fmt(n: number | undefined | null): string {
  if (n == null || isNaN(Number(n))) return "—";
  return Number(n).toLocaleString("en-US", { maximumFractionDigits: 0 });
}

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    queued: "bg-slate-100 text-slate-600",
    running: "bg-yellow-100 text-yellow-700",
    completed: "bg-green-100 text-green-700",
    failed: "bg-red-100 text-red-700",
  };
  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium ${colors[status] ?? "bg-slate-100 text-slate-600"}`}>
      {status === "running" && (
        <svg className="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
        </svg>
      )}
      {status}
    </span>
  );
}

interface RunTabProps {
  volumeReady: boolean;
  scenariosReady: boolean;
}

export default function RunTab({ volumeReady, scenariosReady }: RunTabProps) {
  const [jobId, setJobId] = useState<string | null>(null);
  const [skipFetch, setSkipFetch] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [chartTs, setChartTs] = useState<number>(Date.now());
  const logRef = useRef<HTMLPreElement>(null);

  const canRun = volumeReady && scenariosReady;

  const { data: jobStatus } = useQuery<JobStatus>({
    queryKey: ["jobStatus", jobId],
    queryFn: () => apiGet<JobStatus>(`/run/status/${jobId}`),
    refetchInterval: (query) => {
      const s = query.state.data?.status;
      return s === "completed" || s === "failed" ? false : 5000;
    },
    enabled: !!jobId,
  });

  // Check for a previously running job on mount
  useEffect(() => {
    apiGet<JobStatus | null>("/run/latest").then((job) => {
      if (job && (job.status === "running" || job.status === "queued")) {
        setJobId(job.job_id);
      }
    }).catch(() => {});
  }, []);

  // Auto-scroll log
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [jobStatus?.log_tail]);

  // Bump chart timestamp when job completes
  useEffect(() => {
    if (jobStatus?.status === "completed") {
      setChartTs(Date.now());
    }
  }, [jobStatus?.status]);

  async function handleRun() {
    setRunError(null);
    try {
      const res = await apiPost<{ job_id: string }>(`/run/?skip_fetch=${skipFetch}`);
      setJobId(res.job_id);
    } catch (err: unknown) {
      setRunError((err as Error).message);
    }
  }

  const { data: scenarioData } = useQuery<ScenarioRow[]>({
    queryKey: ["scenarioData", jobStatus?.completed_at],
    queryFn: () => apiGet<ScenarioRow[]>("/results/scenario-data"),
    enabled: jobStatus?.status === "completed",
  });

  const isRunning = jobStatus?.status === "queued" || jobStatus?.status === "running";

  return (
    <div className="space-y-8">
      {/* Run controls */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-4">
        <h2 className="text-lg font-semibold text-slate-800">Run Forecast Pipeline</h2>

        {!canRun && (
          <p className="text-sm text-amber-700 bg-amber-50 border border-amber-200 rounded px-3 py-2">
            Upload both <strong>Volume data</strong> and <strong>Scenarios</strong> before running.
          </p>
        )}

        <label className="flex items-center gap-2 text-sm text-slate-600 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={skipFetch}
            onChange={(e) => setSkipFetch(e.target.checked)}
            className="rounded"
          />
          Skip Yahoo Finance fetch (use uploaded volume file only)
        </label>

        <button
          onClick={handleRun}
          disabled={!canRun || isRunning}
          className="bg-blue-600 text-white px-6 py-2 rounded-lg text-sm font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {isRunning ? "Pipeline Running…" : "Run Forecast Pipeline"}
        </button>

        {runError && (
          <p className="text-sm text-red-600 bg-red-50 border border-red-200 rounded px-3 py-2">
            {runError}
          </p>
        )}
      </div>

      {/* Job status */}
      {jobStatus && (
        <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-4">
          <div className="flex items-center gap-3">
            <h3 className="font-medium text-slate-800">Pipeline Status</h3>
            <StatusBadge status={jobStatus.status} />
          </div>

          {jobStatus.status === "running" && (
            <p className="text-xs text-slate-500">
              Pipeline typically completes in 3–8 minutes. This page polls for updates every 5 seconds.
            </p>
          )}

          {jobStatus.log_tail.length > 0 && (
            <pre
              ref={logRef}
              className="bg-slate-900 text-green-300 text-xs rounded-lg p-4 overflow-auto max-h-48 font-mono"
            >
              {jobStatus.log_tail.join("\n")}
            </pre>
          )}

          {jobStatus.status === "failed" && jobStatus.error && (
            <p className="text-sm text-red-600 bg-red-50 border border-red-200 rounded px-3 py-2">
              Error: {jobStatus.error}
            </p>
          )}

          {jobStatus.status === "completed" && (
            <p className="text-sm text-green-700 bg-green-50 border border-green-200 rounded px-3 py-2">
              Pipeline completed successfully.
            </p>
          )}
        </div>
      )}

      {/* Results */}
      {jobStatus?.status === "completed" && (
        <div className="space-y-6">
          {/* Fan chart */}
          <div className="bg-white border border-slate-200 rounded-xl p-6">
            <h3 className="font-semibold text-slate-800 mb-4">8-Week Scenario Fan Chart</h3>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={`${API_URL}/results/chart/scenario_fan_chart.png?t=${chartTs}`}
              alt="Scenario fan chart"
              className="w-full rounded"
            />
          </div>

          {/* Scenario comparison table */}
          {scenarioData && scenarioData.length > 0 && (
            <div className="bg-white border border-slate-200 rounded-xl p-6">
              <h3 className="font-semibold text-slate-800 mb-4">
                Forecast Summary — All Scenarios (IDR Billion)
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full text-xs border-collapse">
                  <thead>
                    <tr className="bg-slate-50">
                      <th className="text-left px-3 py-2 border border-slate-200 font-medium text-slate-600">Week End</th>
                      <th className="text-right px-3 py-2 border border-slate-200 font-medium text-blue-600">BASE Vol (IDR Bn)</th>
                      <th className="text-right px-3 py-2 border border-slate-200 font-medium text-blue-600">BASE Accts</th>
                      <th className="text-right px-3 py-2 border border-slate-200 font-medium text-green-600">BULL Vol (IDR Bn)</th>
                      <th className="text-right px-3 py-2 border border-slate-200 font-medium text-green-600">BULL Accts</th>
                      <th className="text-right px-3 py-2 border border-slate-200 font-medium text-red-500">BEAR Vol (IDR Bn)</th>
                      <th className="text-right px-3 py-2 border border-slate-200 font-medium text-red-500">BEAR Accts</th>
                    </tr>
                  </thead>
                  <tbody>
                    {scenarioData.map((row, i) => (
                      <tr key={i} className="hover:bg-slate-50">
                        <td className="px-3 py-2 border border-slate-200 text-slate-700">{row.week_end_date}</td>
                        <td className="px-3 py-2 border border-slate-200 text-right">{fmt(row.BASE_vol_idr_bn)}</td>
                        <td className="px-3 py-2 border border-slate-200 text-right">{fmt(row.BASE_new_accounts)}</td>
                        <td className="px-3 py-2 border border-slate-200 text-right">{fmt(row.BULL_vol_idr_bn)}</td>
                        <td className="px-3 py-2 border border-slate-200 text-right">{fmt(row.BULL_new_accounts)}</td>
                        <td className="px-3 py-2 border border-slate-200 text-right">{fmt(row.BEAR_vol_idr_bn)}</td>
                        <td className="px-3 py-2 border border-slate-200 text-right">{fmt(row.BEAR_new_accounts)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Download buttons */}
          <div className="bg-white border border-slate-200 rounded-xl p-6">
            <h3 className="font-semibold text-slate-800 mb-4">Download Results</h3>
            <div className="flex flex-wrap gap-3">
              <a
                href={downloadUrl("/results/download-all")}
                className="bg-blue-600 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors"
              >
                Download All (ZIP)
              </a>
              {[
                ["forecast_BASE.csv", "BASE Forecast"],
                ["forecast_BULL.csv", "BULL Forecast"],
                ["forecast_BEAR.csv", "BEAR Forecast"],
                ["forecast_summary_table.csv", "Summary Table"],
                ["forecast_all_scenarios.csv", "All Scenarios"],
                ["sensitivity_ranking.csv", "Sensitivity Ranking"],
              ].map(([file, label]) => (
                <a
                  key={file}
                  href={downloadUrl(`/results/download/csv/scenarios/${file}`)}
                  className="border border-slate-300 text-slate-700 px-4 py-2 rounded-lg text-sm hover:bg-slate-50 transition-colors"
                >
                  {label}
                </a>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
