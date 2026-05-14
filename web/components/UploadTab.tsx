"use client";

import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { apiUpload } from "@/lib/api";

type UploadState = "idle" | "uploading" | "success" | "error";

interface FileZoneProps {
  label: string;
  description: string;
  fileType: string;
  required?: boolean;
  onStatusChange: (type: string, success: boolean) => void;
}

function FileZone({ label, description, fileType, required, onStatusChange }: FileZoneProps) {
  const [state, setState] = useState<UploadState>("idle");
  const [filename, setFilename] = useState<string | null>(null);
  const [errors, setErrors] = useState<string[]>([]);

  const onDrop = useCallback(
    async (accepted: File[]) => {
      const file = accepted[0];
      if (!file) return;
      setState("uploading");
      setErrors([]);
      try {
        const res = await apiUpload(`/upload/${fileType}`, file);
        setFilename(res.filename);
        setState("success");
        onStatusChange(fileType, true);
      } catch (err: unknown) {
        const e = err as Error & { errors?: string[] };
        setErrors(e.errors ?? [e.message ?? "Upload failed"]);
        setState("error");
        onStatusChange(fileType, false);
      }
    },
    [fileType, onStatusChange],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "text/csv": [".csv"] },
    maxFiles: 1,
    disabled: state === "uploading",
  });

  const borderColor =
    state === "success"
      ? "border-green-400"
      : state === "error"
        ? "border-red-400"
        : isDragActive
          ? "border-blue-400"
          : "border-slate-300";

  const bgColor =
    state === "success"
      ? "bg-green-50"
      : state === "error"
        ? "bg-red-50"
        : isDragActive
          ? "bg-blue-50"
          : "bg-slate-50";

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <span className="text-sm font-medium text-slate-700">{label}</span>
        {required ? (
          <span className="text-xs text-red-500 font-medium">required</span>
        ) : (
          <span className="text-xs text-slate-400">optional</span>
        )}
      </div>
      <p className="text-xs text-slate-500">{description}</p>

      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-5 text-center cursor-pointer transition-colors ${borderColor} ${bgColor}`}
      >
        <input {...getInputProps()} />

        {state === "uploading" && (
          <p className="text-sm text-blue-600">Uploading…</p>
        )}
        {state === "success" && (
          <div className="flex items-center justify-center gap-2 text-green-700">
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
            <span className="text-sm font-medium">{filename}</span>
          </div>
        )}
        {state === "idle" && (
          <p className="text-sm text-slate-500">
            {isDragActive ? "Drop it here" : "Drop CSV here or click to browse"}
          </p>
        )}
        {state === "error" && (
          <p className="text-sm text-slate-500">Drop a new file to retry</p>
        )}
      </div>

      {state === "error" && errors.length > 0 && (
        <ul className="text-xs text-red-600 bg-red-50 border border-red-200 rounded px-3 py-2 space-y-1">
          {errors.map((e, i) => (
            <li key={i}>• {e}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

interface UploadTabProps {
  onUploadSuccess: (type: string) => void;
}

export default function UploadTab({ onUploadSuccess }: UploadTabProps) {
  const handleStatusChange = (type: string, success: boolean) => {
    if (success) onUploadSuccess(type);
  };

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-lg font-semibold text-slate-800 mb-1">Upload Input Data</h2>
        <p className="text-sm text-slate-500">
          Upload the required CSV files before running the forecast. Required files must be uploaded
          first.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <FileZone
          label="Historical Volume Data"
          description="Daily IDR trading volume from 2023-01-01 onwards. Columns: Date, volume."
          fileType="volume"
          required
          onStatusChange={handleStatusChange}
        />

        <FileZone
          label="Scenarios"
          description="8-week forward scenario assumptions (BASE/BULL/BEAR). Must start after the volume data ends."
          fileType="scenarios"
          required
          onStatusChange={handleStatusChange}
        />

        <FileZone
          label="Macro Shocks"
          description="Weekly macro events, shock scores, and BI policy rates. Columns: week_end_date, shock_score, policy_rate…"
          fileType="macro"
          onStatusChange={handleStatusChange}
        />

        <FileZone
          label="IPO Calendar"
          description="IPO announcement dates and details. Use book-open date, NOT listing date."
          fileType="ipo"
          onStatusChange={handleStatusChange}
        />
      </div>

      <div className="text-xs text-slate-400 bg-slate-100 rounded-lg p-3 space-y-1">
        <p>
          <strong>Note:</strong> Volume data must start on or before 2023-01-01.
        </p>
        <p>Scenarios must start after the last date in your volume file.</p>
        <p>Files are validated before saving — errors will be shown above the dropzone.</p>
      </div>
    </div>
  );
}
