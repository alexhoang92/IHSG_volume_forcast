const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export { API_URL };

type FetchOptions = RequestInit & { skipAuth?: boolean };

async function apiFetch(path: string, options: FetchOptions = {}): Promise<Response> {
  const res = await fetch(`${API_URL}${path}`, {
    ...options,
    credentials: "include",
  });

  // Don't redirect on the login endpoint itself — the caller handles 401 there.
  if (res.status === 401 && typeof window !== "undefined" && path !== "/auth/login") {
    window.location.href = "/login";
  }

  return res;
}

export async function apiGet<T>(path: string): Promise<T> {
  const res = await apiFetch(path);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `GET ${path} failed: ${res.status}`);
  }
  return res.json();
}

export async function apiPost<T>(
  path: string,
  body?: Record<string, string>,
): Promise<T> {
  const res = await apiFetch(path, {
    method: "POST",
    headers: body ? { "Content-Type": "application/x-www-form-urlencoded" } : {},
    body: body ? new URLSearchParams(body).toString() : undefined,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `POST ${path} failed: ${res.status}`);
  }
  return res.json();
}

export async function apiUpload(
  path: string,
  file: File,
): Promise<{ message: string; filename: string; size_bytes: number }> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await apiFetch(path, { method: "POST", body: formData });
  if (!res.ok) {
    const json = await res.json().catch(() => null);
    const errors: string[] =
      json?.detail?.errors ?? (json?.detail ? [json.detail] : ["Upload failed"]);
    throw Object.assign(new Error(errors.join("; ")), { errors });
  }
  return res.json();
}

export function downloadUrl(path: string): string {
  return `${API_URL}${path}`;
}
