// Research Lab Admin Dashboard — API client.
//
// Talks to the Express proxy mounted by @workspace/trainer-api at
// /api/admin/research/*, which forwards to the private Python Research Lab
// FastAPI server (port 8002) with the admin bearer token attached.
//
// IMPORTANT: this app's own base path is /research/dashboard/, but the API
// lives on a SIBLING artifact at /api — so every call below uses an
// absolute "/api/..." path (never import.meta.env.BASE_URL). The shared
// Replit proxy at localhost:80 routes "/api/*" to the trainer-api service
// regardless of which artifact's frontend issued the request.

const API_BASE = "/api/admin/research";

export class ApiError extends Error {
  status: number;
  data: unknown;
  constructor(status: number, message: string, data?: unknown) {
    super(message);
    this.status = status;
    this.data = data;
  }
}

let authToken: string | null = null;

type AuthChangeListener = (token: string | null) => void;
const authChangeListeners = new Set<AuthChangeListener>();

/** Subscribe to auth token changes (including forced logout on 401/403). Returns an unsubscribe function. */
export function onAuthChange(listener: AuthChangeListener): () => void {
  authChangeListeners.add(listener);
  return () => authChangeListeners.delete(listener);
}

export function setAuthToken(token: string | null) {
  authToken = token;
  if (token) {
    sessionStorage.setItem("rl_admin_token", token);
  } else {
    sessionStorage.removeItem("rl_admin_token");
  }
  for (const listener of authChangeListeners) listener(token);
}

export function getAuthToken(): string | null {
  if (authToken) return authToken;
  authToken = sessionStorage.getItem("rl_admin_token");
  return authToken;
}

async function request<T>(
  path: string,
  options: { method?: "GET" | "POST"; body?: unknown; params?: Record<string, string | number | undefined> } = {},
): Promise<T> {
  const token = getAuthToken();
  if (!token) {
    throw new ApiError(401, "Not authenticated");
  }
  let url = `${API_BASE}${path}`;
  if (options.params) {
    const qs = new URLSearchParams();
    for (const [k, v] of Object.entries(options.params)) {
      if (v !== undefined && v !== null && v !== "") qs.set(k, String(v));
    }
    const s = qs.toString();
    if (s) url += `?${s}`;
  }
  const res = await fetch(url, {
    method: options.method ?? "GET",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
    body: options.body !== undefined ? JSON.stringify(options.body) : undefined,
  });
  let data: unknown = null;
  try {
    data = await res.json();
  } catch {
    // no body
  }
  if (!res.ok) {
    if (res.status === 401 || res.status === 403) {
      setAuthToken(null);
    }
    const message =
      (data as { detail?: string; error?: string } | null)?.detail ??
      (data as { detail?: string; error?: string } | null)?.error ??
      res.statusText;
    throw new ApiError(res.status, message, data);
  }
  return data as T;
}

const get = <T,>(path: string, params?: Record<string, string | number | undefined>) =>
  request<T>(path, { method: "GET", params });

const post = <T,>(path: string, body?: unknown, params?: Record<string, string | number | undefined>) =>
  request<T>(path, { method: "POST", body: body ?? {}, params });

// ── Verifies a token by hitting a cheap authenticated endpoint ───────────────
export async function verifyToken(token: string): Promise<boolean> {
  const prevToken = getAuthToken();
  setAuthToken(token);
  try {
    await get("/summary");
    return true;
  } catch (err) {
    if (err instanceof ApiError && (err.status === 401 || err.status === 403)) {
      setAuthToken(prevToken);
      return false;
    }
    // Network/other errors: token could be valid, surface as failure but don't discard it silently.
    setAuthToken(prevToken);
    throw err;
  }
}

// ── Types ─────────────────────────────────────────────────────────────────────

export interface DeploymentRecord {
  model_version: string;
  status: string;
  deployed_at?: number;
  weights_url?: string;
  manifest?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface DeploymentEvent {
  event_type: string;
  model_version: string;
  timestamp: number;
  [key: string]: unknown;
}

export interface HealthStatus {
  dimensions?: Record<
    string,
    { severity: string; last_checked?: number; message?: string; [key: string]: unknown }
  >;
  [key: string]: unknown;
}

export interface HealthAlert {
  alert_id: string;
  dimension: string;
  severity: "WARNING" | "CRITICAL" | string;
  model_version?: string;
  message?: string;
  created_at: number;
  [key: string]: unknown;
}

export interface ApprovalRequest {
  request_id: string;
  gate_type: string;
  resource_id: string;
  resource_type?: string;
  status: "PENDING" | "APPROVED" | "REJECTED" | "EXPIRED" | "CONSUMED" | string;
  requested_by?: string;
  action_summary?: string;
  created_at: number;
  resolved_by?: string;
  [key: string]: unknown;
}

export interface SymbolPerformance {
  symbol: string;
  trade_count: number;
  win_rate: number;
  sharpe: number;
  avg_confidence_at_entry: number;
  confidence_reliability: number | null;
  total_pnl: number;
}

export interface CalibrationBucket {
  bucket_lo: number;
  bucket_hi: number;
  trade_count: number;
  mean_confidence: number | null;
  observed_win_rate: number | null;
}

export interface CalibrationResponse {
  n_trades: number;
  expected_calibration_error: number | null;
  curve: CalibrationBucket[];
}

export interface ReasoningBucket {
  bucket_start: number;
  n_events: number;
  head_means: Record<string, number>;
}

export interface ReasoningDistributionResponse {
  n_events: number;
  series: ReasoningBucket[];
}

export interface TrainingRun {
  run_id: string;
  experiment_id: string;
  dataset_version: string;
  feature_version: string;
  code_commit: string;
  config_hash: string;
  artifact_hash: string;
  model_version?: string;
  created_at: number;
  [key: string]: unknown;
}

export interface DiscoveredFeature {
  feature_id: string;
  name: string;
  category: string;
  importance_score: number;
  status: string;
  notes?: string;
  [key: string]: unknown;
}

export interface ShadowComparison {
  test_id: string;
  champion_version: string;
  challenger_version: string;
  dimensions?: Record<string, { winner: string; [key: string]: unknown }>;
  rolling_sharpe?: number;
  agreement_rate?: number;
  [key: string]: unknown;
}

// ── Overview ──────────────────────────────────────────────────────────────────

export const getSummary = () => get<{ success: boolean } & Record<string, unknown>>("/summary");
export const getCurrentDeployment = () =>
  get<{ success: boolean; active: DeploymentRecord | null }>("/deployment/current");
export const getHealthStatus = (model_version?: string) =>
  get<{ success: boolean; status: HealthStatus }>("/health/status", { model_version });
export const getHealthAlerts = (params?: {
  model_version?: string;
  severity?: string;
  dimension?: string;
  limit?: number;
}) => get<{ success: boolean; total: number; alerts: HealthAlert[] }>("/health/alerts", params);

// ── Approvals ─────────────────────────────────────────────────────────────────

export const listApprovals = (params?: { gate_type?: string; status?: string; limit?: number }) =>
  get<{ success: boolean; total: number; approval_requests: ApprovalRequest[] }>(
    "/approvals",
    params,
  );
export const approveApproval = (requestId: string, reason?: string) =>
  post<{ success: boolean; approval_request: ApprovalRequest; authenticated_as: string }>(
    `/approvals/${requestId}/approve`,
    { reason },
  );
export const rejectApproval = (requestId: string, reason: string) =>
  post<{ success: boolean; approval_request: ApprovalRequest; authenticated_as: string }>(
    `/approvals/${requestId}/reject`,
    { reason },
  );

// ── Symbol Performance & Calibration ──────────────────────────────────────────

export const getSymbolPerformance = (params?: {
  model_version?: string;
  since?: number;
  until?: number;
}) =>
  get<{ success: boolean; total: number; symbols: SymbolPerformance[] }>(
    "/symbol-performance",
    params,
  );

export const getCalibration = (params?: {
  model_version?: string;
  since?: number;
  until?: number;
  bucket_width?: number;
}) => get<{ success: boolean } & CalibrationResponse>("/calibration", params);

// ── Reasoning / Drift ─────────────────────────────────────────────────────────

export const getReasoningDistribution = (params?: {
  model_version?: string;
  symbol?: string;
  since?: number;
  until?: number;
  bucket_seconds?: number;
}) => get<{ success: boolean } & ReasoningDistributionResponse>("/reasoning-distribution", params);

// ── Promotion History / Deployment Log ────────────────────────────────────────

export const getDeploymentHistory = (params?: {
  model_version?: string;
  event_type?: string;
  limit?: number;
}) => get<{ success: boolean; events: DeploymentEvent[]; total: number }>(
  "/deployment/history",
  params,
);

// ── Dataset Lineage (Training Runs) ───────────────────────────────────────────

export const listTrainingRuns = (params?: {
  experiment_id?: string;
  pipeline_job_id?: string;
  limit?: number;
}) => get<{ success: boolean; total: number; training_runs: TrainingRun[] }>(
  "/artifacts/runs",
  params,
);

// ── Feature Importance ────────────────────────────────────────────────────────

export const listFeatures = (status?: string) =>
  get<{ success: boolean; total: number; features: DiscoveredFeature[] }>("/features", { status });

// ── Champion vs Challenger ─────────────────────────────────────────────────────

export const getShadowStatus = (testId?: string) =>
  get<{ success: boolean; status: unknown }>("/shadow/status", { test_id: testId });

export const getShadowComparison = (testId: string) =>
  get<{ success: boolean; comparison: ShadowComparison }>("/shadow/comparison", {
    test_id: testId,
  });
