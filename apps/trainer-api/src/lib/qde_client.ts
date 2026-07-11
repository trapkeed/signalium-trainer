/**
 * Lightweight typed HTTP client for the QDE Python service.
 *
 * All routes in the trainer-api that proxy to the QDE service should go
 * through these helpers so error handling and base-URL resolution are
 * consistent across the codebase.
 */

export const QDE_BASE = process.env["QDE_API_URL"] ?? "http://localhost:8001";

export class QdeClientError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "QdeClientError";
  }
}

/** POST to the QDE service. Throws QdeClientError on non-2xx responses. */
export async function qdePost(path: string, body: unknown): Promise<unknown> {
  let res: Response;
  try {
    res = await fetch(`${QDE_BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (err) {
    throw new QdeClientError(502, `QDE service unreachable: ${String(err)}`);
  }
  const data = await res.json().catch(() => ({ error: "Invalid JSON from QDE service" }));
  if (!res.ok) {
    const msg = (data as { detail?: string; error?: string }).detail
      ?? (data as { error?: string }).error
      ?? `QDE returned ${res.status}`;
    throw new QdeClientError(res.status, msg);
  }
  return data;
}

/** GET from the QDE service. Throws QdeClientError on non-2xx responses. */
export async function qdeGet(path: string): Promise<unknown> {
  let res: Response;
  try {
    res = await fetch(`${QDE_BASE}${path}`);
  } catch (err) {
    throw new QdeClientError(502, `QDE service unreachable: ${String(err)}`);
  }
  const data = await res.json().catch(() => ({ error: "Invalid JSON from QDE service" }));
  if (!res.ok) {
    const msg = (data as { detail?: string; error?: string }).detail
      ?? (data as { error?: string }).error
      ?? `QDE returned ${res.status}`;
    throw new QdeClientError(res.status, msg);
  }
  return data;
}
