/**
 * JWKS client for the Signalium Platform.
 *
 * Fetches and caches public keys from:
 *   GET https://signaliumai.com/api/sso/jwks
 *
 * Keys are cached by `kid` for JWKS_CACHE_TTL_MS. On a cache miss
 * for an unknown kid the JWKS is re-fetched once before rejecting.
 */

export interface JwkPublicKey {
  kty: string;
  use?: string;
  kid: string;
  alg?: string;
  crv?: string;
  x?: string;
  y?: string;
  n?: string;
  e?: string;
}

interface JwksCache {
  keys: Map<string, JwkPublicKey>;
  fetchedAt: number;
}

const JWKS_URL =
  process.env["SIGNALIUM_JWKS_URL"] ??
  "https://signaliumai.com/api/sso/jwks";
const JWKS_CACHE_TTL_MS = 10 * 60 * 1000; // 10 minutes

let cache: JwksCache | null = null;

async function fetchAndCache(): Promise<JwksCache> {
  const res = await fetch(JWKS_URL);
  if (!res.ok) {
    throw new Error(`JWKS fetch failed: ${res.status} ${res.statusText}`);
  }
  const body = (await res.json()) as { keys: JwkPublicKey[] };
  const keys = new Map<string, JwkPublicKey>();
  for (const key of body.keys ?? []) {
    if (key.kid) keys.set(key.kid, key);
  }
  cache = { keys, fetchedAt: Date.now() };
  return cache;
}

/** Force-clear the in-memory cache (useful in tests). */
export function clearJwksCache(): void {
  cache = null;
}

/**
 * Return the JWK for the given `kid`.
 *
 * On a cache miss (or a stale cache) the JWKS endpoint is re-fetched once.
 * Throws if the key is still not found after the re-fetch.
 */
export async function getJwk(kid: string): Promise<JwkPublicKey> {
  // Serve from cache if fresh and the key is present
  if (cache && Date.now() - cache.fetchedAt < JWKS_CACHE_TTL_MS) {
    const key = cache.keys.get(kid);
    if (key) return key;
  }

  // Cache is stale or key unknown — re-fetch once
  const fresh = await fetchAndCache();
  const key = fresh.keys.get(kid);
  if (!key) {
    throw new Error(`Unknown JWKS kid: ${kid}`);
  }
  return key;
}

/** Inject keys directly (used in tests to avoid network calls). */
export function _injectJwksForTest(keys: JwkPublicKey[]): void {
  const map = new Map<string, JwkPublicKey>();
  for (const k of keys) map.set(k.kid, k);
  cache = { keys: map, fetchedAt: Date.now() };
}
