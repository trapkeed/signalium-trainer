/**
 * Re-exports the platform client's public API from the trainer service layer.
 *
 * Import from here rather than the internal module directly so we have one
 * canonical import path if the underlying module is ever relocated.
 *
 *   import { pushTelemetry, PlatformClientError } from "./lib/platform_client_exports";
 */

export {
  pushTelemetry,
  pushTick,
  pushHeartbeat,
  pushExecutionQuality,
  pushSymbolDiscovery,
  PlatformClientError,
  PLATFORM_BASE,
} from "./platform_client.js";

export type {
  TelemetryPayload,
  TickPayload,
  HeartbeatPayload,
  ExecutionQualityPayload,
  SymbolDiscoveryPayload,
  PlatformResponse,
} from "./platform_client.js";
