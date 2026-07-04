/**
 * Registry unit tests — state machine, rate limiter, auth middleware.
 *
 * Run with: pnpm --filter @workspace/api-server test
 */

import { describe, it, expect, vi, beforeEach } from "vitest";

// ─── Promotion state machine ───────────────────────────────────────────────────
import { isValidTransition } from "../services/promotionService.js";

describe("isValidTransition", () => {
  it("allows uploading → uploaded", () => {
    expect(isValidTransition("uploading", "uploaded")).toBe(true);
  });

  it("allows uploaded → validating", () => {
    expect(isValidTransition("uploaded", "validating")).toBe(true);
  });

  it("allows validated → candidate", () => {
    expect(isValidTransition("validated", "candidate")).toBe(true);
  });

  it("allows candidate → challenger", () => {
    expect(isValidTransition("candidate", "challenger")).toBe(true);
  });

  it("allows challenger → champion", () => {
    expect(isValidTransition("challenger", "champion")).toBe(true);
  });

  it("allows champion → production", () => {
    expect(isValidTransition("champion", "production")).toBe(true);
  });

  it("allows production → retired", () => {
    expect(isValidTransition("production", "retired")).toBe(true);
  });

  it("allows retired → archived", () => {
    expect(isValidTransition("retired", "archived")).toBe(true);
  });

  it("rejects backward transitions (production → candidate)", () => {
    expect(isValidTransition("production", "candidate")).toBe(false);
  });

  it("rejects archived → champion", () => {
    expect(isValidTransition("archived", "champion")).toBe(false);
  });

  it("rejects rejected → champion", () => {
    expect(isValidTransition("rejected", "champion")).toBe(false);
  });

  it("uploading cannot be rejected directly", () => {
    expect(isValidTransition("uploading", "rejected")).toBe(false);
  });

  it("allows validated → rejected", () => {
    expect(isValidTransition("validated", "rejected")).toBe(true);
  });
});

// ─── Rate limiter ─────────────────────────────────────────────────────────────
import { rateLimiter } from "../middlewares/rateLimit.js";
import type { Request, Response, NextFunction } from "express";

function mockReq(ip = "127.0.0.1"): Partial<Request> {
  return { ip, headers: {}, method: "GET" } as Partial<Request>;
}

function mockRes() {
  const res = {
    _status: 200,
    status: vi.fn().mockReturnThis(),
    json: vi.fn().mockReturnThis(),
    setHeader: vi.fn(),
  };
  return res;
}

describe("rateLimiter", () => {
  it("allows requests under the limit", () => {
    const limit = rateLimiter(5, 60_000);
    const req = mockReq("10.0.0.10");
    let passCount = 0;
    const next: NextFunction = () => { passCount++; };

    for (let i = 0; i < 5; i++) {
      limit(req as Request, mockRes() as unknown as Response, next);
    }
    expect(passCount).toBe(5);
  });

  it("blocks requests over the limit", () => {
    const limit = rateLimiter(3, 60_000);
    const req = mockReq("10.0.0.20");
    let blockedCount = 0;
    const next: NextFunction = () => {};

    for (let i = 0; i < 4; i++) {
      const res = mockRes();
      limit(req as Request, res as unknown as Response, next);
      if (res.status.mock.calls.length > 0) blockedCount++;
    }
    expect(blockedCount).toBe(1);
  });

  it("allows different IPs independently", () => {
    const limit = rateLimiter(1, 60_000);
    let passCount = 0;
    const next: NextFunction = () => { passCount++; };

    limit(mockReq("10.1.1.1") as Request, mockRes() as unknown as Response, next);
    limit(mockReq("10.1.1.2") as Request, mockRes() as unknown as Response, next);
    expect(passCount).toBe(2);
  });
});

// ─── API key middleware ────────────────────────────────────────────────────────
import { requireApiKey, requireAdminApiKey } from "../middlewares/apiKey.js";

describe("requireApiKey (no keys configured)", () => {
  beforeEach(() => {
    delete process.env.REGISTRY_API_KEY;
    delete process.env.REGISTRY_ADMIN_API_KEY;
  });

  it("passes through when no keys are configured (open dev mode)", () => {
    let called = false;
    const next: NextFunction = () => { called = true; };
    requireApiKey(mockReq() as Request, mockRes() as unknown as Response, next);
    expect(called).toBe(true);
  });
});
