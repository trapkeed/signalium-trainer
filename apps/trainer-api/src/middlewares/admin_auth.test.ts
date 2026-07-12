/**
 * Unit tests for the adminAuth Express middleware.
 */

import { describe, it, expect, vi, afterEach } from "vitest";
import type { Request, Response, NextFunction } from "express";
import { adminAuth } from "./admin_auth.js";
import * as jwtVerify from "../lib/jwt_verify.js";
import { JwtVerifyError } from "../lib/jwt_verify.js";

function makeReq(authHeader?: string): Partial<Request> {
  return {
    url: "/api/admin/research/summary",
    headers: authHeader ? { authorization: authHeader } : {},
  };
}

function makeRes(): {
  res: Partial<Response>;
  status: ReturnType<typeof vi.fn>;
  json: ReturnType<typeof vi.fn>;
} {
  const json = vi.fn();
  const status = vi.fn().mockReturnValue({ json });
  const res = { status, json } as unknown as Partial<Response>;
  return { res, status, json };
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("adminAuth middleware", () => {
  it("calls next() on a valid platform JWT", async () => {
    const fakeClaims = {
      iss: "signalium",
      aud: "signalium-trainer",
      sub: "platform-svc",
      exp: 9999999999,
    };
    vi.spyOn(jwtVerify, "verifyPlatformJwt").mockResolvedValue(
      fakeClaims as jwtVerify.JwtClaims,
    );
    const next: NextFunction = vi.fn();
    const { res } = makeRes();

    await adminAuth(makeReq("Bearer validtoken") as Request, res as Response, next);

    expect(next).toHaveBeenCalledOnce();
  });

  it("returns 401 when Authorization header is absent", async () => {
    const next: NextFunction = vi.fn();
    const { res, status, json } = makeRes();

    await adminAuth(makeReq() as Request, res as Response, next);

    expect(next).not.toHaveBeenCalled();
    expect(status).toHaveBeenCalledWith(401);
    expect(json).toHaveBeenCalledWith(
      expect.objectContaining({ success: false }),
    );
  });

  it("returns 401 when Authorization header is not a Bearer token", async () => {
    const next: NextFunction = vi.fn();
    const { res, status } = makeRes();

    await adminAuth(makeReq("Basic dXNlcjpwYXNz") as Request, res as Response, next);

    expect(next).not.toHaveBeenCalled();
    expect(status).toHaveBeenCalledWith(401);
  });

  it("returns 401 when JWT is expired", async () => {
    vi.spyOn(jwtVerify, "verifyPlatformJwt").mockRejectedValue(
      new JwtVerifyError("EXPIRED", "JWT has expired"),
    );
    const next: NextFunction = vi.fn();
    const { res, status } = makeRes();

    await adminAuth(makeReq("Bearer expiredtoken") as Request, res as Response, next);

    expect(next).not.toHaveBeenCalled();
    expect(status).toHaveBeenCalledWith(401);
  });

  it("returns 403 when issuer is wrong", async () => {
    vi.spyOn(jwtVerify, "verifyPlatformJwt").mockRejectedValue(
      new JwtVerifyError("WRONG_ISSUER", "Expected iss=signalium, got evil"),
    );
    const next: NextFunction = vi.fn();
    const { res, status } = makeRes();

    await adminAuth(makeReq("Bearer badissuertoken") as Request, res as Response, next);

    expect(next).not.toHaveBeenCalled();
    expect(status).toHaveBeenCalledWith(403);
  });

  it("returns 403 when audience is wrong", async () => {
    vi.spyOn(jwtVerify, "verifyPlatformJwt").mockRejectedValue(
      new JwtVerifyError("WRONG_AUDIENCE", "Expected aud=signalium-trainer"),
    );
    const next: NextFunction = vi.fn();
    const { res, status } = makeRes();

    await adminAuth(makeReq("Bearer badaudtoken") as Request, res as Response, next);

    expect(next).not.toHaveBeenCalled();
    expect(status).toHaveBeenCalledWith(403);
  });

  it("returns 403 when signature is invalid", async () => {
    vi.spyOn(jwtVerify, "verifyPlatformJwt").mockRejectedValue(
      new JwtVerifyError("INVALID_SIGNATURE", "JWT signature is invalid"),
    );
    const next: NextFunction = vi.fn();
    const { res, status } = makeRes();

    await adminAuth(makeReq("Bearer badsigtoken") as Request, res as Response, next);

    expect(next).not.toHaveBeenCalled();
    expect(status).toHaveBeenCalledWith(403);
  });

  it("returns 403 when JWT is malformed", async () => {
    vi.spyOn(jwtVerify, "verifyPlatformJwt").mockRejectedValue(
      new JwtVerifyError("MALFORMED", "JWT must have three dot-separated parts"),
    );
    const next: NextFunction = vi.fn();
    const { res, status } = makeRes();

    await adminAuth(makeReq("Bearer notajwt") as Request, res as Response, next);

    expect(next).not.toHaveBeenCalled();
    expect(status).toHaveBeenCalledWith(403);
  });

  it("returns 403 when kid is unknown", async () => {
    vi.spyOn(jwtVerify, "verifyPlatformJwt").mockRejectedValue(
      new JwtVerifyError("UNKNOWN_KID", "Unknown JWKS kid: old-key"),
    );
    const next: NextFunction = vi.fn();
    const { res, status } = makeRes();

    await adminAuth(makeReq("Bearer unknownkidtoken") as Request, res as Response, next);

    expect(next).not.toHaveBeenCalled();
    expect(status).toHaveBeenCalledWith(403);
  });

  it("returns 503 on unexpected errors (e.g. JWKS fetch failure)", async () => {
    vi.spyOn(jwtVerify, "verifyPlatformJwt").mockRejectedValue(
      new Error("JWKS fetch failed: 503 Service Unavailable"),
    );
    const next: NextFunction = vi.fn();
    const { res, status } = makeRes();

    await adminAuth(makeReq("Bearer sometoken") as Request, res as Response, next);

    expect(next).not.toHaveBeenCalled();
    expect(status).toHaveBeenCalledWith(503);
  });
});
