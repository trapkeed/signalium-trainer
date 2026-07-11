/**
 * Unit tests for the platformAuth Express middleware.
 */

import { describe, it, expect, vi, afterEach } from "vitest";
import type { Request, Response, NextFunction } from "express";
import { platformAuth } from "./platform_auth.js";
import * as jwtVerify from "../lib/jwt_verify.js";
import { JwtVerifyError } from "../lib/jwt_verify.js";

function makeReq(authHeader?: string): Partial<Request> {
  return {
    url: "/qde/inference",
    headers: authHeader ? { authorization: authHeader } : {},
  };
}

function makeRes(): {
  res: Partial<Response>;
  status: ReturnType<typeof vi.fn>;
  json: ReturnType<typeof vi.fn>;
  locals: Record<string, unknown>;
} {
  const json = vi.fn();
  const status = vi.fn().mockReturnValue({ json });
  const locals: Record<string, unknown> = {};
  const res = { status, json, locals } as unknown as Partial<Response>;
  // Simulate chained status().json()
  (status as ReturnType<typeof vi.fn>).mockReturnValue({ json });
  return { res, status, json, locals };
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("platformAuth middleware", () => {
  it("calls next() on a valid token", async () => {
    const fakeClaims = { iss: "signalium", aud: "signalium-trainer", sub: "svc", exp: 9999999999 };
    vi.spyOn(jwtVerify, "verifyPlatformJwt").mockResolvedValue(fakeClaims as jwtVerify.JwtClaims);
    const next: NextFunction = vi.fn();
    const { res, locals } = makeRes();

    await platformAuth(
      makeReq("Bearer validtoken") as Request,
      res as Response,
      next,
    );

    expect(next).toHaveBeenCalledOnce();
    expect(locals.platformClaims).toEqual(fakeClaims);
  });

  it("returns 401 when Authorization header is absent", async () => {
    const next: NextFunction = vi.fn();
    const { res, status, json } = makeRes();

    await platformAuth(makeReq() as Request, res as Response, next);

    expect(next).not.toHaveBeenCalled();
    expect(status).toHaveBeenCalledWith(401);
    expect(json).toHaveBeenCalledWith({ error: "Unauthorized" });
  });

  it("returns 401 when token is expired", async () => {
    vi.spyOn(jwtVerify, "verifyPlatformJwt").mockRejectedValue(
      new JwtVerifyError("EXPIRED", "JWT has expired"),
    );
    const next: NextFunction = vi.fn();
    const { res, status, json } = makeRes();

    await platformAuth(makeReq("Bearer expiredtoken") as Request, res as Response, next);

    expect(next).not.toHaveBeenCalled();
    expect(status).toHaveBeenCalledWith(401);
    expect(json).toHaveBeenCalledWith({ error: "Unauthorized" });
  });

  it("returns 401 on wrong audience", async () => {
    vi.spyOn(jwtVerify, "verifyPlatformJwt").mockRejectedValue(
      new JwtVerifyError("WRONG_AUDIENCE", "wrong aud"),
    );
    const next: NextFunction = vi.fn();
    const { res, status, json } = makeRes();

    await platformAuth(makeReq("Bearer badaudtoken") as Request, res as Response, next);

    expect(status).toHaveBeenCalledWith(401);
    expect(json).toHaveBeenCalledWith({ error: "Unauthorized" });
  });

  it("returns 401 on invalid signature", async () => {
    vi.spyOn(jwtVerify, "verifyPlatformJwt").mockRejectedValue(
      new JwtVerifyError("INVALID_SIGNATURE", "bad sig"),
    );
    const next: NextFunction = vi.fn();
    const { res, status, json } = makeRes();

    await platformAuth(makeReq("Bearer badsigtoken") as Request, res as Response, next);

    expect(status).toHaveBeenCalledWith(401);
    expect(json).toHaveBeenCalledWith({ error: "Unauthorized" });
  });

  it("returns 401 on unknown kid", async () => {
    vi.spyOn(jwtVerify, "verifyPlatformJwt").mockRejectedValue(
      new JwtVerifyError("UNKNOWN_KID", "unknown kid"),
    );
    const next: NextFunction = vi.fn();
    const { res, status, json } = makeRes();

    await platformAuth(makeReq("Bearer unknownkidtoken") as Request, res as Response, next);

    expect(status).toHaveBeenCalledWith(401);
    expect(json).toHaveBeenCalledWith({ error: "Unauthorized" });
  });

  it("forwards unexpected errors to next(err)", async () => {
    const boom = new Error("network failure");
    vi.spyOn(jwtVerify, "verifyPlatformJwt").mockRejectedValue(boom);
    const next: NextFunction = vi.fn();
    const { res } = makeRes();

    await platformAuth(makeReq("Bearer sometoken") as Request, res as Response, next);

    expect(next).toHaveBeenCalledWith(boom);
  });
});
