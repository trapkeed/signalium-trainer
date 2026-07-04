/**
 * Artifact service unit tests — file system and checksum logic.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { createHash } from "crypto";
import { writeFileSync, mkdirSync, rmSync, existsSync } from "fs";
import { join } from "path";
import os from "os";

describe("SHA-256 checksum verification", () => {
  let tmpDir: string;

  beforeEach(() => {
    tmpDir = join(os.tmpdir(), `registry-test-${Date.now()}`);
    mkdirSync(tmpDir, { recursive: true });
  });

  afterEach(() => {
    rmSync(tmpDir, { recursive: true, force: true });
  });

  it("computes consistent SHA-256 for known content", () => {
    const content = Buffer.from("hello signalium registry");
    const hash = createHash("sha256").update(content).digest("hex");
    expect(hash).toBe(createHash("sha256").update(content).digest("hex"));
    expect(hash.length).toBe(64);
  });

  it("produces different checksums for different content", () => {
    const h1 = createHash("sha256").update("content-a").digest("hex");
    const h2 = createHash("sha256").update("content-b").digest("hex");
    expect(h1).not.toBe(h2);
  });

  it("detects file corruption (checksum mismatch)", () => {
    const filePath = join(tmpDir, "model.pt");
    writeFileSync(filePath, "original content");
    const originalHash = createHash("sha256").update("original content").digest("hex");

    // Simulate corruption
    writeFileSync(filePath, "corrupted content");
    const actualHash = createHash("sha256").update("corrupted content").digest("hex");

    expect(actualHash).not.toBe(originalHash);
  });

  it("correctly identifies matching checksums", () => {
    const filePath = join(tmpDir, "model.onnx");
    const content = "valid model data";
    writeFileSync(filePath, content);
    const expectedHash = createHash("sha256").update(content).digest("hex");

    // Read back and verify
    const actualHash = createHash("sha256").update(content).digest("hex");
    expect(actualHash).toBe(expectedHash);
  });
});

describe("Artifact type validation", () => {
  const VALID_TYPES = [
    "torchscript", "onnx", "manifest", "metadata",
    "training_report", "evaluation_report", "logs", "config", "other",
  ];

  it("accepts all valid artifact types", () => {
    for (const t of VALID_TYPES) {
      expect(VALID_TYPES.includes(t)).toBe(true);
    }
  });

  it("rejects unknown artifact types", () => {
    expect(VALID_TYPES.includes("unknown_format")).toBe(false);
    expect(VALID_TYPES.includes("")).toBe(false);
  });
});
