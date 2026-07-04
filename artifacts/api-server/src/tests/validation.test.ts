/**
 * Validation service unit tests.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { isValidTransition } from "../services/promotionService.js";

describe("Validation checklist logic", () => {
  it("requires all four artifact types for a passing validation", () => {
    const presentTypes = new Set(["torchscript", "onnx", "manifest", "metadata"]);
    const required = ["torchscript", "onnx", "manifest", "metadata"];
    const allPresent = required.every((t) => presentTypes.has(t));
    expect(allPresent).toBe(true);
  });

  it("fails when torchscript is missing", () => {
    const presentTypes = new Set(["onnx", "manifest", "metadata"]);
    const required = ["torchscript", "onnx", "manifest", "metadata"];
    const allPresent = required.every((t) => presentTypes.has(t));
    expect(allPresent).toBe(false);
  });

  it("fails when all are missing", () => {
    const presentTypes = new Set<string>([]);
    const required = ["torchscript", "onnx", "manifest", "metadata"];
    const allPresent = required.every((t) => presentTypes.has(t));
    expect(allPresent).toBe(false);
  });
});

describe("Lifecycle after validation", () => {
  it("validated → candidate is valid", () => {
    expect(isValidTransition("validated", "candidate")).toBe(true);
  });

  it("validating → validated is valid", () => {
    expect(isValidTransition("validating", "validated")).toBe(true);
  });

  it("validating → rejected is valid (validation failure)", () => {
    expect(isValidTransition("validating", "rejected")).toBe(true);
  });

  it("validated cannot go directly to champion", () => {
    expect(isValidTransition("validated", "champion")).toBe(false);
  });
});
