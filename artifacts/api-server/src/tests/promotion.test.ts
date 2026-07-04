/**
 * Promotion workflow unit tests.
 */

import { describe, it, expect } from "vitest";
import { isValidTransition } from "../services/promotionService.js";

type LifecycleStatus =
  | "uploading" | "uploaded" | "validating" | "validated"
  | "candidate" | "challenger" | "champion" | "production"
  | "retired" | "archived" | "rejected";

const ALL_STATUSES: LifecycleStatus[] = [
  "uploading", "uploaded", "validating", "validated",
  "candidate", "challenger", "champion", "production",
  "retired", "archived", "rejected",
];

describe("Full promotion path (happy path)", () => {
  const happy: LifecycleStatus[] = [
    "uploading", "uploaded", "validating", "validated",
    "candidate", "challenger", "champion", "production", "retired", "archived",
  ];

  for (let i = 0; i < happy.length - 1; i++) {
    const from = happy[i];
    const to = happy[i + 1];
    it(`${from} → ${to} is valid`, () => {
      expect(isValidTransition(from, to)).toBe(true);
    });
  }
});

describe("Rejection is allowed from most non-terminal states", () => {
  const rejectable: LifecycleStatus[] = [
    "uploaded", "validating", "validated", "candidate", "challenger", "champion",
  ];
  for (const status of rejectable) {
    it(`${status} → rejected is valid`, () => {
      expect(isValidTransition(status, "rejected")).toBe(true);
    });
  }
});

describe("Terminal states cannot be promoted", () => {
  const terminals: LifecycleStatus[] = ["archived", "rejected"];
  for (const terminal of terminals) {
    for (const target of ALL_STATUSES) {
      if (target === terminal) continue;
      it(`${terminal} → ${target} is invalid`, () => {
        expect(isValidTransition(terminal, target)).toBe(false);
      });
    }
  }
});

describe("No skipping stages", () => {
  it("uploaded cannot skip directly to champion", () => {
    expect(isValidTransition("uploaded", "champion")).toBe(false);
  });

  it("candidate cannot skip directly to production", () => {
    expect(isValidTransition("candidate", "production")).toBe(false);
  });

  it("validated cannot skip to production", () => {
    expect(isValidTransition("validated", "production")).toBe(false);
  });
});
