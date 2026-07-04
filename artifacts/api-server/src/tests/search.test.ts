/**
 * Search service unit tests.
 *
 * The mock must match searchService's query patterns:
 *   - db.select().from(X)             → returns [] (no .where)
 *   - db.select().from(X).where(...)  → returns []
 */

import { describe, it, expect, vi } from "vitest";
import { eq } from "drizzle-orm";

// Mock DB before importing the service
vi.mock("@workspace/db", () => {
  function makeFromBuilder() {
    const fromResult = {
      // .where() chained after .from()
      where: vi.fn().mockResolvedValue([]),
      // Make .from() itself awaitable (used when .where() is NOT chained)
      then(
        onFulfilled: (v: unknown[]) => unknown,
        onRejected: (e: unknown) => unknown,
      ) {
        return Promise.resolve([]).then(onFulfilled, onRejected);
      },
    };
    return fromResult;
  }

  return {
    db: {
      select: vi.fn(() => ({
        from: vi.fn(() => makeFromBuilder()),
      })),
    },
    modelsTable: {},
    modelVersionsTable: {},
  };
});

import { searchRegistry } from "../services/searchService.js";

describe("searchRegistry pagination", () => {
  it("defaults to page 1, limit 20", async () => {
    const result = await searchRegistry({});
    expect(result.pagination.page).toBe(1);
    expect(result.pagination.limit).toBe(20);
  });

  it("enforces maximum limit of 100", async () => {
    const result = await searchRegistry({ limit: 500 });
    expect(result.pagination.limit).toBe(100);
  });

  it("enforces minimum page of 1", async () => {
    const result = await searchRegistry({ page: 0 });
    expect(result.pagination.page).toBe(1);
  });

  it("calculates totalPages correctly for empty results", async () => {
    const result = await searchRegistry({});
    expect(result.pagination.total).toBe(0);
    expect(result.pagination.totalPages).toBe(0);
  });
});

describe("searchRegistry filters (unit)", () => {
  it("accepts all valid filter keys without throwing", async () => {
    await expect(
      searchRegistry({
        architecture: "LSTM",
        status: "champion",
        datasetVersion: "2.1.0",
        featureVersion: "1.0.0",
        labelVersion: "1.0.0",
        experimentId: "exp-123",
        gitCommit: "abc1234",
        trainerVersion: "0.4.0",
        mlflowRunId: "run-456",
        createdAfter: "2024-01-01",
        createdBefore: "2025-01-01",
      }),
    ).resolves.toBeDefined();
  });
});
