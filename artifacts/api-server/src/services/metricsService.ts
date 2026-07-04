import { db, modelMetricsTable } from "@workspace/db";
import { eq } from "drizzle-orm";
import type { InsertModelMetric } from "@workspace/db";

export async function recordMetrics(entries: InsertModelMetric[]) {
  if (entries.length === 0) return [];
  return db.insert(modelMetricsTable).values(entries).returning();
}

export async function getMetrics(modelVersionId: string) {
  const rows = await db
    .select()
    .from(modelMetricsTable)
    .where(eq(modelMetricsTable.modelVersionId, modelVersionId));

  // Group by category
  const grouped: Record<string, Record<string, number>> = {};
  for (const row of rows) {
    grouped[row.category] ??= {};
    grouped[row.category][row.metricKey] = row.metricValue;
  }
  return { raw: rows, grouped };
}
