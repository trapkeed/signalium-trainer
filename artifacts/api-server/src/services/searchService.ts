import { db, modelsTable, modelVersionsTable, type ModelLifecycleStatus } from "@workspace/db";
import { eq, and, gte, lte, like } from "drizzle-orm";

export interface SearchFilters {
  architecture?: string;
  status?: ModelLifecycleStatus;
  datasetVersion?: string;
  featureVersion?: string;
  labelVersion?: string;
  experimentId?: string;
  gitCommit?: string;
  trainerVersion?: string;
  mlflowRunId?: string;
  createdAfter?: string;
  createdBefore?: string;
  page?: number;
  limit?: number;
}

export async function searchRegistry(filters: SearchFilters) {
  const page = Math.max(1, filters.page ?? 1);
  const limit = Math.min(100, Math.max(1, filters.limit ?? 20));
  const offset = (page - 1) * limit;

  // Build version conditions
  let versions = await db.select().from(modelVersionsTable);

  if (filters.status) {
    versions = versions.filter((v) => v.status === filters.status);
  }
  if (filters.architecture) {
    versions = versions.filter((v) => v.architecture === filters.architecture);
  }
  if (filters.datasetVersion) {
    versions = versions.filter((v) => v.datasetVersion === filters.datasetVersion);
  }
  if (filters.featureVersion) {
    versions = versions.filter((v) => v.featureVersion === filters.featureVersion);
  }
  if (filters.labelVersion) {
    versions = versions.filter((v) => v.labelVersion === filters.labelVersion);
  }
  if (filters.experimentId) {
    versions = versions.filter((v) => v.experimentId === filters.experimentId);
  }
  if (filters.gitCommit) {
    versions = versions.filter((v) =>
      v.gitCommitHash?.startsWith(filters.gitCommit!) ?? false,
    );
  }
  if (filters.trainerVersion) {
    versions = versions.filter((v) => v.trainerVersion === filters.trainerVersion);
  }
  if (filters.mlflowRunId) {
    versions = versions.filter((v) => v.mlflowRunId === filters.mlflowRunId);
  }
  if (filters.createdAfter) {
    const after = new Date(filters.createdAfter);
    versions = versions.filter((v) => v.createdAt >= after);
  }
  if (filters.createdBefore) {
    const before = new Date(filters.createdBefore);
    versions = versions.filter((v) => v.createdAt <= before);
  }

  const total = versions.length;
  const paginated = versions.slice(offset, offset + limit);

  // Enrich with model info
  const models = await db
    .select()
    .from(modelsTable)
    .where(eq(modelsTable.isDeleted, false));
  const modelMap = new Map(models.map((m) => [m.id, m]));

  const enriched = paginated.map((v) => ({
    ...v,
    model: modelMap.get(v.modelId) ?? null,
  }));

  return {
    results: enriched,
    pagination: {
      page,
      limit,
      total,
      totalPages: Math.ceil(total / limit),
    },
  };
}

export async function getRegistryStats() {
  const models = await db.select().from(modelsTable).where(eq(modelsTable.isDeleted, false));
  const versions = await db.select().from(modelVersionsTable);

  const byStatus: Record<string, number> = {};
  for (const v of versions) {
    byStatus[v.status] = (byStatus[v.status] ?? 0) + 1;
  }

  return {
    total_models: models.length,
    total_versions: versions.length,
    versions_by_status: byStatus,
    champion_count: byStatus["champion"] ?? 0,
    production_count: byStatus["production"] ?? 0,
  };
}
