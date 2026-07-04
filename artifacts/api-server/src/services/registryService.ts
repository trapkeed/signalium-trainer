import { db, modelsTable, modelVersionsTable, modelTagsTable, type InsertModel, type InsertModelVersion, type ModelLifecycleStatus } from "@workspace/db";
import { eq, and } from "drizzle-orm";

// ─── Models ───────────────────────────────────────────────────────────────────

export async function createModel(data: InsertModel) {
  const [model] = await db.insert(modelsTable).values(data).returning();
  return model;
}

export async function getModelById(id: string) {
  const [model] = await db
    .select()
    .from(modelsTable)
    .where(and(eq(modelsTable.id, id), eq(modelsTable.isDeleted, false)));
  return model ?? null;
}

export async function getModelByName(name: string) {
  const [model] = await db
    .select()
    .from(modelsTable)
    .where(and(eq(modelsTable.name, name), eq(modelsTable.isDeleted, false)));
  return model ?? null;
}

export async function listModels(filters: { architecture?: string; name?: string } = {}) {
  const rows = await db.select().from(modelsTable).where(eq(modelsTable.isDeleted, false));
  let result = rows;
  if (filters.architecture) {
    result = result.filter((m) => m.architecture === filters.architecture);
  }
  if (filters.name) {
    result = result.filter((m) => m.name.includes(filters.name!));
  }
  return result;
}

export async function softDeleteModel(id: string) {
  const [updated] = await db
    .update(modelsTable)
    .set({ isDeleted: true, deletedAt: new Date() })
    .where(eq(modelsTable.id, id))
    .returning();
  return updated ?? null;
}

// ─── Versions ─────────────────────────────────────────────────────────────────

export async function createModelVersion(data: InsertModelVersion) {
  const [version] = await db.insert(modelVersionsTable).values(data).returning();
  return version;
}

export async function getVersionById(versionId: string) {
  const [v] = await db
    .select()
    .from(modelVersionsTable)
    .where(eq(modelVersionsTable.id, versionId));
  return v ?? null;
}

export async function getVersionByModelAndId(modelId: string, versionId: string) {
  const [v] = await db
    .select()
    .from(modelVersionsTable)
    .where(
      and(
        eq(modelVersionsTable.modelId, modelId),
        eq(modelVersionsTable.id, versionId),
      ),
    );
  return v ?? null;
}

export async function listVersionsByModel(modelId: string) {
  return db
    .select()
    .from(modelVersionsTable)
    .where(eq(modelVersionsTable.modelId, modelId));
}

export async function updateVersionStatus(
  versionId: string,
  status: ModelLifecycleStatus,
  extra: { promotedBy?: string; rejectionReason?: string } = {},
) {
  const [updated] = await db
    .update(modelVersionsTable)
    .set({
      status,
      updatedAt: new Date(),
      ...(extra.promotedBy !== undefined ? { promotedBy: extra.promotedBy, promotedAt: new Date() } : {}),
      ...(extra.rejectionReason !== undefined ? { rejectionReason: extra.rejectionReason } : {}),
    })
    .where(eq(modelVersionsTable.id, versionId))
    .returning();
  return updated ?? null;
}

// ─── Tags ─────────────────────────────────────────────────────────────────────

export async function addTag(modelId: string, key: string, value: string) {
  const [tag] = await db
    .insert(modelTagsTable)
    .values({ modelId, tagKey: key, tagValue: value })
    .returning();
  return tag;
}

export async function listTags(modelId: string) {
  return db.select().from(modelTagsTable).where(eq(modelTagsTable.modelId, modelId));
}

export async function removeTag(tagId: string) {
  const [deleted] = await db
    .delete(modelTagsTable)
    .where(eq(modelTagsTable.id, tagId))
    .returning();
  return deleted ?? null;
}
