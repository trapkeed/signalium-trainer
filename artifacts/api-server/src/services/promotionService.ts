import { db, modelAliasesTable, deploymentHistoryTable, rollbackHistoryTable, type ModelLifecycleStatus } from "@workspace/db";
import { eq, and } from "drizzle-orm";
import { getVersionById, updateVersionStatus, listVersionsByModel } from "./registryService.js";
import { writeAuditLog } from "./auditService.js";

/** Valid lifecycle transitions (from → [valid targets]) */
const TRANSITIONS: Record<ModelLifecycleStatus, ModelLifecycleStatus[]> = {
  uploading:  ["uploaded"],
  uploaded:   ["validating", "rejected"],
  validating: ["validated", "rejected"],
  validated:  ["candidate", "rejected"],
  candidate:  ["challenger", "rejected"],
  challenger: ["champion", "rejected"],
  champion:   ["production", "retired", "rejected"],
  production: ["retired"],
  retired:    ["archived"],
  archived:   [],
  rejected:   [],
};

export function isValidTransition(from: ModelLifecycleStatus, to: ModelLifecycleStatus): boolean {
  return TRANSITIONS[from]?.includes(to) ?? false;
}

export async function promoteVersion(
  modelId: string,
  versionId: string,
  targetStatus: ModelLifecycleStatus,
  actor: string,
): Promise<{ success: boolean; error?: string; version?: Awaited<ReturnType<typeof getVersionById>> }> {
  const version = await getVersionById(versionId);
  if (!version || version.modelId !== modelId) {
    return { success: false, error: "Version not found." };
  }

  if (!isValidTransition(version.status, targetStatus)) {
    return {
      success: false,
      error: `Invalid transition: ${version.status} → ${targetStatus}. Allowed: ${(TRANSITIONS[version.status] ?? []).join(", ") || "none"}.`,
    };
  }

  const fromStatus = version.status;
  const updated = await updateVersionStatus(versionId, targetStatus, { promotedBy: actor });

  // Manage aliases for production-grade statuses
  if (["champion", "production"].includes(targetStatus)) {
    const aliasName = targetStatus; // "champion" or "production"
    await upsertAlias(modelId, versionId, aliasName, actor);

    if (targetStatus === "production") {
      await db.insert(deploymentHistoryTable).values({
        modelId,
        modelVersionId: versionId,
        environment: "production",
        status: "active",
        deployedBy: actor,
      });
    }
  }

  await writeAuditLog({
    eventType: "promoted",
    modelId,
    modelVersionId: versionId,
    actor,
    fromStatus,
    toStatus: targetStatus,
    details: { targetStatus },
  });

  return { success: true, version: updated };
}

export async function rejectVersion(
  modelId: string,
  versionId: string,
  reason: string,
  actor: string,
): Promise<{ success: boolean; error?: string }> {
  const version = await getVersionById(versionId);
  if (!version || version.modelId !== modelId) {
    return { success: false, error: "Version not found." };
  }

  const fromStatus = version.status;
  await updateVersionStatus(versionId, "rejected", { rejectionReason: reason });

  await writeAuditLog({
    eventType: "rejected",
    modelId,
    modelVersionId: versionId,
    actor,
    fromStatus,
    toStatus: "rejected",
    details: { reason },
  });

  return { success: true };
}

export async function rollbackProduction(
  modelId: string,
  toVersionId: string,
  reason: string,
  operator: string,
): Promise<{ success: boolean; error?: string }> {
  const toVersion = await getVersionById(toVersionId);
  if (!toVersion || toVersion.modelId !== modelId) {
    return { success: false, error: "Target version not found." };
  }

  // Find current production alias
  const [currentAlias] = await db
    .select()
    .from(modelAliasesTable)
    .where(and(eq(modelAliasesTable.modelId, modelId), eq(modelAliasesTable.aliasName, "production")));

  const fromVersionId = currentAlias?.modelVersionId;

  // Retire previous production version
  if (fromVersionId && fromVersionId !== toVersionId) {
    await db
      .update(deploymentHistoryTable)
      .set({ status: "rolled_back", retiredAt: new Date() })
      .where(
        and(
          eq(deploymentHistoryTable.modelId, modelId),
          eq(deploymentHistoryTable.modelVersionId, fromVersionId),
          eq(deploymentHistoryTable.status, "active"),
        ),
      );
  }

  // Point production alias to rollback target
  await upsertAlias(modelId, toVersionId, "production", operator);
  await updateVersionStatus(toVersionId, "production", { promotedBy: operator });

  // Create new deployment record
  await db.insert(deploymentHistoryTable).values({
    modelId,
    modelVersionId: toVersionId,
    environment: "production",
    status: "active",
    deployedBy: operator,
    notes: `Rollback: ${reason}`,
  });

  // Record the rollback
  if (fromVersionId) {
    await db.insert(rollbackHistoryTable).values({
      modelId,
      fromVersionId,
      toVersionId,
      reason,
      operator,
    });
  }

  await writeAuditLog({
    eventType: "rolled_back",
    modelId,
    modelVersionId: toVersionId,
    actor: operator,
    details: { fromVersionId, toVersionId, reason },
  });

  return { success: true };
}

async function upsertAlias(modelId: string, versionId: string, aliasName: string, actor: string) {
  const [existing] = await db
    .select()
    .from(modelAliasesTable)
    .where(and(eq(modelAliasesTable.modelId, modelId), eq(modelAliasesTable.aliasName, aliasName)));

  if (existing) {
    await db
      .update(modelAliasesTable)
      .set({ modelVersionId: versionId, setBy: actor, updatedAt: new Date() })
      .where(eq(modelAliasesTable.id, existing.id));
  } else {
    await db.insert(modelAliasesTable).values({ modelId, modelVersionId: versionId, aliasName, setBy: actor });
  }
}

export async function getAlias(modelId: string, aliasName: string) {
  const [alias] = await db
    .select()
    .from(modelAliasesTable)
    .where(and(eq(modelAliasesTable.modelId, modelId), eq(modelAliasesTable.aliasName, aliasName)));
  return alias ?? null;
}

export async function listAliases(modelId: string) {
  return db.select().from(modelAliasesTable).where(eq(modelAliasesTable.modelId, modelId));
}

export async function getDeploymentHistory(modelId: string) {
  return db
    .select()
    .from(deploymentHistoryTable)
    .where(eq(deploymentHistoryTable.modelId, modelId));
}

export async function getRollbackHistory(modelId: string) {
  return db
    .select()
    .from(rollbackHistoryTable)
    .where(eq(rollbackHistoryTable.modelId, modelId));
}
