import { db, modelArtifactsTable, type ArtifactType } from "@workspace/db";
import { eq, and } from "drizzle-orm";
import { createHash } from "crypto";
import { createWriteStream, createReadStream, mkdirSync, statSync, existsSync } from "fs";
import { join } from "path";
import type { Request } from "express";

const STORAGE_ROOT = process.env.ARTIFACT_STORAGE_PATH ?? "./data/artifacts";

export function ensureStorageDir(subdir: string): string {
  const dir = join(STORAGE_ROOT, subdir);
  mkdirSync(dir, { recursive: true });
  return dir;
}

export function artifactStoragePath(versionId: string, filename: string): string {
  const dir = ensureStorageDir(versionId);
  return join(dir, filename);
}

/** Stream request body to disk and compute SHA-256 checksum simultaneously. */
export async function streamArtifactToDisk(
  req: Request,
  storagePath: string,
): Promise<{ sizeBytes: number; checksum: string }> {
  return new Promise((resolve, reject) => {
    const hash = createHash("sha256");
    const ws = createWriteStream(storagePath);
    let sizeBytes = 0;

    req.on("data", (chunk: Buffer) => {
      sizeBytes += chunk.length;
      hash.update(chunk);
    });

    req.pipe(ws);
    ws.on("finish", () => resolve({ sizeBytes, checksum: hash.digest("hex") }));
    ws.on("error", reject);
    req.on("error", reject);
  });
}

export async function createArtifactRecord(data: {
  modelVersionId: string;
  artifactType: ArtifactType;
  filename: string;
  storagePath: string;
  mimeType: string;
  sizeBytes: number;
  checksum: string;
  uploadedBy: string;
}) {
  const [artifact] = await db
    .insert(modelArtifactsTable)
    .values({ ...data, status: "uploaded", checksumAlgorithm: "sha256" })
    .returning();
  return artifact;
}

export async function listArtifacts(modelVersionId: string) {
  return db
    .select()
    .from(modelArtifactsTable)
    .where(eq(modelArtifactsTable.modelVersionId, modelVersionId));
}

export async function getArtifactById(artifactId: string) {
  const [a] = await db
    .select()
    .from(modelArtifactsTable)
    .where(eq(modelArtifactsTable.id, artifactId));
  return a ?? null;
}

export async function verifyArtifactChecksum(artifactId: string): Promise<boolean> {
  const artifact = await getArtifactById(artifactId);
  if (!artifact || !artifact.checksum) return false;

  if (!existsSync(artifact.storagePath)) return false;

  const hash = createHash("sha256");
  const stream = createReadStream(artifact.storagePath);

  return new Promise((resolve, reject) => {
    stream.on("data", (chunk) => hash.update(chunk));
    stream.on("end", async () => {
      const computed = hash.digest("hex");
      const ok = computed === artifact.checksum;
      await db
        .update(modelArtifactsTable)
        .set({ status: ok ? "verified" : "corrupted", updatedAt: new Date() })
        .where(eq(modelArtifactsTable.id, artifactId));
      resolve(ok);
    });
    stream.on("error", reject);
  });
}

export function getArtifactFileSize(storagePath: string): number {
  try {
    return statSync(storagePath).size;
  } catch {
    return 0;
  }
}
