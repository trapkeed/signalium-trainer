import { registerHandler, enqueue } from "./queue.js";
import { verifyArtifactChecksum } from "../services/artifactService.js";
import { writeAuditLog } from "../services/auditService.js";

const TASK_TYPE = "verify_checksum";

registerHandler(TASK_TYPE, async (task) => {
  const { artifactId } = task.payload as { artifactId: string };
  const ok = await verifyArtifactChecksum(artifactId);
  if (!ok) {
    await writeAuditLog({
      eventType: "validation_failed",
      actor: "system",
      details: { artifactId, reason: "checksum_mismatch" },
    });
  }
});

export function enqueueChecksumVerification(artifactId: string): void {
  enqueue(TASK_TYPE, { artifactId });
}
