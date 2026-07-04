import { registerHandler, enqueue } from "./queue.js";
import { runValidation } from "../services/validationService.js";
import { logger } from "../lib/logger.js";

const TASK_TYPE = "validate_model_version";

registerHandler(TASK_TYPE, async (task) => {
  const { modelVersionId } = task.payload as { modelVersionId: string };
  logger.info({ modelVersionId }, "Starting background validation");
  const { passed, report } = await runValidation(modelVersionId, "system");
  logger.info({ modelVersionId, passed, durationMs: report.duration_ms }, "Validation complete");
});

export function enqueueValidation(modelVersionId: string): void {
  enqueue(TASK_TYPE, { modelVersionId });
}
