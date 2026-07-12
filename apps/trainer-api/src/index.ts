import "dotenv/config";
import { spawn } from "child_process";
import app from "./app";
import { logger } from "./lib/logger";

// ── Research Lab sidecar ──────────────────────────────────────────────────────
// The Python Research Lab FastAPI server runs on a private port alongside the
// Node API.  Auto-start it in production, or when explicitly enabled for local
// testing via RESEARCH_LAB_AUTOSTART=true.
const _shouldStartRL =
  process.env["NODE_ENV"] === "production" ||
  process.env["RESEARCH_LAB_AUTOSTART"] === "true";

if (_shouldStartRL) {
  const rlPort = process.env["RESEARCH_LAB_PORT"] ?? "8002";
  // Uvicorn must run from the project root so Python can resolve the `trainer`
  // package (it is not installed into site-packages; it is loaded from the
  // source tree).  The compiled file lives at:
  //   <root>/apps/trainer-api/dist/index.mjs
  // so three levels up is always the project root, in any environment.
  // Override with RESEARCH_LAB_ROOT if your deployment layout differs.
  const projectRoot =
    process.env["RESEARCH_LAB_ROOT"] ??
    new URL("../../..", import.meta.url).pathname;

  const rl = spawn(
    "uvicorn",
    [
      "trainer.research_lab.research_lab_server:app",
      "--host",
      "127.0.0.1",
      "--port",
      rlPort,
      "--workers",
      "1",
    ],
    { stdio: "inherit", env: process.env, cwd: projectRoot },
  );
  rl.on("error", (err) =>
    logger.error({ err }, "Research Lab server failed to start"),
  );
  rl.on("exit", (code, signal) => {
    if (code !== 0 && signal !== "SIGTERM") {
      logger.error({ code, signal }, "Research Lab server exited unexpectedly");
    }
  });
  logger.info({ port: rlPort }, "Research Lab server starting");
}

const rawPort = process.env["PORT"];

if (!rawPort) {
  throw new Error(
    "PORT environment variable is required but was not provided.",
  );
}

const port = Number(rawPort);

if (Number.isNaN(port) || port <= 0) {
  throw new Error(`Invalid PORT value: "${rawPort}"`);
}

app.listen(port, (err) => {
  if (err) {
    logger.error({ err }, "Error listening on port");
    process.exit(1);
  }

  logger.info({ port }, "Server listening");
});
