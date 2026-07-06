import { type Server } from "node:http";
import app from "./app";
import { logger } from "./lib/logger";

const REQUIRED_ENV: Record<string, string> = {
  PORT: "TCP port the Express server binds to",
  DATABASE_URL: "PostgreSQL connection string",
  API_KEY: "Secret key clients must supply to access protected endpoints",
};

const missing: string[] = [];
for (const [key, description] of Object.entries(REQUIRED_ENV)) {
  if (!process.env[key]) {
    missing.push(`  ${key}: ${description}`);
  }
}
if (missing.length > 0) {
  console.error(
    `Missing required environment variables:\n${missing.join("\n")}`,
  );
  process.exit(1);
}

const port = Number(process.env["PORT"]);
if (Number.isNaN(port) || port <= 0) {
  console.error(`Invalid PORT value: "${process.env["PORT"]}"`);
  process.exit(1);
}

const server: Server = app.listen(port, () => {
  logger.info({ port }, "Server listening");
});

function shutdown(signal: string): void {
  logger.info({ signal }, "Shutdown signal received — draining connections");
  server.close(() => {
    logger.info("All connections closed — process exiting");
    process.exit(0);
  });
  setTimeout(() => {
    logger.error("Graceful shutdown timed out — forcing exit");
    process.exit(1);
  }, 10_000).unref();
}

process.on("SIGTERM", () => shutdown("SIGTERM"));
process.on("SIGINT", () => shutdown("SIGINT"));
