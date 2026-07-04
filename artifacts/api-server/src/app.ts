import express, { type Express } from "express";
import cors from "cors";
import pinoHttp from "pino-http";
import router from "./routes/index.js";
import { logger } from "./lib/logger.js";
import { startCleanupWorker } from "./workers/cleanupWorker.js";

// Register worker task handlers on startup
import "./workers/checksumWorker.js";
import "./workers/validationWorker.js";

const app: Express = express();

app.use(
  pinoHttp({
    logger,
    serializers: {
      req(req) {
        return {
          id: req.id,
          method: req.method,
          url: req.url?.split("?")[0],
        };
      },
      res(res) {
        return {
          statusCode: res.statusCode,
        };
      },
    },
  }),
);
app.use(cors());

// express.json() only fires when Content-Type is application/json.
// Binary artifact uploads (application/octet-stream) bypass it safely —
// the artifact route reads req as a raw stream directly.
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true }));

app.use("/api", router);

// Start background workers (non-blocking)
startCleanupWorker();

export default app;
