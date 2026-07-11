import express, { type Express } from "express";
import cors from "cors";
import pinoHttp from "pino-http";
import router from "./routes";
import platformRouter from "./routes/platform";
import { logger } from "./lib/logger";
import { platformCorsOptions } from "./middlewares/platform_trust";

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
app.use(cors(platformCorsOptions()));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Platform contract: GET /health — no auth, returns { status: "ok" }
// Must be registered before the /api sub-router.
app.get("/health", (_req, res) => {
  res.status(200).json({ status: "ok" });
});

app.use("/api", router);

// Platform-facing routes — JWT-authenticated, mounted at root.
// Inference and Research Lab contract routes are added by tasks #11 and #12.
app.use(platformRouter);

export default app;
