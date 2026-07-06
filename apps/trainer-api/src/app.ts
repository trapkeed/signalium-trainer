import express, {
  type Express,
  type Request,
  type Response,
  type NextFunction,
} from "express";
import cors from "cors";
import helmet from "helmet";
import pinoHttp from "pino-http";
import router from "./routes";
import { logger } from "./lib/logger";

const app: Express = express();

app.use(helmet());

const rawOrigin = process.env.CORS_ORIGIN;
let corsOrigin: cors.CorsOptions["origin"];
if (!rawOrigin) {
  corsOrigin = false;
} else if (rawOrigin === "*") {
  corsOrigin = "*";
} else {
  const origins = rawOrigin.split(",").map((o) => o.trim());
  corsOrigin = origins.length === 1 ? origins[0] : origins;
}

app.use(cors({ origin: corsOrigin, credentials: true }));

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

const bodyLimit = process.env.BODY_LIMIT ?? "1mb";
app.use(express.json({ limit: bodyLimit }));
app.use(express.urlencoded({ extended: true, limit: bodyLimit }));

app.use("/api", router);

app.use((_req: Request, res: Response) => {
  res.status(404).json({
    status: "error",
    code: "NOT_FOUND",
    message: "Route not found",
  });
});

app.use((err: unknown, _req: Request, res: Response, _next: NextFunction) => {
  if (err instanceof Error) {
    logger.error({ err }, "Unhandled error");
    const httpErr = err as Error & { status?: number; statusCode?: number; type?: string };
    const status = httpErr.status ?? httpErr.statusCode ?? 500;
    const isClientError = status >= 400 && status < 500;
    res.status(status).json({
      status: "error",
      code: isClientError ? (httpErr.type ?? "CLIENT_ERROR") : "INTERNAL_SERVER_ERROR",
      message:
        isClientError || process.env.NODE_ENV !== "production"
          ? err.message
          : "An unexpected error occurred",
    });
  } else {
    logger.error({ err }, "Unknown error type thrown");
    res.status(500).json({
      status: "error",
      code: "INTERNAL_SERVER_ERROR",
      message: "An unexpected error occurred",
    });
  }
});

export default app;
