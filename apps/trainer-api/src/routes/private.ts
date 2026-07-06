import { Router, type IRouter, type Request, type Response } from "express";

const router: IRouter = Router();

router.get("/v1/status", (_req: Request, res: Response) => {
  res.json({
    status: "ok",
    uptime: process.uptime(),
    timestamp: new Date().toISOString(),
    environment: process.env.NODE_ENV ?? "development",
  });
});

export default router;
