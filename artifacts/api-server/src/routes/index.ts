import { Router, type IRouter } from "express";
import healthRouter from "./health.js";
import registryRouter from "./registry/index.js";

const router: IRouter = Router();

router.use(healthRouter);
router.use("/registry", registryRouter);

export default router;
