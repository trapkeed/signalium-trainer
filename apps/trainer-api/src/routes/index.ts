import { Router, type IRouter } from "express";
import healthRouter from "./health";
import qdeRouter from "./qde";
import researchLabRouter from "./research_lab";

const router: IRouter = Router();

router.use(healthRouter);
router.use(qdeRouter);
router.use(researchLabRouter);

export default router;
