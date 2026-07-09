import { Router, type IRouter } from "express";
import healthRouter from "./health";
import qdeRouter from "./qde";
import researchLabRouter from "./research_lab";
import capabilitiesRouter from "./capabilities";

const router: IRouter = Router();

router.use(healthRouter);
router.use(qdeRouter);
router.use(researchLabRouter);
router.use(capabilitiesRouter);

export default router;
