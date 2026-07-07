import { Router, type IRouter } from "express";
import healthRouter from "./health";
import qdeRouter from "./qde";

const router: IRouter = Router();

router.use(healthRouter);
router.use(qdeRouter);

export default router;
