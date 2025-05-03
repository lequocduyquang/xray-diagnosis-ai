import express from "express";
import { analyzeXray } from "../controllers/imageControllers";

const router = express.Router();

router.post("/analyze", analyzeXray);

export default router;
