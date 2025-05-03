import express from "express";
import multer from "multer";
import { analyzeXray } from "../controllers/imageControllers.js";

const upload = multer({ dest: "uploads/" });

const router = express.Router();

router.post("/analyze", upload.single("image"), analyzeXray);

export default router;
