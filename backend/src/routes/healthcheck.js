import express from "express";
import { gpt4oHealthCheck } from "../controllers/healthcheckController.js";

const router = express.Router();

// Route health check cho GPT-4o service
router.get("/gpt4o", gpt4oHealthCheck);

export default router;