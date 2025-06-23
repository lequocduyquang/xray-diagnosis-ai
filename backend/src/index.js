import express from "express";
import dotenv from "dotenv";
import cors from "cors";
import imageRoutes from "./routes/imageRoutes.js";
import { initializeDatabase } from "./config/database.js";
import fs from "fs";
import { fileURLToPath } from "url";
import path from "path";
import { ensureModelDownloaded } from "./services/huggingfaceService.js";

dotenv.config();

const app = express();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export const uploadsDir = path.join(__dirname, "../uploads");

if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true });
}

app.use(cors());

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.get("/health", (req, res) => {
  res.status(200).json({
    status: "OK",
    message: "Server is healthy",
    timestamp: new Date().toISOString(),
  });
});

app.use("/api", imageRoutes);

const PORT = process.env.PORT || 5000;

// Khởi tạo database và khởi động server
async function startServer() {
  try {
    // Khởi tạo database
    await initializeDatabase();

    // Đảm bảo models đã được tải về
    // await ensureModelDownloaded("resnetV1");
    // await ensureModelDownloaded("resnetV2");
    // await ensureModelDownloaded("densenet");

    // Khởi động server
    app.listen(PORT, () => {
      console.log(`🚀 Server đang chạy tại http://localhost:${PORT}`);
      console.log(`📊 Database đã sẵn sàng`);
    });
  } catch (error) {
    console.error("❌ Lỗi khởi động server:", error);
    process.exit(1);
  }
}

startServer();
