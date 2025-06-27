import express from "express";
import multer from "multer";
import { v2 as cloudinary } from "cloudinary";
import dotenv from "dotenv";
import {
  analyzeXray,
} from "../controllers/imageControllers.js";
import { getImageByCloudinaryId } from "../services/databaseService.js";
import { uploadsDir } from "../index.js";
import { handleDicomFile, validateOpenAIKey, validateClinicalInfo } from "../middleware/index.js";

dotenv.config();

// Cấu hình Cloudinary
cloudinary.config({
  cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
  api_key: process.env.CLOUDINARY_API_KEY,
  api_secret: process.env.CLOUDINARY_API_SECRET,
});

// Cấu hình Multer
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadsDir); // Lưu file tạm thời vào thư mục "uploads"
  },
  filename: (req, file, cb) => {
    cb(null, `${Date.now()}-${file.originalname}`); // Đặt tên file tạm thời
  },
});

const upload = multer({ storage });

const router = express.Router();



// ==================== MAIN ROUTES ====================

// Route phân tích ảnh X-ray (3-AI Hybrid System)
router.post("/analyze",
  upload.single("image"),
  handleDicomFile,
  validateOpenAIKey,
  validateClinicalInfo,
  analyzeXray
);

// Route lấy thông tin ảnh theo cloudinary_id
router.get("/image/:cloudinaryId", async (req, res) => {
  try {
    const { cloudinaryId } = req.params;

    if (!cloudinaryId) {
      return res.status(400).json({
        success: false,
        error: "cloudinary_id là bắt buộc",
      });
    }

    const image = await getImageByCloudinaryId(`xray-images/${cloudinaryId}`);

    if (!image) {
      return res.status(404).json({
        success: false,
        error: "Không tìm thấy ảnh với cloudinary_id này",
      });
    }

    res.json({
      success: true,
      data: {
        id: image.id,
        cloudinary_id: image.cloudinary_id,
        cloudinary_url: image.cloudinary_url,
        model_name: image.model_name,
        created_at: image.created_at,
        updated_at: image.updated_at,
      },
    });
  } catch (error) {
    console.error("Lỗi khi lấy thông tin ảnh:", error);
    res.status(500).json({
      success: false,
      error: "Lỗi khi lấy thông tin ảnh từ database",
    });
  }
});

export default router;
