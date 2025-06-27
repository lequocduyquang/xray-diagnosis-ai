import express from "express";
import multer from "multer";
import { v2 as cloudinary } from "cloudinary";
import dotenv from "dotenv";
import {
  analyzeXray,
  analyzeXrayGPT4oOnly,
} from "../controllers/imageControllers.js";
import { dicomToPng } from "../utils/imageProcessing.js";
import { getImageByCloudinaryId } from "../services/databaseService.js";
import fs from "fs/promises";
import path from "path";
import { uploadsDir } from "../index.js";

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

// Middleware xử lý file DICOM
const handleDicomFile = async (req, res, next) => {
  if (!req.file) {
    console.log("Không tìm thấy file để xử lý.");
    return res.status(400).json({ error: "Không tìm thấy file để upload!" });
  }

  const filePath = req.file.path; // Đường dẫn file tạm thời
  const fileExtension = path.extname(filePath).toLowerCase();

  console.log(`Đang kiểm tra file: ${filePath}`);
  console.log(`Phần mở rộng file: ${fileExtension}`);

  if (fileExtension === ".dcm" || fileExtension === ".dicom") {
    try {
      console.log("Đang xử lý file DICOM...");
      // Chuyển đổi DICOM sang PNG
      const convertedPath = await dicomToPng(filePath);

      console.log(`Đã chuyển DICOM sang PNG: ${convertedPath}`);

      // Upload file PNG đã chuyển đổi lên Cloudinary
      const uploadResult = await cloudinary.uploader.upload(convertedPath, {
        folder: "xray-images",
        use_filename: true,
        unique_filename: false,
        resource_type: "image", // Đảm bảo Cloudinary xử lý file PNG như ảnh
      });

      console.log(`Đã upload PNG lên Cloudinary: ${uploadResult.secure_url}`);

      // Cập nhật thông tin file trong req.file
      req.file.path = uploadResult.secure_url; // URL của file trên Cloudinary
      req.file.mimetype = "image/png"; // MIME type của file PNG
      req.file.cloudinaryId = uploadResult.public_id; // Thêm cloudinary_id

      // Xóa file PNG tạm thời sau khi upload
      await fs.unlink(convertedPath);
      // Xóa file DICOM tạm thời
      await fs.unlink(filePath);
    } catch (error) {
      console.error(`Lỗi khi xử lý file DICOM: ${JSON.stringify(error)}`);
      return res.status(500).json({ error: "Lỗi khi xử lý file DICOM!" });
    }
  } else {
    try {
      console.log(
        "File không phải là DICOM, upload trực tiếp lên Cloudinary..."
      );
      // Upload file PNG/JPEG trực tiếp lên Cloudinary
      const uploadResult = await cloudinary.uploader.upload(filePath, {
        folder: "xray-images",
        use_filename: true,
        unique_filename: false,
        resource_type: "image", // Đảm bảo Cloudinary xử lý file PNG/JPEG như ảnh
      });

      console.log(`Đã upload file lên Cloudinary: ${uploadResult.secure_url}`);

      // Cập nhật thông tin file trong req.file
      req.file.path = uploadResult.secure_url; // URL của file trên Cloudinary
      req.file.cloudinaryId = uploadResult.public_id; // Thêm cloudinary_id

      // Xóa file tạm thời sau khi upload
      await fs.unlink(filePath);
    } catch (error) {
      console.error(`Lỗi khi upload file: ${JSON.stringify(error)}`);
      return res.status(500).json({ error: "Lỗi khi upload file!" });
    }
  }

  next();
};

// Middleware kiểm tra OpenAI API key (chỉ cho các endpoint cần GPT-4o)
const validateOpenAIKey = (req, res, next) => {
  const enableGPT4o = req.query.enable_gpt4o === 'true' ||
    req.body.enable_gpt4o === true ||
    req.body.enable_gpt4o === 'true';

  // Chỉ kiểm tra nếu GPT-4o được enable hoặc là endpoint GPT-4o only
  if (enableGPT4o || req.originalUrl.includes('gpt4o')) {
    if (!process.env.OPENAI_API_KEY) {
      return res.status(400).json({
        error: "OPENAI_API_KEY không được cấu hình! Vui lòng thêm vào file .env",
        suggestion: "Thêm OPENAI_API_KEY=your_api_key_here vào file .env"
      });
    }
  }
  next();
};

// ==================== MAIN ROUTES ====================

// Route phân tích ảnh X-ray (ONNX models + GPT-4o tùy chọn)
router.post("/analyze", upload.single("image"), handleDicomFile, validateOpenAIKey, analyzeXray);

// Route chỉ chạy GPT-4o analysis (cho testing)
router.post("/gpt4o-only", upload.single("image"), handleDicomFile, validateOpenAIKey, analyzeXrayGPT4oOnly);

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
