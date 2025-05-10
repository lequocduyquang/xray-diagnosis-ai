import express from "express";
import multer from "multer";
import { v2 as cloudinary } from "cloudinary";
import dotenv from "dotenv";
import { analyzeXray } from "../controllers/imageControllers.js";
import { dicomToPng } from "../utils/imageProcessing.js";
import fs from "fs/promises";
import path from "path";

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
    cb(null, "uploads/"); // Lưu file tạm thời vào thư mục "uploads"
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

  if (fileExtension === ".dcm") {
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

      // Xóa file tạm thời sau khi upload
      await fs.unlink(filePath);
    } catch (error) {
      console.error(`Lỗi khi upload file: ${JSON.stringify(error)}`);
      return res.status(500).json({ error: "Lỗi khi upload file!" });
    }
  }

  next();
};

// Route phân tích ảnh X-ray
router.post("/analyze", upload.single("image"), handleDicomFile, analyzeXray);

export default router;
