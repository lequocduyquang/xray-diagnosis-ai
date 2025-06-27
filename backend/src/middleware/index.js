import path from "path";
import fs from "fs/promises";
import { v2 as cloudinary } from "cloudinary";
import { dicomToPng } from "../utils/imageProcessing.js";

/**
 * Middleware xử lý file DICOM và upload lên Cloudinary
 * @param {object} req - Request object
 * @param {object} res - Response object  
 * @param {function} next - Next middleware function
 */
export const handleDicomFile = async (req, res, next) => {
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
      req.file.cloudinaryUrl = uploadResult.secure_url; // Add cloudinaryUrl
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
      req.file.cloudinaryUrl = uploadResult.secure_url; // Add cloudinaryUrl
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

/**
 * Middleware kiểm tra OpenAI API key (cho 3-AI system)  
 * @param {object} req - Request object
 * @param {object} res - Response object
 * @param {function} next - Next middleware function
 */
export const validateOpenAIKey = (req, res, next) => {
  // 🚀 UPDATED: 3-AI system requires OpenAI API key for ALL analyze endpoints
  const isAnalyzeEndpoint = req.originalUrl.includes('/analyze');
  const isGPT4oEndpoint = req.originalUrl.includes('gpt4o') || req.originalUrl.includes('second-opinion');

  // Check if this endpoint requires OpenAI API key
  if (isAnalyzeEndpoint || isGPT4oEndpoint) {
    if (!process.env.OPENAI_API_KEY) {
      return res.status(400).json({
        error: "OPENAI_API_KEY không được cấu hình! Hệ thống 3-AI cần GPT-4o để hoạt động.",
        suggestion: "Thêm OPENAI_API_KEY=your_api_key_here vào file .env",
        endpoint_info: {
          endpoint: req.originalUrl,
          requires_openai: true,
          reason: isAnalyzeEndpoint ? "3-AI Hybrid System" : "GPT-4o specific endpoint"
        }
      });
    }

    console.log(`✅ OpenAI API key validated for endpoint: ${req.originalUrl}`);
  }

  next();
};

/**
 * Middleware validation cho clinical_info
 * @param {object} req - Request object
 * @param {object} res - Response object
 * @param {function} next - Next middleware function
 */
export const validateClinicalInfo = (req, res, next) => {
  if (req.body.clinical_info) {
    try {
      const clinicalInfo = JSON.parse(req.body.clinical_info);

      // Validate structure
      if (clinicalInfo.symptoms && !Array.isArray(clinicalInfo.symptoms)) {
        return res.status(400).json({
          error: "clinical_info.symptoms phải là một mảng (array) các chuỗi!"
        });
      }

      // Store parsed clinical_info back to req.body
      req.body.parsed_clinical_info = clinicalInfo;

      console.log(`✅ Clinical info validated:`, clinicalInfo);
    } catch (err) {
      return res.status(400).json({
        error: "clinical_info phải là JSON hợp lệ!",
        details: err.message
      });
    }
  }

  next();
}; 