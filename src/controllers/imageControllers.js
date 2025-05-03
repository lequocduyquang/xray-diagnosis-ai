import { analyzeXrayImage } from "../services/onnxService.js";

/**
 * API xử lý ảnh X-ray và trả kết quả phân tích
 * @param {object} req - Request object
 * @param {object} res - Response object
 */
export async function analyzeXray(req, res) {
  try {
    // Kiểm tra và lấy đường dẫn file DICOM từ request
    const dicomFilePath = req.file?.path;
    if (!dicomFilePath) {
      return res.status(400).json({ error: "Không tìm thấy file DICOM!" });
    }

    console.log(`File DICOM đã upload: ${dicomFilePath}`);

    // Phân tích ảnh X-ray
    const result = await analyzeXrayImage(dicomFilePath);

    // Trả về kết quả phân tích
    res.json(result);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Đã xảy ra lỗi khi phân tích ảnh X-ray!" });
  }
}
