import { analyzeXrayImage } from "../services/onnxService.js";

/**
 * API xử lý ảnh X-ray và trả kết quả phân tích
 * @param {object} req - Request object
 * @param {object} res - Response object
 */
export async function analyzeXray(req, res) {
  try {
    const dicomFilePath = req.body.dicomFilePath; // Đường dẫn file DICOM từ client
    const result = await analyzeXrayImage(dicomFilePath);
    res.json(result);
  } catch (err) {
    res.status(500).json({ error: "Đã xảy ra lỗi khi phân tích ảnh X-ray!" });
  }
}
