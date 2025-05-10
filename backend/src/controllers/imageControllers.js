import { analyzeXrayImage } from "../services/onnxService.js";

/**
 * API xử lý ảnh X-ray và trả kết quả phân tích
 * @param {object} req - Request object
 * @param {object} res - Response object
 */
export async function analyzeXray(req, res) {
  try {
    const imagePath = req.file?.path;
    if (!imagePath) {
      return res.status(400).json({ error: "Không tìm thấy file ảnh!" });
    }

    console.log(`File ảnh đã upload: ${imagePath}`);

    const result = await analyzeXrayImage(imagePath);

    res.json(result);
  } catch (err) {
    console.error("Lỗi xử lý phân tích ảnh:", err);
    res.status(500).json({ error: "Đã xảy ra lỗi khi phân tích ảnh X-ray!" });
  }
}
