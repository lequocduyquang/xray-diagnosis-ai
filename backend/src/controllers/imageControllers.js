import { analyzeXrayImage } from "../services/onnxService.js";

// Danh sách nhãn hợp lệ để kiểm tra clinical_info
const validLabels = [
  "Normal",
  "Pneumonia",
  "Bronchitis",
  "Brocho-pneumonia",
  "Other disease",
  "Bronchiolitis",
];

/**
 * API xử lý ảnh X-ray và trả kết quả phân tích
 * @param {object} req - Request object (chứa file ảnh và clinical_info)
 * @param {object} res - Response object
 */
export async function analyzeXray(req, res) {
  try {
    const imagePath = req.file?.path;
    if (!imagePath) {
      return res.status(400).json({ error: "Không tìm thấy file ảnh!" });
    }

    // Parse clinical_info từ body
    let clinical_info = {};
    if (req.body.clinical_info) {
      try {
        clinical_info = JSON.parse(req.body.clinical_info);
      } catch (err) {
        return res
          .status(400)
          .json({ error: "clinical_info phải là JSON hợp lệ!" });
      }
    }

    // Kiểm tra tính hợp lệ của initial_diagnosis (nếu có)
    if (
      clinical_info.initial_diagnosis &&
      !validLabels.includes(clinical_info.initial_diagnosis)
    ) {
      return res.status(400).json({
        error: `Chẩn đoán lâm sàng không hợp lệ! Phải thuộc: ${validLabels.join(
          ", "
        )}`,
      });
    }

    // Kiểm tra symptoms (nếu có)
    if (clinical_info.symptoms && !Array.isArray(clinical_info.symptoms)) {
      return res.status(400).json({
        error: "Triệu chứng phải là một mảng (array) các chuỗi!",
      });
    }

    console.log(`File ảnh đã upload: ${imagePath}`);
    console.log(`Thông tin lâm sàng: ${JSON.stringify(clinical_info)}`);

    // Gọi hàm analyzeXrayImage với imagePath và clinical_info
    const result = await analyzeXrayImage(imagePath, clinical_info);

    res.json(result);
  } catch (err) {
    console.error("Lỗi xử lý phân tích ảnh:", err);
    res.status(500).json({ error: "Đã xảy ra lỗi khi phân tích ảnh X-ray!" });
  }
}
