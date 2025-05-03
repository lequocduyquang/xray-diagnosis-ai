import * as ort from "onnxruntime-node";
import { dicomToPng } from "../utils/imageProcessing.js";

/**
 * Sử dụng ONNX model để phân tích ảnh X-quang
 * @param {string} dicomFilePath Đường dẫn đến file DICOM
 * @returns {Promise<any>} Kết quả inference từ model ONNX
 */
export async function analyzeXrayImage(dicomFilePath) {
  const pngPath = await dicomToPng(dicomFilePath);

  const modelPath =
    process.env.ONNX_MODEL_PATH || "models/resnet50-pneumonia.onnx";
  const session = await ort.InferenceSession.create(modelPath);

  const inputTensor = await preprocessImage(pngPath); // Preprocess ảnh PNG
  const feeds = { input: inputTensor };
  const results = await session.run(feeds);

  return results; // Kết quả từ mô hình ONNX
}

/**
 * Tiền xử lý ảnh (PNG) trước khi đưa vào mô hình ONNX
 * @param {string} pngFilePath Đường dẫn đến file PNG
 * @returns {Promise<ort.Tensor>} Tensor đầu vào cho mô hình ONNX
 */
async function preprocessImage(pngFilePath) {
  // Đọc ảnh PNG, xử lý và chuyển thành tensor cho ONNX
  const imageBuffer = fs.readFileSync(pngFilePath);
  const tensor = new ort.Tensor(
    "float32",
    new Float32Array(imageBuffer),
    [1, 3, 224, 224]
  ); // Ví dụ 224x224x3 ảnh RGB
  return tensor;
}
