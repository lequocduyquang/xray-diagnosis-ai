import sharp from "sharp";
import * as ort from "onnxruntime-node";
import { dicomToPng } from "../utils/imageProcessing.js";
import path from "path";
import { fileURLToPath } from "url";
import { softmax, getPredictedClass } from "../utils/calculation.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * Sử dụng ONNX model để phân tích ảnh X-quang
 * @param {string} dicomFilePath Đường dẫn đến file DICOM
 * @returns {Promise<any>} Kết quả inference từ model ONNX
 */
export async function analyzeXrayImage(dicomFilePath) {
  try {
    // Chuyển file DICOM sang PNG
    const pngPath = await dicomToPng(dicomFilePath);
    console.log(`Ảnh đã được chuyển thành công sang PNG: ${pngPath}`);

    // Đường dẫn đến file ONNX model
    const modelPath = path.join(
      __dirname,
      "../../",
      "../ml/models/resnet50-pneumonia.onnx"
    );

    // Tải mô hình ONNX
    const session = await ort.InferenceSession.create(modelPath);

    // Xử lý ảnh PNG thành tensor
    const inputTensor = await preprocessImage(pngPath); // Preprocess ảnh PNG
    const feeds = { input: inputTensor };

    // Chạy suy luận (inference)
    const results = await session.run(feeds);

    // Xử lý logits thành xác suất
    const logits = results.output.cpuData;
    const probabilities = softmax(logits);
    const predictedClass = getPredictedClass(probabilities);

    // Bọc kết quả trả về
    return {
      success: true,
      message: "Inference completed successfully",
      data: {
        probabilities,
        predictedClass,
        classLabels: ["Normal", "Pneumonia"], // Gán nhãn lớp
      },
    };
  } catch (error) {
    console.error("Error during ONNX inference:", error);
    return {
      success: false,
      message: "Error during inference",
      error: error.message,
    };
  }
}

/**
 * Tiền xử lý ảnh (PNG) trước khi đưa vào mô hình ONNX
 * @param {string} pngFilePath Đường dẫn đến file PNG
 * @returns {Promise<ort.Tensor>} Tensor đầu vào cho mô hình ONNX
 */
async function preprocessImage(pngFilePath) {
  try {
    // Resize ảnh về kích thước 224x224 và chuyển sang định dạng raw
    const imageBuffer = await sharp(pngFilePath)
      .resize(224, 224) // Resize ảnh về 224x224
      .raw() // Lấy dữ liệu ảnh ở định dạng raw (RGB)
      .toBuffer();

    // Chuyển đổi buffer thành tensor
    const floatImage = new Float32Array(imageBuffer.length);
    for (let i = 0; i < imageBuffer.length; i++) {
      floatImage[i] = imageBuffer[i] / 255.0; // Chuẩn hóa giá trị pixel về [0, 1]
    }

    // Tạo tensor với kích thước [1, 3, 224, 224]
    const tensor = new ort.Tensor("float32", floatImage, [1, 3, 224, 224]);
    return tensor;
  } catch (error) {
    console.error("Error during image preprocessing:", error);
    throw error;
  }
}
