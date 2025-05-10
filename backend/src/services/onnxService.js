import fs from "fs/promises";
import { Jimp } from "jimp";
import * as ort from "onnxruntime-node";
import path from "path";
import { fileURLToPath } from "url";
import { softmax, getPredictedClass } from "../utils/calculation.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * Sử dụng ONNX model để phân tích ảnh X-quang
 * @param {string} filePathOrUrl Đường dẫn đến file ảnh (PNG/JPEG)
 * @returns {Promise<any>}
 */
export async function analyzeXrayImage(filePathOrUrl) {
  try {
    let fileBuffer;

    if (filePathOrUrl.startsWith("http")) {
      // Tải file từ URL
      const response = await fetch(filePathOrUrl);
      if (!response.ok) {
        throw new Error(
          `Failed to fetch file from URL: ${response.statusText}`
        );
      }
      const arrayBuffer = await response.arrayBuffer();
      fileBuffer = Buffer.from(arrayBuffer);
    } else {
      // Đọc file từ local
      fileBuffer = await fs.readFile(filePathOrUrl);
    }

    console.log(`Đã tải file: ${filePathOrUrl}`);

    // Tiền xử lý ảnh
    const inputTensor = await preprocessImage(fileBuffer);

    // Đường dẫn đến model ONNX
    const modelPath = path.join(
      __dirname,
      "../ml-models/resnet50-pneumonia.onnx"
    );

    // Tạo session ONNX
    const session = await ort.InferenceSession.create(modelPath);

    // Chạy inference
    const feeds = { input: inputTensor };
    const results = await session.run(feeds);

    // Xử lý kết quả
    const logits = results.output.data;
    const probabilities = softmax(logits);
    const predictedClass = getPredictedClass(probabilities);

    return {
      success: true,
      message: "Inference completed successfully",
      data: {
        probabilities,
        predictedClass,
        classLabels: ["Normal", "Pneumonia"],
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

async function preprocessImage(imagePath) {
  try {
    const image = await Jimp.read(imagePath);

    const targetWidth = 224;
    const targetHeight = 224;
    image.resize({
      w: targetWidth,
      h: targetHeight,
    });

    const pixels = image.bitmap.data;

    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    const tensorData = new Float32Array(3 * targetWidth * targetHeight);

    for (let y = 0; y < targetHeight; y++) {
      for (let x = 0; x < targetWidth; x++) {
        const pixelIdx = (y * targetWidth + x) * 4; // Jimp includes alpha channel

        const r = pixels[pixelIdx] / 255.0;
        const g = pixels[pixelIdx + 1] / 255.0;
        const b = pixels[pixelIdx + 2] / 255.0;

        const idx = y * targetWidth + x;

        tensorData[idx] = (r - mean[0]) / std[0]; // R
        tensorData[targetWidth * targetHeight + idx] = (g - mean[1]) / std[1]; // G
        tensorData[2 * targetWidth * targetHeight + idx] =
          (b - mean[2]) / std[2]; // B
      }
    }

    const tensor = new ort.Tensor("float32", tensorData, [
      1,
      3,
      targetHeight,
      targetWidth,
    ]);

    return tensor;
  } catch (err) {
    console.error("❌ Error in preprocessImage:", err);
    throw err;
  }
}
