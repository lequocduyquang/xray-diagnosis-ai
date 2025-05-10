import fs from "fs/promises";
import { Jimp } from "jimp";
import * as ort from "onnxruntime-node";
import path from "path";
import { fileURLToPath } from "url";
import { dicomToPng } from "../utils/imageProcessing.js";
import { softmax, getPredictedClass } from "../utils/calculation.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * Kiểm tra xem có phải DICOM file không
 * @param {string} filePath
 * @returns {Promise<boolean>}
 */
async function isDicomFile(filePath) {
  const buffer = await fs.readFile(filePath);
  return buffer.slice(128, 132).toString() === "DICM";
}

/**
 * Sử dụng ONNX model để phân tích ảnh X-quang
 * @param {string} filePath Đường dẫn đến file ảnh (DICOM hoặc PNG/JPEG)
 * @returns {Promise<any>}
 */
export async function analyzeXrayImage(filePath) {
  try {
    let processedPath;

    if (await isDicomFile(filePath)) {
      processedPath = await dicomToPng(filePath);
      console.log(`Đã chuyển DICOM sang PNG: ${processedPath}`);
    } else {
      processedPath = filePath;
      console.log(`Ảnh thường (PNG/JPEG) nhận vào: ${processedPath}`);
    }

    const modelPath = path.join(
      __dirname,
      "../../",
      "../ml/models/resnet50-pneumonia.onnx"
    );

    const session = await ort.InferenceSession.create(modelPath);
    const inputTensor = await preprocessImage(processedPath);

    const feeds = { input: inputTensor };
    const results = await session.run(feeds);

    const logits = results.output.cpuData;
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

/**
 * Trích xuất embedding vector từ ảnh bằng model embedding
 * @param {string} filePath Đường dẫn ảnh (có thể là DICOM)
 * @returns {Promise<number[]>} Vector embedding
 */
export async function extractEmbedding(filePath) {
  try {
    let processedPath;

    if (await isDicomFile(filePath)) {
      processedPath = await dicomToPng(filePath);
      console.log(
        `🧠 Đã convert DICOM sang PNG cho embedding: ${processedPath}`
      );
    } else {
      processedPath = filePath;
    }

    // Load model embedding (resnet50 cắt layer gần cuối)
    const modelPath = path.join(
      __dirname,
      "../../",
      "ml/models/resnet50-embedding.onnx"
    );
    const session = await ort.InferenceSession.create(modelPath);

    const inputTensor = await preprocessImage(processedPath);
    const feeds = { input: inputTensor };

    const results = await session.run(feeds);

    const outputKey = Object.keys(results)[0];
    const embeddingTensor = results[outputKey];

    return Array.from(embeddingTensor.data); // Trả về mảng số thực
  } catch (error) {
    console.error("❌ Lỗi khi trích xuất embedding:", error);
    throw error;
  }
}
