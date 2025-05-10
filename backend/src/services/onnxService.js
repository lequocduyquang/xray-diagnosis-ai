import fs from "fs/promises";
import sharp from "sharp";
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

async function preprocessImage(pngFilePath) {
  try {
    const imageBuffer = await sharp(pngFilePath)
      .resize(224, 224)
      .removeAlpha()
      .ensureAlpha() // đảm bảo luôn có 3 channels RGB
      .raw()
      .toBuffer();

    const floatImage = new Float32Array(imageBuffer.length);
    for (let i = 0; i < imageBuffer.length; i++) {
      floatImage[i] = imageBuffer[i] / 255.0;
    }

    const tensor = new ort.Tensor("float32", floatImage, [1, 3, 224, 224]);
    return tensor;
  } catch (error) {
    console.error("Error during image preprocessing:", error);
    throw error;
  }
}
