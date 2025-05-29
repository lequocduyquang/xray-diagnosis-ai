import fs from "fs/promises";
import { Jimp } from "jimp";
import * as ort from "onnxruntime-node";
import path from "path";
import { fileURLToPath } from "url";
import { softmax, sigmoid } from "../utils/calculation.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Các label binary và multi-label
const binaryClassLabels = ["Normal", "Pneumonia"];
const multiLabelNames = [
  "Bronchitis",
  "Brocho-pneumonia",
  "Other disease",
  "Bronchiolitis",
  "Pneumonia",
];

/**
 * Phân tích ảnh X-quang bằng 3 model ONNX: ResNet50-v1, ResNet50-v2, DenseNet121
 * @param {string} filePathOrUrl Đường dẫn ảnh local hoặc URL
 * @returns {Promise<any>}
 */
export async function analyzeXrayImage(filePathOrUrl) {
  try {
    let fileBuffer;

    if (filePathOrUrl.startsWith("http")) {
      const response = await fetch(filePathOrUrl);
      if (!response.ok) {
        throw new Error(
          `Failed to fetch file from URL: ${response.statusText}`
        );
      }
      const arrayBuffer = await response.arrayBuffer();
      fileBuffer = Buffer.from(arrayBuffer);
    } else {
      fileBuffer = await fs.readFile(filePathOrUrl);
    }

    const inputTensor = await preprocessImage(fileBuffer);

    // Đường dẫn tới model
    const modelPaths = {
      resnetV1: path.join(__dirname, "../ml-models/resnet50_v1.onnx"),
      resnetV2: path.join(__dirname, "../ml-models/resnet50_v2.onnx"),
      densenet: path.join(__dirname, "../ml-models/densenet121.onnx"),
    };

    // Chạy song song ResNet50 V1 và V2
    const [adult, child] = await Promise.all([
      runBinaryClassifier(modelPaths.resnetV1, inputTensor),
      runBinaryClassifier(modelPaths.resnetV2, inputTensor),
    ]);

    // Weighted Ensemble
    const w1 = 0.3; // ResNet50-v1
    const w2 = 0.7; // ResNet50-v2
    const avgProbs = [
      adult.probabilities[0] * w1 + child.probabilities[0] * w2,
      adult.probabilities[1] * w1 + child.probabilities[1] * w2,
    ];
    const predictedIdx = avgProbs.indexOf(Math.max(...avgProbs));
    const finalLabel = binaryClassLabels[predictedIdx];

    if (finalLabel === "Normal") {
      return {
        success: true,
        stage: "binary-classification",
        message: "Result: Normal",
        data: {
          probabilities: avgProbs,
          predictedClass: finalLabel,
          classLabels: binaryClassLabels,
        },
      };
    }

    // Nếu là Pneumonia thì chạy thêm DenseNet để phân tích chuyên sâu
    const multiLabelProbs = await runMultiLabelClassifier(
      modelPaths.densenet,
      inputTensor
    );

    // Trả top-n nhãn (ví dụ top 3)
    const topN = 3;
    const topLabels = multiLabelProbs
      .map((prob, i) => ({ label: multiLabelNames[i], score: prob }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topN);

    return {
      success: true,
      stage: "multi-label-diagnosis",
      message: "Result: Pneumonia with subtypes",
      data: {
        binaryProbabilities: avgProbs,
        predictedClass: finalLabel,
        classLabels: binaryClassLabels,
        multiLabelTop: topLabels,
        allMultiLabelScores: multiLabelNames.map((label, idx) => ({
          label,
          score: multiLabelProbs[idx],
        })),
      },
    };
  } catch (error) {
    console.error("❌ Error during ONNX inference:", error);
    return {
      success: false,
      message: "Error during inference",
      error: error.message,
    };
  }
}

// -------------------------------------
// Sub-functions
// -------------------------------------

async function runBinaryClassifier(modelPath, inputTensor) {
  const session = await ort.InferenceSession.create(modelPath);
  const feeds = { input: inputTensor };
  const results = await session.run(feeds);
  const logits = results.output.data;
  const probabilities = softmax(logits);
  return { probabilities };
}

async function runMultiLabelClassifier(modelPath, inputTensor) {
  const session = await ort.InferenceSession.create(modelPath);
  const feeds = { input: inputTensor };
  const results = await session.run(feeds);
  const logits = results.output.data;
  const probabilities = sigmoid(logits);
  return probabilities;
}

async function preprocessImage(imageBuffer) {
  const image = await Jimp.read(imageBuffer);
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
      const pixelIdx = (y * targetWidth + x) * 4;
      const r = pixels[pixelIdx] / 255.0;
      const g = pixels[pixelIdx + 1] / 255.0;
      const b = pixels[pixelIdx + 2] / 255.0;
      const idx = y * targetWidth + x;
      tensorData[idx] = (r - mean[0]) / std[0];
      tensorData[targetWidth * targetHeight + idx] = (g - mean[1]) / std[1];
      tensorData[2 * targetWidth * targetHeight + idx] = (b - mean[2]) / std[2];
    }
  }

  return new ort.Tensor("float32", tensorData, [
    1,
    3,
    targetHeight,
    targetWidth,
  ]);
}
