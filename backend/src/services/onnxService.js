import fs from "fs/promises";
import { Jimp } from "jimp";
import * as ort from "onnxruntime-node";
import path from "path";
import { fileURLToPath } from "url";
import { softmax, sigmoid } from "../utils/calculation.js";
import { saveImageToDatabase } from "./databaseService.js";
import { ensureModelDownloaded } from "./huggingfaceService.js";

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
 * Điều chỉnh xác suất dựa trên thông tin lâm sàng, chỉ cho binary labels
 * @param {Object} probs Xác suất từ AI (Normal, Pneumonia)
 * @param {Object} clinical_info Thông tin lâm sàng
 * @param {string} predictedClass Nhãn dự đoán của AI
 * @param {number} highThreshold Ngưỡng cho xác suất cao của nhãn khác
 * @returns {Object} Xác suất điều chỉnh và cảnh báo
 */
function adjust_probabilities(
  probs,
  clinical_info,
  predictedClass,
  highThreshold = 0.49
) {
  const weights = {};
  for (const label of binaryClassLabels) {
    weights[label] = 1.0;
  }

  // Áp dụng trọng số nhẹ dựa trên chẩn đoán lâm sàng, chỉ cho binary labels
  const initial_diagnosis = clinical_info?.initial_diagnosis || "";
  if (initial_diagnosis === "Normal") {
    weights["Normal"] = 1.2;
    weights["Pneumonia"] = 0.8;
  } else if (initial_diagnosis === "Pneumonia") {
    weights["Pneumonia"] = 1.2;
    weights["Normal"] = 0.8;
  }

  // Điều chỉnh xác suất
  const final_probs = {};
  let total = 0;
  for (const label in probs) {
    final_probs[label] = probs[label] * weights[label];
    total += final_probs[label];
  }
  for (const label in final_probs) {
    final_probs[label] /= total;
  }

  // Log debug để kiểm tra ảnh hưởng trọng số
  console.log(`Original probs: ${JSON.stringify(probs)}`);
  console.log(`Weights: ${JSON.stringify(weights)}`);
  console.log(`Final probs: ${JSON.stringify(final_probs)}`);

  // Tính độ mâu thuẫn và tạo cảnh báo (chỉ khi có initial_diagnosis và là binary label)
  const warnings = [];
  if (initial_diagnosis && binaryClassLabels.includes(initial_diagnosis)) {
    const conflicts = {};
    for (const label of binaryClassLabels) {
      const clinical_value = label === initial_diagnosis ? 1.0 : 0.0;
      conflicts[label] = Math.abs(final_probs[label] - clinical_value);
      console.log(
        `Conflict for ${label}: ${conflicts[label]}, Prob: ${final_probs[label]}, Clinical: ${clinical_value}`
      ); // Debug
    }

    // Tạo cảnh báo nếu initial_diagnosis khác predictedClass và nhãn khác có xác suất đủ cao
    if (initial_diagnosis !== predictedClass) {
      if (final_probs[predictedClass] > highThreshold) {
        warnings.push(
          `Cảnh báo: AI phát hiện ${predictedClass} với xác suất cao (${(
            final_probs[predictedClass] * 100
          ).toFixed(
            1
          )}%), nhưng bác sĩ chẩn đoán ${initial_diagnosis}. Đề nghị kiểm tra lại ảnh X-quang hoặc xét nghiệm bổ sung (máu, CRP, CT).`
        );
      }
    }

    console.log(`Warnings generated: ${JSON.stringify(warnings)}`); // Debug
  }

  return { final_probs, warnings };
}

/**
 * Phân tích ảnh X-quang với thông tin lâm sàng
 * @param {string} filePathOrUrl Đường dẫn ảnh local hoặc URL
 * @param {Object} clinical_info Thông tin lâm sàng { initial_diagnosis, symptoms }
 * @param {string} cloudinaryId ID của ảnh trên Cloudinary (optional)
 * @returns {Promise<any>}
 */
export async function analyzeXrayImage(
  filePathOrUrl,
  clinical_info = {},
  cloudinaryId = null
) {
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

    // // Đường dẫn tới model
    // const modelPaths = {
    //   resnetV1: path.join(__dirname, "../ml-models/resnet50_v1.onnx"),
    //   resnetV2: path.join(__dirname, "../ml-models/resnet50_v2.onnx"),
    //   densenet: path.join(__dirname, "../ml-models/densenet121.onnx"),
    // };

    // await Promise.all([
    //   ensureModelDownloaded("resnetV1"),
    //   ensureModelDownloaded("resnetV2"),
    //   ensureModelDownloaded("densenet"),
    // ]);

    // Đường dẫn tới model local (đồng bộ với huggingfaceService.js)
    const modelPaths = {
      resnetV1: path.join(__dirname, "../ml-models-v2/resnet50_v1.onnx"),
      resnetV2: path.join(__dirname, "../ml-models-v2/resnet50_v2.onnx"),
      densenet: path.join(__dirname, "../ml-models-v2/densenet121.onnx"),
    };

    // Chạy song song ResNet50 V1 và V2
    const [adult, child] = await Promise.all([
      runBinaryClassifier(modelPaths.resnetV1, inputTensor),
      runBinaryClassifier(modelPaths.resnetV2, inputTensor),
    ]);

    // Weighted Ensemble
    const w1 = 0.4; // ResNet50-v1
    const w2 = 0.6; // ResNet50-v2
    const avgProbs = {
      Normal: adult.probabilities[0] * w1 + child.probabilities[0] * w2,
      Pneumonia: adult.probabilities[1] * w1 + child.probabilities[1] * w2,
    };

    // Điều chỉnh xác suất binary dựa trên lâm sàng
    const { final_probs: finalBinaryProbs, warnings: binaryWarnings } =
      adjust_probabilities(
        avgProbs,
        clinical_info,
        binaryClassLabels[
          Object.values(avgProbs).indexOf(Math.max(...Object.values(avgProbs)))
        ]
      );
    const predictedIdx = Object.values(finalBinaryProbs).indexOf(
      Math.max(...Object.values(finalBinaryProbs))
    );
    const finalLabel = binaryClassLabels[predictedIdx];

    console.log(
      `Binary probs: ${JSON.stringify(
        finalBinaryProbs
      )}, Predicted: ${finalLabel}`
    ); // Debug

    if (finalLabel === "Normal") {
      // Cập nhật model_name trong database nếu có cloudinaryId
      if (cloudinaryId) {
        try {
          await saveImageToDatabase({
            cloudinaryId: cloudinaryId,
            cloudinaryUrl: filePathOrUrl,
            modelName: "ResNet50",
          });
          console.log(
            `✅ Đã cập nhật model_name = ResNet50 cho cloudinary_id: ${cloudinaryId}`
          );
        } catch (dbError) {
          console.error(
            "⚠️ Lỗi khi cập nhật model_name trong database:",
            dbError
          );
        }
      }

      return {
        success: true,
        stage: "binary-classification",
        message: "Result: Normal",
        data: {
          clinical_info,
          binaryProbabilities: finalBinaryProbs,
          predictedClass: finalLabel,
          classLabels: binaryClassLabels,
          warnings: binaryWarnings,
          cloudinaryId,
          modelName: "ResNet50",
        },
      };
    }

    // Multi-label (không điều chỉnh xác suất dựa trên clinical_info)
    const multiLabelProbs = await runMultiLabelClassifier(
      modelPaths.densenet,
      inputTensor
    );
    const multiLabelProbsObj = {};
    multiLabelNames.forEach(
      (label, idx) => (multiLabelProbsObj[label] = multiLabelProbs[idx])
    );

    // Lấy top 3 label lớn nhất
    const allMultiLabelScores = multiLabelNames.map((label) => ({
      label,
      score: multiLabelProbsObj[label],
    }));
    const sorted = allMultiLabelScores
      .slice()
      .sort((a, b) => b.score - a.score);
    const multiLabelTop = {};
    for (let i = 0; i < 3; i++) {
      multiLabelTop[i] = sorted[i] || null;
    }

    // Cập nhật model_name trong database nếu có cloudinaryId
    if (cloudinaryId) {
      try {
        await saveImageToDatabase({
          cloudinaryId: cloudinaryId,
          cloudinaryUrl: filePathOrUrl,
          modelName: "DenseNet121",
        });
        console.log(
          `✅ Đã cập nhật model_name = DenseNet121 cho cloudinary_id: ${cloudinaryId}`
        );
      } catch (dbError) {
        console.error(
          "⚠️ Lỗi khi cập nhật model_name trong database:",
          dbError
        );
      }
    }

    return {
      success: true,
      stage: "multi-label-diagnosis",
      message: "Result: Pneumonia with subtypes",
      data: {
        clinical_info,
        binaryProbabilities: finalBinaryProbs,
        predictedClass: finalLabel,
        classLabels: binaryClassLabels,
        multiLabelTop,
        allMultiLabelScores,
        cloudinaryId,
        warnings: binaryWarnings,
        modelName: "DenseNet121",
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
