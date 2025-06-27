import fs from "fs/promises";
import { Jimp } from "jimp";
import * as ort from "onnxruntime-node";
import path from "path";
import { fileURLToPath } from "url";
import { softmax, sigmoid } from "../utils/calculation.js";
import { saveImageToDatabase } from "./databaseService.js";
// import { ensureModelDownloaded } from "./huggingfaceService.js";

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

// Memory management for ONNX sessions
let sessionCache = {};
let lastCleanup = Date.now();
const CLEANUP_INTERVAL = 5 * 60 * 1000; // 5 minutes
const MAX_SESSIONS = 3; // Limit concurrent sessions

// Force cleanup sessions to free memory
function forceCleanupSessions() {
  console.log('🧹 Force cleaning up ONNX sessions...');
  Object.keys(sessionCache).forEach(key => {
    try {
      if (sessionCache[key]?.session) {
        sessionCache[key].session.release?.();
      }
    } catch (e) {
      console.warn(`Warning releasing session ${key}:`, e.message);
    }
  });
  sessionCache = {};

  // Force garbage collection if available
  if (global.gc) {
    global.gc();
    console.log('🗑️ Forced garbage collection');
  }

  lastCleanup = Date.now();
}

// Get or create session with memory management
async function getOrCreateSession(modelPath, modelType) {
  const now = Date.now();

  // Force cleanup if too long since last cleanup or too many sessions
  if (now - lastCleanup > CLEANUP_INTERVAL || Object.keys(sessionCache).length >= MAX_SESSIONS) {
    forceCleanupSessions();
  }

  const cacheKey = `${modelType}_${modelPath}`;

  if (sessionCache[cacheKey] && sessionCache[cacheKey].session) {
    sessionCache[cacheKey].lastUsed = now;
    return sessionCache[cacheKey].session;
  }

  try {
    console.log(`📥 Loading ONNX session: ${modelType}`);

    // Create session with memory optimization settings
    const sessionOptions = {
      executionProviders: ['cpu'],
      graphOptimizationLevel: 'basic', // Reduce memory usage
      enableMemPattern: false,
      enableCpuMemArena: false,
    };

    const session = await ort.InferenceSession.create(modelPath, sessionOptions);

    sessionCache[cacheKey] = {
      session,
      lastUsed: now,
      modelType
    };

    console.log(`✅ Session loaded: ${modelType}, Cache size: ${Object.keys(sessionCache).length}`);
    return session;

  } catch (error) {
    console.error(`❌ Failed to create session for ${modelType}:`, error);
    throw error;
  }
}

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
// Memory monitoring helper
function logMemoryUsage(stage) {
  const used = process.memoryUsage();
  const mb = (bytes) => Math.round(bytes / 1024 / 1024 * 100) / 100;
  console.log(`📊 [${stage}] Memory: RSS=${mb(used.rss)}MB, Heap=${mb(used.heapUsed)}/${mb(used.heapTotal)}MB`);

  // Warning if memory usage is high
  if (mb(used.heapUsed) > 350) {
    console.warn(`⚠️ HIGH MEMORY WARNING: ${mb(used.heapUsed)}MB used`);
    // Force cleanup if memory is very high
    if (mb(used.heapUsed) > 400) {
      forceCleanupSessions();
    }
  }

  return mb(used.heapUsed);
}

// Check if we should use single model mode due to memory constraints
function shouldUseSingleModelMode() {
  const used = process.memoryUsage();
  const mb = (bytes) => Math.round(bytes / 1024 / 1024 * 100) / 100;
  return mb(used.heapUsed) > 300; // Use single model if > 300MB
}

export async function analyzeXrayImage(
  filePathOrUrl,
  clinical_info = {},
  cloudinaryId = null
) {
  logMemoryUsage('START');
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

    logMemoryUsage('BEFORE_PREPROCESSING');
    const inputTensor = await preprocessImage(fileBuffer);
    logMemoryUsage('AFTER_PREPROCESSING');

    // // Đường dẫn tới model
    const modelPaths = {
      resnetV1: path.join(__dirname, "../ml-models/resnet50_v1.onnx"),
      resnetV2: path.join(__dirname, "../ml-models/resnet50_v2.onnx"),
      densenet: path.join(__dirname, "../ml-models/densenet121.onnx"),
    };

    // await Promise.all([
    //   ensureModelDownloaded("resnetV1"),
    //   ensureModelDownloaded("resnetV2"),
    //   ensureModelDownloaded("densenet"),
    // ]);

    // Đường dẫn tới model local (đồng bộ với huggingfaceService.js)
    // const modelPaths = {
    //   resnetV1: path.join(__dirname, "../ml-models-v2/resnet50_v1.onnx"),
    //   resnetV2: path.join(__dirname, "../ml-models-v2/resnet50_v2.onnx"),
    //   densenet: path.join(__dirname, "../ml-models-v2/densenet121.onnx"),
    // };

    // 🚀 MEMORY OPTIMIZATION: Smart model selection based on available memory
    let avgProbs;

    if (shouldUseSingleModelMode()) {
      console.log('⚡ MEMORY SAVER MODE: Using single ResNet50 V2 model only');
      logMemoryUsage('BEFORE_SINGLE_MODEL');
      const child = await runBinaryClassifier(modelPaths.resnetV2, inputTensor);
      logMemoryUsage('AFTER_SINGLE_MODEL');

      // Use single model results
      avgProbs = {
        Normal: child.probabilities[0],
        Pneumonia: child.probabilities[1],
      };
    } else {
      // Normal mode: Sequential processing of both models
      console.log('🔄 Running ResNet50 V1...');
      logMemoryUsage('BEFORE_RESNET_V1');
      const adult = await runBinaryClassifier(modelPaths.resnetV1, inputTensor);
      logMemoryUsage('AFTER_RESNET_V1');

      console.log('🔄 Running ResNet50 V2...');
      const child = await runBinaryClassifier(modelPaths.resnetV2, inputTensor);
      logMemoryUsage('AFTER_RESNET_V2');

      // Weighted Ensemble
      const w1 = 0.4; // ResNet50-v1
      const w2 = 0.6; // ResNet50-v2
      avgProbs = {
        Normal: adult.probabilities[0] * w1 + child.probabilities[0] * w2,
        Pneumonia: adult.probabilities[1] * w1 + child.probabilities[1] * w2,
      };
    }

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
      // 🎯 MEMORY OPTIMIZATION: Early exit for Normal cases (no multi-label needed)
      logMemoryUsage('NORMAL_RESULT_EARLY_EXIT');

      // Cập nhật model_name trong database nếu có cloudinaryId
      if (cloudinaryId) {
        try {
          const modelName = shouldUseSingleModelMode() ? "ResNet50-V2-Single" : "ResNet50-Ensemble";
          await saveImageToDatabase({
            cloudinaryId: cloudinaryId,
            cloudinaryUrl: filePathOrUrl,
            modelName: modelName,
          });
          console.log(
            `✅ Đã cập nhật model_name = ${modelName} cho cloudinary_id: ${cloudinaryId}`
          );
        } catch (dbError) {
          console.error(
            "⚠️ Lỗi khi cập nhật model_name trong database:",
            dbError
          );
        }
      }

      const warnings = [...binaryWarnings];
      if (shouldUseSingleModelMode()) {
        warnings.push("⚡ Memory Saver Mode: Used single ResNet50-V2 model");
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
          warnings: warnings,
          cloudinaryId,
          modelName: shouldUseSingleModelMode() ? "ResNet50-V2-Single" : "ResNet50-Ensemble",
        },
      };
    }

    // Multi-label classification for Pneumonia cases
    logMemoryUsage('BEFORE_MULTILABEL');
    console.log('🔄 Running DenseNet121 for multi-label classification...');

    // Check memory before running DenseNet121
    const currentMemory = logMemoryUsage('CHECK_BEFORE_DENSENET');

    if (currentMemory > 450) {
      // 🚨 EMERGENCY: Use binary model for approximate multi-label results
      console.warn('🚨 CRITICAL MEMORY: Using ResNet50 V2 for approximate multi-label results');
      forceCleanupSessions();

      // Generate approximate multi-label results based on binary confidence
      const confidence = Math.max(...Object.values(finalBinaryProbs));

      // Use actual multiLabelNames from the system
      const approximateMultiLabel = {};
      multiLabelNames.forEach((label, index) => {
        if (label === "Pneumonia") {
          approximateMultiLabel[label] = confidence * 0.8; // Highest for main diagnosis
        } else if (label === "Brocho-pneumonia") {
          approximateMultiLabel[label] = confidence * 0.6; // Common complication
        } else if (label === "Bronchitis") {
          approximateMultiLabel[label] = confidence * 0.4; // Related respiratory
        } else if (label === "Bronchiolitis") {
          approximateMultiLabel[label] = confidence * 0.3; // Pediatric common
        } else if (label === "Other disease") {
          approximateMultiLabel[label] = confidence * 0.2; // Catch-all
        }
      });

      const allMultiLabelScores = Object.entries(approximateMultiLabel).map(([label, score]) => ({
        label,
        score
      })).sort((a, b) => b.score - a.score);

      const multiLabelTop = {};
      for (let i = 0; i < Math.min(3, allMultiLabelScores.length); i++) {
        multiLabelTop[i] = allMultiLabelScores[i];
      }

      return {
        success: true,
        stage: "binary-approximated-multilabel",
        message: "Result: Pneumonia (Approximated multi-label from binary model)",
        data: {
          clinical_info,
          binaryProbabilities: finalBinaryProbs,
          predictedClass: finalLabel,
          classLabels: binaryClassLabels,
          multiLabelTop,
          allMultiLabelScores,
          warnings: [
            ...binaryWarnings,
            "🚨 CRITICAL MEMORY: Used ResNet50-V2 for approximate multi-label results",
            "⚠️ Multi-label results are approximated from binary classification",
            "🩺 Consider DenseNet121 analysis when memory allows for precise subtypes"
          ],
          cloudinaryId,
          modelName: "ResNet50-V2-Approximated",
        },
      };
    }

    // Multi-label (không điều chỉnh xác suất dựa trên clinical_info)
    const multiLabelProbs = await runMultiLabelClassifier(
      modelPaths.densenet,
      inputTensor
    );
    logMemoryUsage('AFTER_MULTILABEL');
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
  let session = null;
  try {
    session = await getOrCreateSession(modelPath, 'binary');
    const feeds = { input: inputTensor };
    const results = await session.run(feeds);
    const logits = results.output.data;
    const probabilities = softmax(logits);
    return { probabilities };
  } catch (error) {
    console.error('❌ Binary classifier error:', error);
    // Cleanup on error
    forceCleanupSessions();
    throw error;
  }
}

async function runMultiLabelClassifier(modelPath, inputTensor) {
  let session = null;
  try {
    session = await getOrCreateSession(modelPath, 'multilabel');
    const feeds = { input: inputTensor };
    const results = await session.run(feeds);
    const logits = results.output.data;
    const probabilities = sigmoid(logits);
    return probabilities;
  } catch (error) {
    console.error('❌ Multi-label classifier error:', error);
    // Cleanup on error
    forceCleanupSessions();
    throw error;
  }
}

async function preprocessImage(imageBuffer) {
  let image = null;
  try {
    // Keep original model size (models were trained on 224x224)
    const targetWidth = 224; // Must match ONNX model input
    const targetHeight = 224; // Must match ONNX model input

    image = await Jimp.read(imageBuffer);

    // Resize with memory optimization - use faster but more memory-efficient method
    image.resize(targetWidth, targetHeight, Jimp.RESIZE_BILINEAR); // Use bilinear for speed

    // Convert to grayscale if it's RGB to reduce memory by ~66%
    if (image.bitmap.width * image.bitmap.height * 4 > 1024 * 1024) { // If > 1MB
      console.log('🔄 Converting large image to grayscale for memory optimization');
      image.greyscale();
    }

    const pixels = image.bitmap.data;
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    const tensorData = new Float32Array(3 * targetWidth * targetHeight);

    // Fast pixel processing - process all at once for better performance
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

    // Clean up Jimp image
    if (image) {
      image = null;
    }

    return new ort.Tensor("float32", tensorData, [
      1,
      3,
      targetHeight,
      targetWidth,
    ]);

  } catch (error) {
    console.error('❌ Image preprocessing error:', error);
    // Clean up on error
    if (image) {
      image = null;
    }
    throw error;
  }
}
