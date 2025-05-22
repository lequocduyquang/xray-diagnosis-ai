import { runModelInference } from "./onnxService.js";
import { softmax, getPredictedClass } from "../utils/calculation.js";

const CLASS_LABELS = ["Normal", "Pneumonia"];
const RARE_DISEASE_LABELS = ["Rare Disease", "Not Rare"];

async function handleMCPRequest(request) {
  const { requestId, input, context } = request;
  const base64Image = input.data;

  // Step 1: Run ResNet50
  const logitsResNet50 = await runModelInference("ResNet50", base64Image);
  const probsResNet50 = softmax(logitsResNet50);
  const predClassResNet50 = getPredictedClass(probsResNet50);

  if (predClassResNet50 === 0) {
    // Normal case
    return {
      requestId,
      timestamp: new Date().toISOString(),
      model: "ResNet50",
      status: "success",
      result: {
        predictedClass: CLASS_LABELS[predClassResNet50],
        probabilities: probsResNet50,
        rawOutput: logitsResNet50,
      },
      error: null,
    };
  }

  // Step 2: Pneumonia detected → check with ResNet18 for rare disease
  const logitsResNet18 = await runModelInference("ResNet18", base64Image);
  const probsResNet18 = softmax(logitsResNet18);
  const predClassResNet18 = getPredictedClass(probsResNet18);

  if (predClassResNet18 === 0) {
    // Not rare pneumonia
    return {
      requestId,
      timestamp: new Date().toISOString(),
      model: "ResNet18",
      status: "success",
      result: {
        predictedClass: RARE_DISEASE_LABELS[predClassResNet18],
        probabilities: probsResNet18,
        rawOutput: logitsResNet18,
      },
      error: null,
    };
  }

  // Step 3: Rare disease detected → run EfficientNet for final decision
  const logitsEfficientNet = await runModelInference("EfficientNet", base64Image);
  const probsEfficientNet = softmax(logitsEfficientNet);
  const predClassEfficientNet = getPredictedClass(probsEfficientNet);

  return {
    requestId,
    timestamp: new Date().toISOString(),
    model: "EfficientNet",
    status: "success",
    result: {
      predictedClass: RARE_DISEASE_LABELS[predClassEfficientNet],
      probabilities: probsEfficientNet,
      rawOutput: logitsEfficientNet,
    },
    error: null,
  };
}

export { handleMCPRequest };
