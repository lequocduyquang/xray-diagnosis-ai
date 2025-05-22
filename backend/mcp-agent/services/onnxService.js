import * as ort from "onnxruntime-node";
import Jimp from "jimp";

const modelSessions = {};

async function loadModel(modelName, modelPath) {
  if (!modelSessions[modelName]) {
    modelSessions[modelName] = await ort.InferenceSession.create(modelPath);
  }
  return modelSessions[modelName];
}

async function preprocessImage(base64Image) {
  // base64 → buffer
  const buffer = Buffer.from(base64Image, "base64");
  const image = await Jimp.read(buffer);

  const targetWidth = 224;
  const targetHeight = 224;
  image.resize(targetWidth, targetHeight);

  const pixels = image.bitmap.data;

  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  const tensorData = new Float32Array(3 * targetWidth * targetHeight);

  for (let y = 0; y < targetHeight; y++) {
    for (let x = 0; x < targetWidth; x++) {
      const idx = (y * targetWidth + x) * 4;
      const r = pixels[idx] / 255;
      const g = pixels[idx + 1] / 255;
      const b = pixels[idx + 2] / 255;

      const pixelIdx = y * targetWidth + x;
      tensorData[pixelIdx] = (r - mean[0]) / std[0];
      tensorData[targetWidth * targetHeight + pixelIdx] = (g - mean[1]) / std[1];
      tensorData[2 * targetWidth * targetHeight + pixelIdx] = (b - mean[2]) / std[2];
    }
  }

  return new ort.Tensor("float32", tensorData, [1, 3, targetHeight, targetWidth]);
}

async function runModelInference(modelName, base64Image) {
  let modelPath;
  switch (modelName) {
    case "ResNet50":
      modelPath = "./ml-models/resnet50-pneumonia.onnx";
      break;
    case "ResNet18":
      modelPath = "./ml-models/resnet18-pneumonia.onnx";
      break;
    case "EfficientNet":
      modelPath = "./ml-models/efficientnet-common.onnx";
      break;
    default:
      throw new Error("Unsupported model: " + modelName);
  }

  const session = await loadModel(modelName, modelPath);
  const inputTensor = await preprocessImage(base64Image);
  const feeds = { input: inputTensor };
  const results = await session.run(feeds);
  return results.output.data;
}

export { runModelInference };
