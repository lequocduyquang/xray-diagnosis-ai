import fs from "fs/promises";
import fetch from "node-fetch";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// HuggingFace model URLs
const HF_MODELS = {
  resnetV1: "https://huggingface.co/quanglequocduy/resnet50/resolve/main/resnet50_v1.onnx",
  resnetV2: "https://huggingface.co/quanglequocduy/resnet50/resolve/main/resnet50_v2.onnx",
  densenet: "https://huggingface.co/quanglequocduy/densenet121/resolve/main/densenet121.onnx",
};

// Local cache paths
const modelPaths = {
  resnetV1: path.join(__dirname, "../ml-models-v2/resnet50_v1.onnx"),
  resnetV2: path.join(__dirname, "../ml-models-v2/resnet50_v2.onnx"),
  densenet: path.join(__dirname, "../ml-models-v2/densenet121.onnx"),
};

/**
 * Tải model từ HuggingFace về local nếu chưa có
 * @param {string} modelKey - resnetV1 | resnetV2 | densenet
 */
export async function ensureModelDownloaded(modelKey) {
  const localPath = modelPaths[modelKey];
  try {
    await fs.access(localPath);
  } catch {
    console.log(`Downloading ${modelKey} from HuggingFace...`);
    const res = await fetch(HF_MODELS[modelKey]);
    if (!res.ok) throw new Error(`Failed to download model ${modelKey} from HuggingFace`);
    const buffer = await res.arrayBuffer();
    await fs.mkdir(path.dirname(localPath), { recursive: true });
    await fs.writeFile(localPath, Buffer.from(buffer));
    console.log(`Downloaded ${modelKey} to ${localPath}`);
  }
}