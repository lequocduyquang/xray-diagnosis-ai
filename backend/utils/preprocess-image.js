const fs = require("fs");
const sharp = require("sharp");

/**
 * Tiền xử lý ảnh PNG thành tensor 4D [1, 3, 224, 224] cho ONNX
 * @param {string} imagePath - Đường dẫn ảnh PNG
 * @returns {Float32Array} - Dữ liệu tensor chuẩn hóa
 */
async function preprocessImage(imagePath) {
  const imageBuffer = await sharp(imagePath)
    .resize(224, 224)
    .removeAlpha()
    .raw()
    .toBuffer();

  const floatArray = new Float32Array(3 * 224 * 224);
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  for (let i = 0; i < 224 * 224; i++) {
    const r = imageBuffer[i * 3];
    const g = imageBuffer[i * 3 + 1];
    const b = imageBuffer[i * 3 + 2];

    floatArray[i] = (r / 255 - mean[0]) / std[0];
    floatArray[i + 224 * 224] = (g / 255 - mean[1]) / std[1];
    floatArray[i + 2 * 224 * 224] = (b / 255 - mean[2]) / std[2];
  }

  return floatArray;
}

module.exports = { preprocessImage };
