// src/utils/imageProcessing.js
import fs from "fs";
import dicomParser from "dicom-parser";
import sharp from "sharp";

/**
 * Chuyển đổi ảnh DICOM sang PNG
 * @param {string} dicomFilePath Đường dẫn đến file DICOM
 * @returns {Promise<string>} Đường dẫn đến file PNG đã chuyển đổi
 */
export async function dicomToPng(dicomFilePath) {
  try {
    // Đọc file DICOM
    const dicomBuffer = fs.readFileSync(dicomFilePath);
    const dataSet = dicomParser.parseDicom(dicomBuffer);

    // Lấy thông tin ảnh từ DICOM
    const pixelDataElement = dataSet.elements.x7fe00010;
    const pixelData = dicomBuffer.slice(pixelDataElement.dataOffset);

    // Lấy kích thước ảnh
    const width = dataSet.uint16("x00280011"); // Chiều rộng (Width)
    const height = dataSet.uint16("x00280010"); // Chiều cao (Height)
    const bitsAllocated = dataSet.uint16("x00280100"); // Bit Depth (Số bit)

    // Kiểm tra độ sâu bit của ảnh
    if (bitsAllocated !== 16) {
      throw new Error("Chỉ hỗ trợ ảnh DICOM 16-bit!");
    }

    // Chuyển đổi dữ liệu pixel 16-bit sang 8-bit và chuẩn hóa lại
    const pixelArray = new Uint16Array(pixelData.buffer);
    const normalizedPixels = normalizePixels(pixelArray, width, height);

    // Lưu ảnh dưới định dạng PNG
    const outputPath = dicomFilePath.replace(".dcm", ".png"); // Tạo đường dẫn ảnh PNG từ file DICOM
    await sharp(Buffer.from(normalizedPixels))
      .toFormat("png")
      .toFile(outputPath);

    console.log(`Ảnh đã được chuyển thành công sang PNG: ${outputPath}`);
    return outputPath;
  } catch (err) {
    console.error("Lỗi khi chuyển đổi DICOM sang PNG:", err);
    throw err; // Ném lỗi để có thể xử lý tiếp ở các bước sau
  }
}

/**
 * Chuyển đổi dữ liệu pixel 16-bit sang 8-bit và chuẩn hóa lại
 * @param {Uint16Array} pixelArray - Mảng dữ liệu pixel
 * @param {number} width - Chiều rộng của ảnh
 * @param {number} height - Chiều cao của ảnh
 * @returns {Buffer} - Buffer ảnh chuẩn hóa
 */
function normalizePixels(pixelArray, width, height) {
  const normalized = new Uint8Array(width * height);

  // Giả sử phạm vi giá trị pixel là 0-65535 (16-bit), chuẩn hóa về 0-255 (8-bit)
  const maxPixelValue = 65535;
  const minPixelValue = 0;

  for (let i = 0; i < pixelArray.length; i++) {
    // Tính toán pixel chuẩn hóa
    const normalizedValue = Math.round(
      ((pixelArray[i] - minPixelValue) / (maxPixelValue - minPixelValue)) * 255
    );
    normalized[i] = normalizedValue;
  }

  return normalized;
}
