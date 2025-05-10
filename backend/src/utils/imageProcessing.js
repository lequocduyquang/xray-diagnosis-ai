// src/utils/imageProcessing.js
import fs from "fs";
import { PNG } from "pngjs";
import dicomParser from "dicom-parser";

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
    if (!pixelDataElement) {
      throw new Error("Không tìm thấy dữ liệu pixel trong file DICOM!");
    }
    console.log(`Kích thước dữ liệu pixel: ${pixelDataElement.length} bytes`);

    // Lấy kích thước ảnh
    const width = dataSet.uint16("x00280011"); // Chiều rộng (Width)
    const height = dataSet.uint16("x00280010"); // Chiều cao (Height)
    const bitsAllocated = dataSet.uint16("x00280100"); // Bit Depth (Số bit)
    console.log(
      `Kích thước ảnh: ${width}x${height}, Độ sâu bit: ${bitsAllocated} bits`
    );

    // Kiểm tra độ sâu bit của ảnh
    if (bitsAllocated !== 16 && bitsAllocated !== 8) {
      throw new Error("Chỉ hỗ trợ ảnh DICOM 8-bit hoặc 16-bit!");
    }

    // Tạo mảng pixel từ dữ liệu DICOM
    const pixelData = new Uint8Array(
      dicomBuffer.buffer,
      pixelDataElement.dataOffset,
      pixelDataElement.length
    );

    // Chuẩn hóa pixel về 8-bit nếu cần
    const normalizedPixels = new Uint8Array(width * height);
    if (bitsAllocated === 16) {
      const pixelArray = new Uint16Array(pixelData.buffer);
      const maxPixelValue = Math.max(...pixelArray);
      for (let i = 0; i < pixelArray.length; i++) {
        normalizedPixels[i] = Math.round((pixelArray[i] / maxPixelValue) * 255);
      }
    } else {
      normalizedPixels.set(pixelData); // Nếu là 8-bit, giữ nguyên
    }

    // Tạo ảnh PNG
    const png = new PNG({ width, height });
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) << 2; // RGBA index
        const pixelValue = normalizedPixels[y * width + x];
        png.data[idx] = pixelValue; // R
        png.data[idx + 1] = pixelValue; // G
        png.data[idx + 2] = pixelValue; // B
        png.data[idx + 3] = 255; // Alpha
      }
    }

    // Lưu file PNG tạm thời
    const pngFilePath = dicomFilePath.replace(".dcm", ".png");
    await new Promise((resolve, reject) => {
      png
        .pack()
        .pipe(fs.createWriteStream(pngFilePath))
        .on("finish", resolve)
        .on("error", reject);
    });

    console.log(`Ảnh đã được chuyển thành công sang PNG: ${pngFilePath}`);
    return pngFilePath;
  } catch (err) {
    console.error("Lỗi khi chuyển đổi DICOM sang PNG:", err);
    throw err;
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
