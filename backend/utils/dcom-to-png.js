const fs = require("fs");
const dicomParser = require("dicom-parser");
const sharp = require("sharp");

/**
 * Chuyển đổi file DICOM sang PNG
 * @param {string} dicomPath - Đường dẫn file DICOM đầu vào
 * @param {string} outputPath - Đường dẫn file PNG đầu ra
 */
async function dicomToPng(dicomPath, outputPath) {
  try {
    // Đọc file DICOM
    const dicomBuffer = fs.readFileSync(dicomPath);
    const dataSet = dicomParser.parseDicom(dicomBuffer);

    // Lấy thông tin ảnh từ DICOM
    const pixelDataElement = dataSet.elements.x7fe00010;
    const pixelData = dicomBuffer.slice(pixelDataElement.dataOffset);

    // Giải mã ảnh và chuyển sang PNG
    const width = dataSet.uint16("x00280011"); // Width (Cột)
    const height = dataSet.uint16("x00280010"); // Height (Dài)
    const bitsAllocated = dataSet.uint16("x00280100"); // Bit Depth

    if (bitsAllocated !== 16) {
      throw new Error("Chỉ hỗ trợ ảnh DICOM 16-bit!");
    }

    // Dữ liệu pixel có thể cần được chuyển đổi sang định dạng phù hợp
    const pixelArray = new Uint16Array(pixelData.buffer);

    // Xử lý pixel theo yêu cầu của ảnh
    const normalizedPixels = normalizePixels(pixelArray, width, height);

    // Lưu ảnh dưới định dạng PNG
    await sharp(Buffer.from(normalizedPixels))
      .toFormat("png")
      .toFile(outputPath);

    console.log(`Ảnh đã được chuyển thành công sang PNG: ${outputPath}`);
  } catch (err) {
    console.error("Lỗi khi chuyển đổi DICOM sang PNG:", err);
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

module.exports = { dicomToPng };
