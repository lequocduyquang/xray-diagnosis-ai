/**
 * Chuyển đổi logits thành xác suất bằng softmax
 * @param {Object} logits Logits từ mô hình ONNX
 * @returns {Array<number>} Xác suất cho từng lớp
 */
export function softmax(logits) {
  const expValues = Object.values(logits).map((x) => Math.exp(x));
  const sumExp = expValues.reduce((a, b) => a + b, 0);
  return expValues.map((x) => x / sumExp);
}

/**
 * Lấy lớp dự đoán từ xác suất
 * @param {Array<number>} probabilities Xác suất cho từng lớp
 * @returns {number} Lớp dự đoán (chỉ số của lớp có xác suất cao nhất)
 */
export function getPredictedClass(probabilities) {
  return probabilities.indexOf(Math.max(...probabilities));
}
