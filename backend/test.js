import fs from "fs";
import csvParser from "csv-parser";
import { parse } from "json2csv";

// Đường dẫn file gốc và file đích
const inputCsvFile = "filtered_image_labels_train.csv";
const outputCsvFile = "filtered_labels.csv";

// Các cột cần giữ lại
const selectedColumns = [
  "image_id",
  "Bronchitis",
  "Brocho-pneumonia",
  "Other disease",
  "Bronchiolitis",
  "Pneumonia",
];

// Mảng để lưu dữ liệu đã lọc
const filteredData = [];

// Đọc file CSV gốc và lọc dữ liệu
fs.createReadStream(inputCsvFile)
  .pipe(csvParser())
  .on("data", (row) => {
    // Bỏ qua các dòng có Rare diseases = 1.0
    if (row["Rare diseases"] !== "1.0") {
      // Chỉ giữ lại các cột được chọn
      const filteredRow = {};
      selectedColumns.forEach((col) => {
        filteredRow[col] = row[col];
      });
      filteredData.push(filteredRow);
    }
  })
  .on("end", () => {
    console.log(`Đã lọc ${filteredData.length} dòng.`);

    // Ghi dữ liệu đã lọc vào file CSV mới
    const csvData = parse(filteredData);
    fs.writeFileSync(outputCsvFile, csvData, "utf8");
    console.log(`Dữ liệu đã được ghi vào file: ${outputCsvFile}`);
  });