import fs from "fs";
import csv from "csv-parser";

const idsFile = "./selected_image_ids.json";
const csvFile = "./image_labels_train.csv";
const outputFile = "./filtered_image_labels_train.csv";

// Đọc danh sách id cần lọc
const selectedIds = new Set(JSON.parse(fs.readFileSync(idsFile, "utf-8")));

// Tạo stream đọc và ghi
const results = [];
fs.createReadStream(csvFile)
  .pipe(csv())
  .on("data", (row) => {
    if (selectedIds.has(row["image_id"])) {
      results.push(row);
    }
  })
  .on("end", () => {
    if (results.length === 0) {
      console.log("Không tìm thấy id nào thỏa mãn.");
      return;
    }
    // Ghi file CSV mới
    const headers = Object.keys(results[0]);
    const lines = [
      headers.join(","),
      ...results.map((row) => headers.map((h) => row[h]).join(",")),
    ];
    fs.writeFileSync(outputFile, lines.join("\n"));
    console.log(`Đã lọc xong. Kết quả lưu ở: ${outputFile}`);
  });
