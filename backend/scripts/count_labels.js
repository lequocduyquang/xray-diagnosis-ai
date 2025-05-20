import fs from "fs";
import csv from "csv-parser";

const csvFile = "./image_labels_train.csv";
const labelCounts = {};

let labelNames = [];

fs.createReadStream(csvFile)
  .pipe(csv())
  .on("headers", (headers) => {
    // Lấy tên các nhãn bệnh (bỏ image_id, rad_ID)
    labelNames = headers.filter((h) => h !== "image_id" && h !== "rad_ID");
    labelNames.forEach((label) => (labelCounts[label] = 0));
  })
  .on("data", (row) => {
    labelNames.forEach((label) => {
      if (row[label] === "1.0") {
        labelCounts[label]++;
      }
    });
  })
  .on("end", () => {
    console.log("Thống kê số lượng ảnh theo từng nhãn bệnh:");
    for (const label of labelNames) {
      console.log(`${label}: ${labelCounts[label]}`);
    }
  });
