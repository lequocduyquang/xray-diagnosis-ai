import fs from "fs";
import csv from "csv-parser";

const csvFile = "./image_labels_train.csv";
const outputFile = "./random_1000_image_ids.json";
const targetCount = 1000;

const smallLabels = [
  "Situs inversus",
  "Pleuro-pneumonia",
  "Diagphramatic hernia",
  "Tuberculosis",
  "Congenital emphysema",
  "CPAM",
  "Hyaline membrane disease",
  "Mediastinal tumor",
  "Lung tumor",
];

const labelNames = [];
const labelToIds = {};

fs.createReadStream(csvFile)
  .pipe(csv())
  .on("headers", (headers) => {
    headers.forEach((h) => {
      if (h !== "image_id" && h !== "rad_ID" && h !== "No finding") {
        labelNames.push(h);
        labelToIds[h] = new Set();
      }
    });
  })
  .on("data", (row) => {
    for (const label of labelNames) {
      if (row[label] === "1.0") {
        labelToIds[label].add(row["image_id"]);
      }
    }
  })
  .on("end", () => {
    // Lấy full tất cả id của smallLabels trước
    const selectedSet = new Set();
    smallLabels.forEach((label) => {
      labelToIds[label].forEach((id) => selectedSet.add(id));
    });

    // Tính số lượng còn thiếu
    const needMore = targetCount - selectedSet.size;

    // Gom tất cả id của bigLabels, loại bỏ id đã có ở selectedSet
    const bigLabels = labelNames.filter((l) => !smallLabels.includes(l));
    const bigLabelIds = new Set();
    bigLabels.forEach((label) => {
      labelToIds[label].forEach((id) => {
        if (!selectedSet.has(id)) bigLabelIds.add(id);
      });
    });

    // Shuffle bigLabelIds
    const bigLabelIdsArr = Array.from(bigLabelIds);
    for (let i = bigLabelIdsArr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [bigLabelIdsArr[i], bigLabelIdsArr[j]] = [
        bigLabelIdsArr[j],
        bigLabelIdsArr[i],
      ];
    }

    // Lấy thêm cho đủ targetCount
    bigLabelIdsArr.slice(0, needMore).forEach((id) => selectedSet.add(id));

    const selected = Array.from(selectedSet).slice(0, targetCount);
    fs.writeFileSync(outputFile, JSON.stringify(selected, null, 2));
    console.log(
      `Đã random và lưu ${selected.length} image_id vào ${outputFile}`
    );

    // Thống kê số lượng mỗi loại trong tập đã pick (unique image_id)
    const labelPickCount = {};
    for (const label of labelNames) {
      labelPickCount[label] = 0;
    }
    selected.forEach((id) => {
      for (const label of labelNames) {
        if (labelToIds[label].has(id)) {
          labelPickCount[label]++;
        }
      }
    });
    console.log("Thống kê số lượng mỗi nhãn trong tập đã pick:");
    for (const label of labelNames) {
      console.log(`${label}: ${labelPickCount[label]}`);
    }
  });
