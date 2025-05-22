import fs from "fs";
import csv from "csv-parser";

const csvFile = "./image_labels_train.csv";
const outputFile = "./random_500_no_finding_image_ids.json";
const targetCount = 500;

const noFindingIds = new Set();

fs.createReadStream(csvFile)
  .pipe(csv())
  .on("data", (row) => {
    if (row["No finding"] === "1.0") {
      noFindingIds.add(row["image_id"]);
    }
  })
  .on("end", () => {
    // Shuffle
    const allIds = Array.from(noFindingIds);
    for (let i = allIds.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [allIds[i], allIds[j]] = [allIds[j], allIds[i]];
    }
    const selected = allIds.slice(0, targetCount);
    fs.writeFileSync(outputFile, JSON.stringify(selected, null, 2));
    console.log(
      `Đã random và lưu ${selected.length} image_id của nhãn "No finding" vào ${outputFile}`
    );
  });
