import fs from "fs";
import csv from "csv-parser";

const INPUT_CSV = "image_labels_train.csv";
const MAX_IMAGES_PER_LABEL = 100;

const labelMap = {};
const selectedIds = new Set();

fs.createReadStream(INPUT_CSV)
  .pipe(csv())
  .on("data", (row) => {
    const imageId = row["image_id"];
    // Duyệt qua từng label (bỏ qua 2 cột đầu)
    Object.keys(row)
      .filter((key) => key !== "image_id" && key !== "rad_ID")
      .forEach((label) => {
        if (row[label] === "1.0") {
          if (!labelMap[label]) labelMap[label] = [];
          labelMap[label].push(imageId);
        }
      });
  })
  .on("end", () => {
    console.log("✅ Đã đọc xong CSV");

    for (const label in labelMap) {
      const imageList = labelMap[label];
      const selected = imageList.slice(0, MAX_IMAGES_PER_LABEL);

      selected.forEach((id) => selectedIds.add(id));
      console.log(`📂 ${label}: chọn ${selected.length} ảnh`);
    }

    // Xuất ra file JSON để dễ tải về sau
    fs.writeFileSync(
      "selected_image_ids.json",
      JSON.stringify([...selectedIds], null, 2)
    );
    console.log(
      `\n🎯 Tổng cộng chọn ${selectedIds.size} ảnh. Lưu vào file selected_image_ids.json`
    );
  });
