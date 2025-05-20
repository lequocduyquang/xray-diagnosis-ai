import fs from "fs";
import csv from "csv-parser";

const filePath = "image_labels_train.csv"; // Thay bằng đường dẫn thực tế tới file CSV
const results = [];

function countImagesByLabel(filePath, callback) {
  const labelCounts = {};

  fs.createReadStream(filePath)
    .pipe(csv())
    .on("data", (row) => {
      // Lấy danh sách nhãn (bỏ cột đầu tiên, giả sử là ID hoặc tên ảnh)
      const labels = Object.keys(row).slice(1);

      // Duyệt qua từng nhãn và đếm nếu giá trị là 1.0
      labels.forEach((label) => {
        if (parseFloat(row[label]) === 1.0) {
          labelCounts[label] = (labelCounts[label] || 0) + 1;
        }
      });
    })
    .on("end", () => {
      // Trả về kết quả qua callback
      callback(null, labelCounts);
    })
    .on("error", (error) => {
      callback(error, null);
    });
}

countImagesByLabel(filePath, (error, counts) => {
  if (error) {
    console.error("Lỗi khi đọc file CSV:", error.message);
    return;
  }
  console.log("Số lượng hình theo nhãn:");
  for (const [label, count] of Object.entries(counts)) {
    console.log(`${label}: ${count}`);
  }
});

// fs.createReadStream(filePath)
//   .pipe(csv())
//   .on("data", (row) => {
//     // Giả sử cột đầu tiên là ID hoặc tên ảnh, các cột còn lại là label
//     const labels = Object.values(row).slice(1); // Bỏ cột đầu tiên
//     // Đếm số label có giá trị 1.0
//     const trueCount = labels.reduce((count, value) => {
//       return count + (parseFloat(value) === 1.0 ? 1 : 0);
//     }, 0);

//     // Nếu có ít nhất 2 label là 1.0, lưu record
//     if (trueCount >= 2) {
//       results.push({ ...row, true_count: trueCount });
//     }
//   })
//   .on("end", () => {
//     if (results.length > 0) {
//       console.log(
//         `Có ${results.length} record(s) với ít nhất 2 label là True (1.0):`
//       );
//       console.log(results);
//     } else {
//       console.log("Không có record nào có 2 label là True (1.0).");
//     }
//   })
//   .on("error", (error) => {
//     console.error("Lỗi khi đọc file CSV:", error.message);
//   });
