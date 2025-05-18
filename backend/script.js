import axios from "axios";
import fs from "fs";
import path from "path";
import csv from "csv-parser";

const csvFilePath = "./data.csv"; // đường dẫn file CSV
const downloadFolder = "./downloads"; // thư mục lưu ảnh
const batchSize = 10; // số ảnh tải mỗi lần
const delayMs = 10000; // delay 10 giây

if (!fs.existsSync(downloadFolder)) {
  fs.mkdirSync(downloadFolder);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// Hàm tải 1 ảnh và lưu
async function downloadImage(imageId) {
  const url = `https://physionet.org/files/vindr-pcxr/1.0.0/train/${imageId}.dicom?download`;
  const filePath = path.join(downloadFolder, `${imageId}.dicom`);

  console.log(`Downloading ${imageId}.dicom from ${url}...`);

  try {
    const response = await axios.get(url, {
      responseType: "stream",
      headers: {
        accept:
          "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "accept-language": "en-GB,en-US;q=0.9,en;q=0.8",
        cookie:
          "csrftoken=AD7FIJwTrySxtNhTK1UkWWr2tqQGiTFr; sessionid=cmrhb1gvjm0qbbyx8jsv04e1bqxqi2z7; _gid=GA1.2.2041111508.1747471507; _ga=GA1.1.1650955361.1746885588; _ga_YKC8ZQQ4FF=GS2.1.s1747534105$o12$g1$t1747536032$j0$l0$h0",
        priority: "u=0, i",
        referer: "https://physionet.org/content/vindr-pcxr/1.0.0/train/",
        "sec-ch-ua": `"Chromium";v="136", "Google Chrome";v="136", "Not.A/Brand";v="99"`,
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": `"macOS"`,
        "sec-fetch-dest": "document",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "same-origin",
        "sec-fetch-user": "?1",
        "upgrade-insecure-requests": "1",
        "user-agent":
          "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
      },
    });

    // Lưu file từ stream
    const writer = fs.createWriteStream(filePath);
    response.data.pipe(writer);

    return new Promise((resolve, reject) => {
      writer.on("finish", () => {
        console.log(`Downloaded ${imageId}.dicom`);
        resolve();
      });
      writer.on("error", reject);
    });
  } catch (error) {
    console.error(`Error downloading ${imageId}:`, error.message);
  }
}

// Đọc CSV, lấy 1000 image_id đầu tiên
async function getImageIdsFromCSV(filePath, limit = 1000) {
  return new Promise((resolve, reject) => {
    const imageIds = [];
    fs.createReadStream(filePath)
      .pipe(csv())
      .on("data", (row) => {
        if (imageIds.length < limit) {
          imageIds.push(row.image_id);
        }
      })
      .on("end", () => {
        resolve(imageIds);
      })
      .on("error", reject);
  });
}

// Tải theo batch
async function downloadInBatches(imageIds, batchSize, delayMs) {
  for (let i = 0; i < imageIds.length; i += batchSize) {
    const batch = imageIds.slice(i, i + batchSize);
    console.log(`Starting batch ${i / batchSize + 1} (${batch.length} images)`);

    // Tải từng ảnh trong batch tuần tự (hoặc có thể chạy đồng thời nếu muốn)
    for (const imageId of batch) {
      await downloadImage(imageId);
    }

    if (i + batchSize < imageIds.length) {
      console.log(`Waiting for ${delayMs / 1000} seconds before next batch...`);
      await sleep(delayMs);
    }
  }
}

(async () => {
  try {
    const imageIds = await getImageIdsFromCSV(csvFilePath, 1000);
    await downloadInBatches(imageIds, batchSize, delayMs);
    console.log("All downloads finished!");
  } catch (error) {
    console.error("Error:", error);
  }
})();
