import axios from "axios";
import fs from "fs";
import path from "path";
import csv from "csv-parser";

// const csvFilePath = "./data.csv"; // đường dẫn file CSV
const downloadFolder = "./NORMAL"; // thư mục lưu ảnh
const batchSize = 20; // số ảnh tải mỗi lần
const delayMs = 10000; // delay 10 giây
const idsFile = "./image_ids.json";

if (!fs.existsSync(downloadFolder)) {
  fs.mkdirSync(downloadFolder);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// Hàm tải 1 ảnh và lưu
async function downloadImage(imageId) {
  const url = `https://physionet.org/files/vindr-pcxr/1.0.0/test/${imageId}.dicom?download`;
  const filePath = path.join(downloadFolder, `${imageId}.dicom`);

  console.log(`Downloading ${imageId}.dicom from ${url}...`);

  try {
    const response = await axios.get(url, {
      responseType: "stream",
      headers: {
        accept:
          "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "accept-language": "en-GB,en-US;q=0.9,en;q=0.8",
        cookie: "_gid=GA1.2.1220436327.1748311355; csrftoken=7sMTY4Urs4SeQC8c7Uyght2VzI09ck12; sessionid=ajlh5yaa7i6g2shizzmt6sggqjy09k3x; _ga_YKC8ZQQ4FF=GS2.1.s1748311356$o23$g1$t1748312755$j0$l0$h0; _ga=GA1.2.256742238.1747018865; _gat_gtag_UA_87592301_7=1",
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

async function main() {
  const ids = JSON.parse(fs.readFileSync(idsFile, "utf-8"));
  await downloadInBatches(ids, batchSize, delayMs);
  console.log("🎉 Đã tải xong tất cả ảnh!");
}

main();
