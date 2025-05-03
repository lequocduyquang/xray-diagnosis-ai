import express from "express";
import multer from "multer";
import path from "path";
import { convertDicomToPng } from "./utils/dcom-to-png.js";
import { predictXRay } from "./onnx/predict.js";
import { extractEmbedding } from "./embedding/extract.js";
import { searchSimilarImages } from "./vector-db/search.js";

const app = express();
const upload = multer({ dest: "uploads/" });

app.post("/upload", upload.single("dicom"), async (req, res) => {
  const dicomPath = req.file.path;
  const outputPath = `uploads/${req.file.filename}.png`;

  try {
    await dicomToPng(dicomPath, outputPath);
    res.json({ message: "File converted successfully", path: outputPath });
  } catch (err) {
    res.status(500).json({ error: "Error converting DICOM file" });
  }
});

app.post("/analyze", upload.single("xray"), async (req, res) => {
  try {
    const uploadedPath = req.file.path;
    const fileExt = path.extname(req.file.originalname).toLowerCase();
    let imagePath = uploadedPath;

    // Step 0: Convert DICOM to PNG if necessary
    if (fileExt === ".dcm") {
      imagePath = await convertDicomToPng(uploadedPath);
    }

    // Step 1: Predict with ONNX model
    const { label, confidence } = await predictXRay(imagePath);

    // Step 2: Extract embedding
    const embedding = await extractEmbedding(imagePath);

    // Step 3: Search vector DB
    const similarImages = await searchSimilarImages(embedding);

    res.json({ prediction: label, confidence, similar_images: similarImages });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Something went wrong" });
  }
});

app.listen(3000, () => {
  console.log("Server running on http://localhost:3000");
});
