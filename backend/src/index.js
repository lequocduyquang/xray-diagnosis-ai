import express from "express";
import dotenv from "dotenv";
import cors from "cors";
import imageRoutes from "./routes/imageRoutes.js";

dotenv.config();

const app = express();

const allowedOrigins = ["https://xray-ui.vercel.app"];

app.use(
  cors((origin, callback) => {
    if (!origin || allowedOrigins.indexOf(origin) !== -1) {
      callback(null, true);
    } else {
      callback(new Error("Not allowed by CORS"));
    }
  })
);

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.get("/health", (req, res) => {
  res.status(200).json({
    status: "OK",
    message: "Server is healthy",
    timestamp: new Date().toISOString(),
  });
});

app.use("/api", imageRoutes);

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Server đang chạy tại http://localhost:${PORT}`);
});
