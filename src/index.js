import express from "express";
import dotenv from "dotenv";
import imageRoutes from "./routes/imageRoutes";

dotenv.config();

const app = express();
app.use(express.json());
app.use("/api", imageRoutes);

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Server đang chạy tại http://localhost:${PORT}`);
});
