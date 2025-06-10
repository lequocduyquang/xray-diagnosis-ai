import pkg from "pg";
const { Pool } = pkg;
import dotenv from "dotenv";

dotenv.config();

// Cấu hình kết nối PostgreSQL
const pool = new Pool({
  user: process.env.DB_USER || "postgres",
  host: process.env.DB_HOST || "localhost",
  database: process.env.DB_NAME || "xray_diagnosis",
  password: process.env.DB_PASSWORD || "password",
  port: process.env.DB_PORT || 5432,
  max: 20, // Số kết nối tối đa trong pool
  idleTimeoutMillis: 30000, // Thời gian timeout cho kết nối idle
  connectionTimeoutMillis: 2000, // Thời gian timeout khi tạo kết nối mới
});

// Test kết nối database
pool.on("connect", () => {
  console.log("✅ Đã kết nối thành công với PostgreSQL database");
});

pool.on("error", (err) => {
  console.error("❌ Lỗi kết nối PostgreSQL:", err);
});

// Khởi tạo database và tạo bảng nếu chưa tồn tại
export async function initializeDatabase() {
  try {
    const client = await pool.connect();

    // Tạo bảng images nếu chưa tồn tại
    const createTableQuery = `
      CREATE TABLE IF NOT EXISTS images (
        id SERIAL PRIMARY KEY,
        cloudinary_id VARCHAR(255) NOT NULL,
        cloudinary_url TEXT NOT NULL,
        model_name VARCHAR(100) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      );
    `;

    await client.query(createTableQuery);
    console.log("✅ Đã tạo bảng images thành công");

    client.release();
  } catch (error) {
    console.error("❌ Lỗi khởi tạo database:", error);
    throw error;
  }
}

export default pool;
