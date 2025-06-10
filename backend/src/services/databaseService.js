import pool from "../config/database.js";

/**
 * Lưu thông tin ảnh vào database
 * @param {Object} imageData - Dữ liệu ảnh cần lưu
 * @param {string} imageData.cloudinaryId - ID của ảnh trên Cloudinary
 * @param {string} imageData.cloudinaryUrl - URL của ảnh trên Cloudinary
 * @param {string} imageData.modelName - Tên model AI được sử dụng
 * @param {string} imageData.originalFilename - Tên file gốc
 * @param {number} imageData.fileSize - Kích thước file (bytes)
 * @param {string} imageData.mimeType - MIME type của file
 * @returns {Promise<Object>} - Kết quả lưu database
 */
export async function saveImageToDatabase(imageData) {
  try {
    const { cloudinaryId, cloudinaryUrl, modelName } = imageData;

    // Kiểm tra xem ảnh đã tồn tại trong database chưa
    const existingImage = await getImageByCloudinaryId(cloudinaryId);

    if (existingImage) {
      // Nếu ảnh đã tồn tại, cập nhật model_name
      const updateQuery = `
        UPDATE images 
        SET model_name = $1, updated_at = CURRENT_TIMESTAMP
        WHERE cloudinary_id = $2
        RETURNING id, cloudinary_id, model_name, updated_at;
      `;

      const result = await pool.query(updateQuery, [modelName, cloudinaryId]);

      console.log(
        `✅ Đã cập nhật model_name thành "${modelName}" cho ảnh ID: ${result.rows[0].id}`
      );
      return {
        success: true,
        data: result.rows[0],
        action: "updated",
      };
    } else {
      // Nếu ảnh chưa tồn tại, thêm mới
      const insertQuery = `
        INSERT INTO images (cloudinary_id, cloudinary_url, model_name)
        VALUES ($1, $2, $3)
        RETURNING id, cloudinary_id, model_name, created_at;
      `;

      const values = [cloudinaryId, cloudinaryUrl, modelName];

      const result = await pool.query(insertQuery, values);

      console.log(
        `✅ Đã lưu ảnh mới vào database với ID: ${result.rows[0].id}`
      );
      return {
        success: true,
        data: result.rows[0],
        action: "inserted",
      };
    }
  } catch (error) {
    console.error("❌ Lỗi khi lưu/cập nhật ảnh vào database:", error);
    throw error;
  }
}

/**
 * Cập nhật model_name cho ảnh đã tồn tại
 * @param {string} cloudinaryId - ID của ảnh trên Cloudinary
 * @param {string} modelName - Tên model AI mới
 * @returns {Promise<Object>} - Kết quả cập nhật
 */
export async function updateModelName(cloudinaryId, modelName) {
  try {
    const query = `
      UPDATE images 
      SET model_name = $1, updated_at = CURRENT_TIMESTAMP
      WHERE cloudinary_id = $2
      RETURNING id, cloudinary_id, model_name, updated_at;
    `;

    const result = await pool.query(query, [modelName, cloudinaryId]);

    if (result.rows.length > 0) {
      console.log(
        `✅ Đã cập nhật model_name thành "${modelName}" cho cloudinary_id: ${cloudinaryId}`
      );
      return {
        success: true,
        data: result.rows[0],
      };
    } else {
      console.log(`⚠️ Không tìm thấy ảnh với cloudinary_id: ${cloudinaryId}`);
      return {
        success: false,
        message: "Image not found",
      };
    }
  } catch (error) {
    console.error("❌ Lỗi khi cập nhật model_name:", error);
    throw error;
  }
}

/**
 * Lấy thông tin ảnh theo cloudinary_id
 * @param {string} cloudinaryId - ID của ảnh trên Cloudinary
 * @returns {Promise<Object|null>} - Thông tin ảnh hoặc null nếu không tìm thấy
 */
export async function getImageByCloudinaryId(cloudinaryId) {
  try {
    const query = `
      SELECT * FROM images 
      WHERE cloudinary_id = $1
      ORDER BY created_at DESC
      LIMIT 1;
    `;

    const result = await pool.query(query, [cloudinaryId]);

    if (result.rows.length > 0) {
      return result.rows[0];
    }

    return null;
  } catch (error) {
    console.error("❌ Lỗi khi lấy thông tin ảnh:", error);
    throw error;
  }
}

/**
 * Lấy danh sách ảnh theo model name
 * @param {string} modelName - Tên model AI
 * @param {number} limit - Số lượng kết quả tối đa
 * @param {number} offset - Số lượng bỏ qua
 * @returns {Promise<Array>} - Danh sách ảnh
 */
export async function getImagesByModel(modelName, limit = 10, offset = 0) {
  try {
    const query = `
      SELECT * FROM images 
      WHERE model_name = $1
      ORDER BY created_at DESC
      LIMIT $2 OFFSET $3;
    `;

    const result = await pool.query(query, [modelName, limit, offset]);
    return result.rows;
  } catch (error) {
    console.error("❌ Lỗi khi lấy danh sách ảnh theo model:", error);
    throw error;
  }
}

/**
 * Lấy thống kê tổng quan
 * @returns {Promise<Object>} - Thống kê database
 */
export async function getDatabaseStats() {
  try {
    const statsQuery = `
      SELECT 
        COUNT(*) as total_images,
        COUNT(DISTINCT model_name) as total_models,
        model_name,
        COUNT(*) as count_by_model
      FROM images 
      GROUP BY model_name
      ORDER BY count_by_model DESC;
    `;

    const result = await pool.query(statsQuery);

    return {
      totalImages: result.rows.reduce(
        (sum, row) => sum + parseInt(row.count_by_model),
        0
      ),
      totalModels: result.rows.length,
      models: result.rows,
    };
  } catch (error) {
    console.error("❌ Lỗi khi lấy thống kê database:", error);
    throw error;
  }
}
