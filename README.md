# 🩻 X-Ray Diagnosis AI

**X-Ray Diagnosis AI** là một hệ thống trí tuệ nhân tạo giúp **phân loại ảnh X-quang phổi** thành 2 nhóm:

- ✅ **Normal** – Phổi bình thường
- ⚠️ **Pneumonia** – Phổi viêm

Hệ thống gồm 2 phần chính:

- Mô hình AI huấn luyện bằng **PyTorch + ResNet-50**
- Backend **Node.js** sử dụng mô hình ONNX để dự đoán

---

## 📚 Mục lục

1. [Cấu trúc dự án](#cấu-trúc-dự-án)
2. [Cách hoạt động](#cách-hoạt-động)
3. [Hướng dẫn cài đặt](#hướng-dẫn-cài-đặt)
4. [Chi tiết các thành phần](#chi-tiết-các-thành-phần)
5. [API Backend](#api-backend)
6. [Kết quả Inference](#kết-quả-inference)

---

## 🚧 Tiến độ hiện tại

Dự án đang trong giai đoạn phát triển ban đầu. Các phần đã hoàn thành:

- ✅ **API Upload DICOM**: Cho phép người dùng upload file ảnh X-quang định dạng `.dcm` (DICOM).
- ✅ **API Analyze**: Phân tích ảnh vừa upload và trả về kết quả dự đoán bệnh phổi.
- ✅ **Tích hợp ONNX**: Model AI đã được chuyển sang định dạng ONNX để inference trên Node.js.
- ✅ **Huấn luyện mô hình AI**:
  - Sử dụng **PyTorch** kết hợp với mô hình **ResNet-50**.
  - Dataset hiện tại mới chỉ có **1 ảnh X-quang** (dùng cho test/train/val).
  - ⚠️ **Lưu ý**: Sẽ cập nhật thêm nhiều dữ liệu sau để tăng độ chính xác.

---

## 🧠 Kế hoạch tiếp theo

- [ ]  Thêm Explainability (XAI): Tích hợp Grad-CAM / Heatmap để highlight vùng ảnh khiến mô hình quyết định.
- [ ]  Checklist
- [ ]  Dựng thêm pipeline để xử lý fine-tune định kỳ / Giải quyết bài toán model .pth được cập nhật sau khi fine-tune (Redis)

Với Multimodal AI, chỉ cần bác sĩ nhập vào các thông tin lâm sàng như:
  ✅ Input từ bác sĩ (Checklist):
  - 👶 Giới tính (Nam / Nữ)
  - 🎂 Tuổi (hoặc nhóm tuổi: <1, 1–5, >5 tuổi)
  - 🌡️ Sốt (Có / Không)
  - 😮‍💨 Ho (Có / Không)
  - 💨 Thở nhanh (Có / Không)
  - ⚖️ Cân nặng giảm (Có / Không)
  - 🩸 SpO2 thấp (nếu có)
  (v.v…)

✅ Output mà mô hình có thể tự động dự đoán:
🔍 Bé có mắc bệnh gì không?
  - Pneumonia
  - Viêm tiểu phế quản 
  - Viêm phổi cấp
  - Normal

📋 Gợi ý các triệu chứng phụ kèm theo
💬 Giải thích vùng ảnh (Grad-CAM) nếu có ảnh X-quang

