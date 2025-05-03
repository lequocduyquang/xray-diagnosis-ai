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
