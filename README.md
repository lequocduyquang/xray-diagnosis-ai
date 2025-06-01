# 🩻 X-Ray Diagnosis AI

**X-Ray Diagnosis AI** là một hệ thống trí tuệ nhân tạo hỗ trợ bác sĩ tại Bệnh viện Nhi đồng 2 trong việc phân tích ảnh X-quang phổi trẻ em. Hệ thống kết hợp thông tin lâmL sàng từ bác sĩ và phân tích ảnh X-quang để chẩn đoán các bệnh lý hô hấp, với mục tiêu tăng độ chính xác và hỗ trợ ra quyết định lâm sàng.

Hệ thống hiện tại:

- **Phân loại nhị phân**: Xác định phổi **Normal** (bình thường) hoặc **Pneumonia** (viêm phổi) bằng hai mô hình ResNet50.
- **Phân loại đa nhãn**: Nếu phát hiện Pneumonia, phân loại chi tiết thành 5 bệnh lý (Bronchitis, Brocho-pneumonia, Other disease, Bronchiolitis, Pneumonia) bằng mô hình DenseNet121.
- **Tích hợp thông tin lâm sàng**: Điều chỉnh xác suất chẩn đoán dựa trên chẩn đoán ban đầu và triệu chứng từ bác sĩ, với cơ chế xử lý mâu thuẫn giữa AI và lâm sàng.
- **Giao diện người dùng**: React UI cho phép bác sĩ upload ảnh X-quang và nhập thông tin lâm sàng (chẩn đoán ban đầu, triệu chứng).

---

## 📚 Mục lục

1. [Cấu trúc dự án](#cấu-trúc-dự-án)
2. [Cách hoạt động](#cách-hoạt-động)
3. [Hướng dẫn cài đặt](#hướng-dẫn-cài-đặt)
4. [Chi tiết các thành phần](#chi-tiết-các-thành-phần)
5. [API Backend](#api-backend)
6. [Kết quả Inference](#kết-quả-inference)
7. [Kế hoạch tiếp theo](#kế-hoạch-tiếp-theo)

---

## 🚧 Cấu trúc dự án

```
xray-diagnosis-ai/
├── ml-models/                 # Thư mục chứa các mô hình ONNX
│   ├── resnet50_v1.onnx      # ResNet50 v1 (train trên 2000 ảnh trẻ 1–5 tuổi, Quảng Châu)
│   ├── resnet50_v2.onnx      # ResNet50 v2 (train trên 1700 ảnh trẻ <10 tuổi, Việt Nam)
│   └── densenet121.onnx      # DenseNet121 (train trên 1000 ảnh Việt Nam, phân loại 5 bệnh)
├── services/                 # Chứa logic xử lý AI
│   └── onnxService.js        # Hàm phân tích ảnh và điều chỉnh xác suất với clinical_info
├── controllers/              # API controllers
│   └── xray_controller.js    # Xử lý request upload ảnh và clinical_info
├── public/                   # Tài nguyên tĩnh
│   └── Logo_ND2.png          # Logo Bệnh viện Nhi đồng 2
├── app/                      # Frontend React
│   └── index.tsx             # Giao diện nhập liệu và hiển thị kết quả
├── utils/                    # Hàm tiện ích
│   └── calculation.js        # Hàm softmax, sigmoid
├── README.md                 # Tài liệu hướng dẫn
└── package.json              # Dependencies và scripts
```

---

## 🧠 Cách hoạt động

Hệ thống hoạt động theo quy trình sau:

1. **Nhập liệu**:
   - Bác sĩ upload ảnh X-quang (định dạng `.jpeg`, `.png`, hoặc `.dicom`) qua giao diện React.
   - Bác sĩ nhập thông tin lâm sàng:
     - **Chẩn đoán ban đầu** (`initial_diagnosis`): Normal, Pneumonia, Bronchitis, Brocho-pneumonia, Other disease, Bronchiolitis.
     - **Triệu chứng** (`symptoms`): Sốt, khó thở, ho, thở khò khè.
2. **Phân tích nhị phân**:
   - Hai mô hình ResNet50 (v1 và v2) chạy song song để phân loại Normal hoặc Pneumonia.
   - Kết quả được kết hợp bằng trọng số (0.5 cho v1, 0.5 cho v2).
   - Xác suất được điều chỉnh dựa trên thông tin lâm sàng (ví dụ: tăng xác suất Pneumonia nếu bác sĩ chọn Pneumonia).
3. **Phân loại đa nhãn** (nếu phát hiện Pneumonia):
   - Mô hình DenseNet121 phân loại 5 bệnh lý: Bronchitis, Brocho-pneumonia, Other disease, Bronchiolitis, Pneumonia.
   - Xác suất được điều chỉnh dựa trên thông tin lâm sàng.
4. **Xử lý mâu thuẫn**:
   - Nếu chẩn đoán AI và bác sĩ mâu thuẫn (ví dụ: bác sĩ chọn Pneumonia, AI chọn Normal), hệ thống:
     - Điều chỉnh xác suất bằng trọng số (`W_clinical`): 1.5 cho chẩn đoán của bác sĩ, 0.5 cho nhãn mâu thuẫn, 0.8 cho nhãn khác.
     - Đưa ra cảnh báo nếu độ mâu thuẫn > 0.4.
5. **Kết quả**:
   - Hiển thị: Xác suất nhị phân, top 3 chẩn đoán phụ, tất cả chẩn đoán phụ, thông tin lâm sàng, và cảnh báo mâu thuẫn.

---

## 🔧 Hướng dẫn cài đặt

### Yêu cầu

- **Node.js**: v16 hoặc cao hơn
- **Python**: v3.8+ (cho huấn luyện mô hình, nếu cần)
- **Dependencies**:
  ```bash
  npm install onnxruntime-node jimp node-fetch
  ```
  ```bash
  pip install torch torchvision onnx
  ```

### Cài đặt

1. **Clone repository**:

   ```bash
   git clone <repository-url>
   cd xray-diagnosis-ai
   ```

2. **Cài đặt dependencies**:

   ```bash
   npm install
   ```

3. **Đặt mô hình ONNX**:

   - Copy các file `resnet50_v1.onnx`, `resnet50_v2.onnx`, `densenet121.onnx` vào thư mục `ml-models/`.

4. **Chạy ứng dụng**:
   ```bash
   npm start
   ```
   - Truy cập giao diện tại: `http://localhost:3000`
   - API endpoint: `http://localhost:3000/api/analyze`

---

## 🛠 Chi tiết các thành phần

### 1. Mô hình AI

- **ResNet50 v1**: Huấn luyện trên 2000 ảnh X-quang trẻ em 1–5 tuổi từ Quảng Châu, phân loại Normal/Pneumonia.
- **ResNet50 v2**: Huấn luyện trên 1700 ảnh X-quang trẻ em <10 tuổi từ Việt Nam, phân loại Normal/Pneumonia.
- **DenseNet121**: Huấn luyện trên 1000 ảnh X-quang Việt Nam, phân loại 5 bệnh lý: Bronchitis, Brocho-pneumonia, Other disease, Bronchiolitis, Pneumonia.
- **Định dạng**: Mô hình được export từ PyTorch sang ONNX để sử dụng trong Node.js.

### 2. Backend (Node.js)

- **Framework**: Node.js với `onnxruntime-node` để chạy inference.
- **API**: Xử lý upload ảnh và thông tin lâm sàng, trả về kết quả phân tích.
- **Logic**:
  - Tiền xử lý ảnh (resize, normalize).
  - Chạy mô hình ResNet50 v1/v2 song song, kết hợp xác suất.
  - Chạy DenseNet121 nếu phát hiện Pneumonia.
  - Điều chỉnh xác suất dựa trên `clinical_info` (chẩn đoán ban đầu, triệu chứng).
  - Xử lý mâu thuẫn giữa AI và bác sĩ.

### 3. Frontend (React)

- **Giao diện**: React với Tailwind CSS.
- **Chức năng**:
  - Upload ảnh X-quang (`.jpeg`, `.png`, `.dicom`).
  - Nhập thông tin lâm sàng (dropdown cho chẩn đoán, checkbox cho triệu chứng).
  - Hiển thị kết quả: Xác suất nhị phân, chẩn đoán phụ, thông tin lâm sàng, cảnh báo mâu thuẫn.

### 4. Tích hợp thông tin lâm sàng

- **Input**:
  - `initial_diagnosis`: Chọn từ danh sách nhãn hợp lệ.
  - `symptoms`: Chọn từ danh sách (sốt, khó thở, ho, thở khò khè).
- **Xử lý**:
  - Điều chỉnh xác suất bằng trọng số: 1.5 cho nhãn bác sĩ chọn, 0.5 cho nhãn mâu thuẫn, 0.8 cho nhãn khác.
  - Cảnh báo nếu mâu thuẫn lớn (độ mâu thuẫn > 0.4).
- **Output**: Kết quả bao gồm xác suất điều chỉnh, thông tin lâm sàng, và khuyến nghị.

---

## 🌐 API Backend

### Endpoint: `/api/analyze`

- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Body**:
  - `image`: File ảnh X-quang (`.jpeg`, `.png`, `.dicom`)
  - `clinical_info`: JSON chứa:
    - `initial_diagnosis`: Chuỗi (Normal, Pneumonia, Bronchitis, v.v.)
    - `symptoms`: Mảng chuỗi (["fever", "dyspnea", "cough", "wheezing"])
- **Response**:
  ```json
  {
    "success": true,
    "stage": "binary-classification" | "multi-label-diagnosis",
    "message": "Result: Normal" | "Result: Pneumonia with subtypes",
    "data": {
      "clinical_info": {
        "initial_diagnosis": "Pneumonia",
        "symptoms": ["fever", "dyspnea", "cough"]
      },
      "binaryProbabilities": { "Normal": 0.595, "Pneumonia": 0.605 },
      "predictedClass": "Pneumonia",
      "classLabels": ["Normal", "Pneumonia"],
      "multiLabelTop": {
        "0": { "label": "Pneumonia", "score": 0.429 },
        "1": { "label": "Bronchitis", "score": 0.143 },
        "2": { "label": "Bronchiolitis", "score": 0.086 }
      },
      "allMultiLabelScores": [
        { "label": "Bronchitis", "score": 0.143 },
        { "label": "Brocho-pneumonia", "score": 0.057 },
        { "label": "Other disease", "score": 0.029 },
        { "label": "Bronchiolitis", "score": 0.086 },
        { "label": "Pneumonia", "score": 0.429 }
      ],
      "warnings": [
        "Cảnh báo: AI cho xác suất Normal cao (39.5%), nhưng bác sĩ chẩn đoán Pneumonia. Đề nghị xét nghiệm bổ sung (máu, CRP, CT) hoặc theo dõi sát."
      ]
    }
  }
  ```

---

## 📈 Kết quả Inference

- **Nhị phân**: Xác suất Normal/Pneumonia sau khi kết hợp ResNet50 v1 (30%) và v2 (70%), điều chỉnh bởi thông tin lâm sàng.
- **Đa nhãn**: Nếu phát hiện Pneumonia, trả về xác suất 5 bệnh lý, với top 3 được hiển thị nổi bật.
- **Mâu thuẫn**: Cảnh báo nếu chẩn đoán AI và bác sĩ không khớp (độ mâu thuẫn > 0.4), kèm khuyến nghị xét nghiệm hoặc theo dõi.
- **Thời gian xử lý**: ~1 giây cho mỗi ảnh (tùy cấu hình server).

---

## 🚀 Kế hoạch tiếp theo

- [ ] **Explainability (XAI)**: Tích hợp Grad-CAM/Heatmap để highlight vùng ảnh X-quang dẫn đến chẩn đoán.
- [ ] **Mở rộng thông tin lâm sàng**:
  - Thêm các trường: Giới tính, nhóm tuổi (<1, 1–5, >5), SpO2, cân nặng giảm, thở nhanh.
  - Tích hợp thông tin này vào mô hình (multimodal learning).
- [ ] **Fine-tune định kỳ**:
  - Xây dựng pipeline để cập nhật mô hình `.pth` và export sang `.onnx`.
  - Sử dụng Redis để quản lý phiên bản mô hình.
- [ ] **Tăng dataset**:
  - Thu thập thêm ảnh X-quang và thông tin lâm sàng từ Bệnh viện Nhi đồng 2.
  - Tham khảo dataset từ Kaggle, PhysioNet
- [ ] **Cải thiện giao diện**:
  - Thêm validate client-side cho thông tin lâm sàng.
  - Hỗ trợ đa ngôn ngữ (tiếng Việt, tiếng Anh).
- [ ] **Triển khai sản phẩm**:
  - Tối ưu hóa hiệu suất trên server production.
  - Đảm bảo tuân thủ bảo mật y tế (Luật An ninh mạng Việt Nam).

---

## 📜 Giấy phép

Dự án được phát triển cho **Bệnh viện Nhi đồng 2**. Vui lòng liên hệ nhóm phát triển để biết thêm chi tiết về quyền sử dụng.

---

**Built by Quang Le with 🧡**
