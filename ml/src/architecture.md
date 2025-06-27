[Ảnh X-quang upload lên]
          |
          v
+------------------------+
| Run song song:         |
|  - ResNet50 v1 (kid) |
|  - ResNet50 v2 (child) |
+------------------------+
          |
          v
[Ensemble hoặc chọn nhãn nếu cả 2 đồng thuận]
          |
          +--> Nếu NORMAL => Return luôn
          |
          +--> Nếu PNEUMONIA =>
                       |
                       v
           Run DenseNet121 (5-label)
                       |
                       v
     Return multi-label result (top-n scores)
