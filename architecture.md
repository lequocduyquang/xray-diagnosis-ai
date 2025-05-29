Frontend
   │
   ▼
Upload X-ray + info ➜ NodeJS API ➜ Save / Inference
                                       │
     ┌────────────────────────────┐    ▼
     │  (User clicks GradCAM btn) │→ Call Python API (/gradcam)
     └────────────────────────────┘    │
                                       ▼
                          Python backend gen GradCAM + upload Cloudinary
                                       ▼
                         NodeJS nhận URL, gửi về frontend
