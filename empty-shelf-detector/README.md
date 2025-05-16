# 🛒 Empty Shelf Detector using YOLOv8

A computer vision project to detect out-of-stock shelf areas in retail or supermarket environments using YOLOv8 object detection. This model helps automate the task of shelf monitoring by identifying empty shelf spaces from images.

---

##  Problem Statement

Retailers lose sales when products go out of stock and are not replenished on time. Manual monitoring is inefficient. This project uses computer vision to detect empty shelves from images and can be integrated into real-time alert systems or edge devices.

---

##  Project Overview

- ✅ Built using YOLOv8n (lightweight, fast object detector)
- ✅ Trained on a labeled dataset of supermarket shelf images
- ✅ Detects `empty_shelf` class with high precision
- ✅ Visualizes detections using bounding boxes
- ✅ Includes training notebook, results, and sample inference

---

##  Tools & Technologies

- **YOLOv8** (Ultralytics)
- **PyTorch**
- **OpenCV**
- **Google Colab**
- **Roboflow** (for dataset)
- **Streamlit** *(optional UI not deployed here)*

---

## 📊 Performance Metrics

| Metric           | Value   |
|------------------|---------|
| Precision        | 0.92    |
| Recall           | 0.82    |
| mAP@0.5          | 0.90    |
| mAP@0.5:0.95     | 0.64    |
| Inference Speed  | ~7 ms/image (CPU) |

---

##  How to Run

### Option 1: Colab Notebook

- Open [`training_and_inference.ipynb`](./empty_shelf_detector_YOLOv8.ipynb)
- Train or load a model
- Upload test image
- View predictions with bounding boxes

### Option 2: Local Inference (Python)

```bash
pip install ultralytics opencv-python pillow
```

```python
from ultralytics import YOLO
model = YOLO("best.pt")
results = model.predict(source="your_test_image.jpg", save=True, conf=0.5)
```

---

##  Directory Structure

```
empty-shelf-detector/
├── empty_shelf_detector_YOLOv8.ipynb   # End-to-end training & inference notebook
├── best.pt                             # Trained model weights
├── requirements.txt                    # Required dependencies
├── README.md                           # Project documentation
└── runs/
    └── sample_predictions/             # Example outputs
```

---



---

##  Future Improvements

- Train with multi-class shelf/product labels
- Deploy Streamlit web demo (UI ready)
- Convert model to ONNX or TensorRT for edge devices




