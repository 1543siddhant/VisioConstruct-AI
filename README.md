# VisioConstruct-AI: Door & Window Detection in Architectural Blueprints

---

## 🚀 Project Overview

**VisioConstruct-AI** detects **doors** and **windows** in blueprint-style images and multi-page PDFs.

* Manually labeled with LabelImg (YOLO format)
* Trained a lightweight **YOLOv8-n** model from scratch
* Exported to **ONNX** for fast CPU inference
* Flask app with both Web UI and a `POST /detect` API
* Supports image & PDF uploads, batch processing
* Generates annotated PDF reports and emails them to users
* Deployed on **Render**:
  [https://visio-construct-ai.onrender.com](https://visio-construct-ai.onrender.com)

---

## 📂 Repository Structure

```
VisioConstruct-AI/
├── images/                   # Original blueprint images
├── labels/                   # YOLO-format .txt label files
├── classes.txt               # ["door", "window"]
├── best_onnx.onnx            # ONNX-exported YOLOv8-n model
├── app.py                    # Flask web & API + PDF/email logic
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── static/
│   ├── img/logo.png          # Your project logo
│   └── img/preview.png       # Annotated preview output
├── templates/
│   └── index.html            # Web UI template
├── uploads/                  # Temp file uploads
├── Proof_of_Work/            # Screenshots & logs
│   ├── labeling.png          # LabelImg in-progress
│   ├── training_plot.png     # Loss & metric curves
│   └── api_response.png      # Sample API output
└── .env                      # (gitignored) SMTP & config vars
```

---

## 🏷️ Classes

```
door
window
```

---

## ⚙️ Setup & Run Locally

1. **Clone repo**

   ```bash
   git clone https://github.com/1543siddhant/VisioConstruct-AI.git
   cd VisioConstruct-AI
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure**
   Create a `.env` in project root:

   ```
   EMAIL_ADDRESS=youremail@example.com
   EMAIL_PASSWORD=your_smtp_password
   SMTP_SERVER=smtp.gmail.com
   SMTP_PORT=465
   ```

4. **Run**

   ```bash
   python app.py
   ```

   * Web UI: [http://localhost:8000](http://localhost:8000)
   * API:      [http://localhost:8000/detect](http://localhost:8000/detect)

---

## 🖼️ Web UI & API Usage

### Web UI

* Browse to `/`
* Upload an image or multi-page PDF
* Enter your email & optional comment
* Click **Detect & Email**
* Preview appears, and a PDF report is emailed

### API: `POST /detect`

* **Input**: form-data `file=@path/to/image_or_pdf`
* **Output** (JSON):

  ```json
  {
    "counts": { "door": 16, "window": 4 },
    "detections": [
      { "label": "door",   "confidence": 0.91, "bbox":[x,y,w,h] },
      { "label": "window", "confidence": 0.84, "bbox":[x,y,w,h] }
    ]
  }
  ```

#### Example `curl`

```bash
curl -X POST https://visio-construct-ai.onrender.com/detect \
  -F "file=@16.png" \
  -F "email=you@example.com" \
  -F "comment=Here’s my note"
```

---

## 🚀 Performance & Throughput

On a single CPU instance (Render free tier):

```
0: 640x640 16 doors,  4 windows → 264.2 ms total
0: 640x640 16 doors,  4 windows → 267.2 ms total
0: 640x640  4 doors,  2 windows → 277.0 ms total
```

* **Preprocess**: \~7 ms
* **ONNX Inference**: \~260 ms
* **Postprocess**: \~2–3 ms

---

## 📈 Training Metrics

![Training & Metrics](Proof_of_Work/training_plot.png)

* **Box/Cls/DFL losses** converge by epoch 60
* **Precision/Recall** both exceed 0.8
* **mAP\@0.5** ≈ 0.85 (door: 0.88, window: 0.81)

---

## 📸 Proof of Work

1. **Labeling Screenshot**
   ![Labeling](Proof_of_Work/labeling.png)

2. **Training Curves**
   ![Training Plot](Proof_of_Work/training_plot.png)

3. **API Response**
   ![API Response](Proof_of_Work/api_response.png)

---

## 🔗 Final Submission

* **GitHub**: [https://github.com/1543siddhant/VisioConstruct-AI](https://github.com/1543siddhant/VisioConstruct-AI)
* **Deployed Demo**: [https://visioconstruct-ai.onrender.com](https://visioconstruct-ai.onrender.com)
* **Loom Video**: [https://loom.com/share/your-video-link](https://loom.com/share/your-video-link)

---

## ✅ Assignment Checklist

| Item                                                      | Status |
| --------------------------------------------------------- | :----: |
| Manual YOLO-format labels (LabelImg)                      |    ✅   |
| `classes.txt`                                             |    ✅   |
| Train YOLOv8-n on only hand-drawn boxes (no JSON imports) |    ✅   |
| Export to ONNX + use ONNXRuntime                          |    ✅   |
| Build `POST /detect` endpoint                             |    ✅   |
| PDF & Email report feature                                |    ✅   |
| Web UI + API                                              |    ✅   |
| Deployed on Render                                        |    ✅   |
| README + `curl` example                                   |    ✅   |
| Screenshots: labeling, training, API                      |    ✅   |
| Loom walkthrough                                          |    ✅   |

---

Thank you—looking forward to your feedback!
