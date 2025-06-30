# AI-AutomatedImageAnnotation
create a virtual env python version == 3.10.6
install requirements.txt
create a models folder inside the project directory
download sam_vit_b_01ec64.pth inside the models folder(https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
download groundingdino_swint_ogc.pth inside the models folder(https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth)

# 🧠 Veroke AI-Powered Image Annotation Platform

**One Unified AI Platform for Image Annotation & Smart Visual Search**  
At Veroke, we’re reshaping how teams and individuals interact with visual data — combining cutting-edge AI annotation tools with intelligent image organization and search in one seamless, web-based platform.

Whether you’re labeling training data for machine learning or organizing a lifetime of digital images, our platform delivers automation, insight, and control — all in one place.

---

## 🚀 Smart AI-Powered Image Annotation — Built for Accuracy & Scale

Designed for AI/ML teams, researchers, and product builders, our annotation suite streamlines every step of dataset creation:

### ✨ Key Capabilities:
- 🔍 **Zero-Shot Object Detection**  
  Use natural language prompts like “solar panel”, “military vehicle”, or “broken window”.  
  No prior training needed. Powered by **YOLOWorld** and **GroundingDINO**. (If label is in the yolo class yolo model otherwise GroundingDino)

- ✂️ **Auto & Manual Segmentation**  
  Detected objects are segmented instantly using **SAM** or **EfficientSAM**, with optional refinement using polygons, points, or masks.

- 📁 **One-Click Dataset Export**  
  Export annotations in COCO, Pascal VOC, or YOLO formats with:
  - Cropped objects
  - Organized folder structure
  - Train/val splits
  - Class label mapping

- 🔁 **Iterative Labeling & Human-in-the-Loop Reprocessing**  
  Upload revised annotations and regenerate outputs — ideal for QA cycles or dataset evolution.

- 🧠 **CLIP-Powered Mislabel Detection**  
  We use **CLIP + UMAP + k-NN** to visualize embeddings, identify outliers, and fix class label inconsistencies *before* model training.
  (This feature is implemented in the backend but not configured at the frontend)

- 📊 **Interactive Review Dashboards**  
  Explore data using **Plotly** visualizations — clusters, anomalies, and dataset quality at a glance.(also only implemented in backend)

---

## 📸 Smart Photo Gallery — Visual Search & Image Management

Beyond annotation, the platform includes a visual gallery for organizing and searching image collections using the same AI backbone.

### 🌟 Features:
- 📝 **Text-Based Search with CLIP**  
  Query images with phrases like *“beach sunset with silhouette”* or *“red car on snowy road”* — and retrieve semantically matched results.

- 🖼️ **Reverse Image Search**  
  Upload an image and find visually similar items, regardless of filenames or folders.

- 🧹 **Duplicate & Near-Duplicate Detection**  
  Automatically clean galleries by detecting visual redundancy.

- 🧩 **Image Clustering**  
  Organize massive collections with unsupervised image clustering.

---
## 📦 Project Structure
```bash
.
├── AI-AutomatedImageAnnotation/ # FastAPI backend
│ ├── main.py # FastAPI app entry point
│ ├── requirements.txt # Python dependencies
│ ├── Dockerfile # Backend Dockerfile
│ └── models/ # Stores .pth model files
│ ├── sam_vit_b_01ec64.pth
│ └── groundingdino_swint_ogc.pth
│
├── gui/ # Angular frontend
│ ├── src/
│ ├── angular.json
│ ├── package.json
│ └── Dockerfile # Frontend Dockerfile
│
├── docker-compose.yml # Service orchestration

```
---

## 🚀 Features

- ✅ **Click-based segmentation** using Meta's Segment Anything Model (SAM)
- ✅ **Object detection without prompts** via Grounding DINO or YOLO models
- ✅ Docker file included
- ✅ Clean separation of backend and frontend services
- ✅ REST API for programmatic annotation workflows

---

## ⚙️ Requirements
Download following models in models folder:
- https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
- https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
 

For Docker(optional):
- [Docker Desktop](https://www.docker.com/products/docker-desktop)
- [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) enabled (for Windows users) 
---

## 📥 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Veroke-IT/AI-AutomatedImageAnnotation
cd ai-image-annotator
2. Download Model Weights
Place the following .pth files in the AI-AutomatedImageAnnotation/models/ directory:

🔗 SAM ViT-B Model

🔗 Grounding DINO Swin-T Model

3. Build and Run with Docker
docker compose up --build
🚀 Frontend: http://localhost:4200

🧠 Backend API: http://localhost:8000/docs
```

## 🛠️ Development (Without Docker)
```bash
🔹 Backend (FastAPI)

cd AI-AutomatedImageAnnotation
python -m venv venv
venv\Scripts\activate      # On Windows
# source venv/bin/activate # On Linux/Mac
pip install -r requirements.txt
uvicorn main:app --reload --port 8000


🔹 Frontend (Angular)

cd gui
npm install --legacy-peer-deps
npm start

```
```bash
📤 API Docs
Visit Swagger UI at:
👉 http://localhost:8000/docs

🧠 Models Used
Segment Anything (SAM)

Grounding DINO

EfficientSAM (if integrated)

Yolo and YoloE models

Clip models
```
---
## 🔐 License
This project is licensed under your company’s terms or an open-source license of your choice.

## 🧩 Troubleshooting
❌ Docker crashes or fails to build?
Run wsl --shutdown, restart Docker Desktop, and retry docker compose up.

## ❌ Model files not found?
Make sure both .pth files are in AI-AutomatedImageAnnotation/models/.

## 👥 Maintainers
Built by Veroke — Transforming Ideas into Digital Reality

## 🤝 Let’s Collaborate
We’re working with AI teams, research labs, and startups to optimize their visual data workflows.

Want a demo, integration support, or early access?
## 📬 Reach out at https://veroke.com

