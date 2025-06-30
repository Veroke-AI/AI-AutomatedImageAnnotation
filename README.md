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

- 🖼 **Open-Source GUI for Data Annotation**
  Our platform includes a user-friendly, open-source GUI annotation interface designed for seamless dataset creation and editing — directly in your browser.

- ✅ Draw / Remove Bounding Boxes
  Users can manually add or delete bounding boxes on any image to define or refine annotations.

- 🏷 **Add New Classes On The Fly**
  Easily create new object categories via dropdown or text input, even during annotation sessions.

- 📷 **Upload Your Own Images**
  Supports uploading images or loading them via API from external datasets.

- 🔁 **Edit, Relabel, or Reassign Annotations**
  Existing bounding boxes can be reclassified, resized, or deleted in one click.

- 💾 Annotation Autosave
  Changes are saved to disk or via API without refreshing the page.

- 📤 Export in Standard Formats
  Annotations can be exported in COCO, Pascal VOC, or YOLO format — ready for training.

- 🏷️ Multiple Annotation Methods:
  - Text Labels: Add comma-separated class labels
  - Click Labels: Add positive/negative click annotations
  - Brush Labels: Create freeform brush strokes for region selection

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

# Data Annotation Tool

A modern web application for annotating images with various labeling methods, built with Angular and Material Design.

## Usage

1. **Upload Images**
   - Click "Select Folder" or drag and drop a folder containing images
   - Supported formats: PNG, JPEG, GIF, etc.

2. **Select an Image**
   - Click on any thumbnail in the gallery to select it for annotation
   - The selected image will be displayed in the main canvas area

3. **Add Annotations**
   - **Text Labels**:
     - Select "Text" mode
     - Enter comma-separated labels in the input field
   
   - **Box label**:
     - Select "polygon" mode
     - Click and drag to draw box

4. **Apply Preprocessing**
   - Enable desired preprocessing options
   - Adjust thresholds and parameters as needed

5. **Submit Annotations**
   - Click "Submit Annotations" to send the data
   - The payload will include all annotations and preprocessing options

## Development

- **Components**: Located in `src/app/components/`
- **Styles**: Using Tailwind CSS with Material Design
- **State Management**: Component-based with Angular's built-in features

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
├── backend/                    # FastAPI backend
│   ├── detector                # Yolo script
│   ├── segmentation            # SAM scripts
│   ├── utils                   # utility scripts for cropping, formatting etc.
│   ├── zero_detector           # GroundingDino script
│   ├── main.py                 # FastAPI entry point
│   ├── requirements.txt        # Python dependencies
│   ├── Dockerfile              # Backend container setup
│   └── models/                 # AI model weights (used by backend)
│       ├── sam_vit_b_01ec64.pth
│       └── groundingdino_swint_ogc.pth
│
├── frontend/                   # Angular frontend
│   ├── src/
│   ├── angular.json
│   ├── package.json
│   └── Dockerfile              # Frontend container setup
│
├── docker-compose.yml          # Orchestrates frontend & backend

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
- python version used 3.10.6 cuda enabled
- https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
- https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
 

For Docker(optional):
- [Docker Desktop](https://www.docker.com/products/docker-desktop)
- [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) enabled (for Windows users) 
---

## 📥 Setup Instructions

```bash
1. Clone the Repository
git clone https://github.com/Veroke-IT/AI-AutomatedImageAnnotation
cd ai-image-annotator/backend

2. Download Model Weights
Place the following .pth files in the AI-AutomatedImageAnnotation/models/ directory:

🔗 SAM ViT-B Model

🔗 Grounding DINO Swin-T Model

3. Create a python environment
python -m venv venv
venv\Scripts\activate  

4. Install requirements
pip install -r requirements.txt

5. Run app
uvicorn main:app --reload --port 8000

🧠 Backend API: http://localhost:8000/docs

first time setup will install one time models such as yolo text encoders etc. so be patient, it will take some time.
```
```bash
🔹 Frontend (Angular)

1. Install requirements
cd ai-image-annotator/frontend
npm install --legacy-peer-deps
npm start

```
```bash
📤 API Docs
Visit Swagger UI at:
👉 http://localhost:8000/docs

```
```bash
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

