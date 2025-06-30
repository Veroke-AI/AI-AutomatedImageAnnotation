# Data Annotation Tool

A modern web application for annotating images with various labeling methods, built with Angular and Material Design.

## Features

- 📁 Folder Upload: Upload entire folders of images for annotation
- 🖼️ Image Viewer: View and select images from the uploaded collection
- 🏷️ Multiple Annotation Methods:
  - Text Labels: Add comma-separated class labels
  - Click Labels: Add positive/negative click annotations
  - Brush Labels: Create freeform brush strokes for region selection
- ⚙️ Image Preprocessing Options:
  - Metadata extraction
  - Grayscale conversion with threshold
  - Image binarization with threshold and inversion
  - Image resizing with aspect ratio preservation
- 🎯 User-Friendly Interface:
  - Keyboard shortcuts for common actions
  - Tooltips and help information
  - Responsive design for all screen sizes

## Prerequisites

- Node.js (v14 or later)
- npm (v6 or later)
- Angular CLI (`npm install -g @angular/cli`)

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd data-annotation-tool
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   ng serve
   ```

4. Open your browser and navigate to `http://localhost:4200`

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
   
   - **Click Labels**:
     - Select "Click" mode
     - Left-click to add positive annotations
     - Ctrl + Left-click to add negative annotations
   
   - **Brush Labels**:
     - Select "Brush" mode
     - Click and drag to draw regions
     - Use [ and ] keys to adjust brush size

4. **Apply Preprocessing**
   - Enable desired preprocessing options
   - Adjust thresholds and parameters as needed
   - Changes will be included in the API payload

5. **Submit Annotations**
   - Click "Submit Annotations" to send the data
   - The payload will include all annotations and preprocessing options

## Keyboard Shortcuts

- `[` / `]`: Adjust brush size
- `Ctrl + Click`: Add negative click annotation
- `Z`: Undo last annotation

## Development

- **Components**: Located in `src/app/components/`
- **Styles**: Using Tailwind CSS with Material Design
- **State Management**: Component-based with Angular's built-in features

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 