# 🚦 Traffic Sign Recognition Project

![Traffic Sign Detection](https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Traffic_sign_detection.png/800px-Traffic_sign_detection.png)

## 📌 Introduction
This project focuses on image processing and computer vision techniques to detect and recognize traffic signs. The methodology includes image segmentation, edge detection, shape recognition, and classification using OpenCV.

## 📂 Table of Contents
- [Techniques and Algorithms](#techniques-and-algorithms)
  - [Image Segmentation](#image-segmentation)
  - [Edge Detection](#edge-detection)
  - [Shape Recognition](#shape-recognition)
  - [Traffic Sign Classification](#traffic-sign-classification)
- [Proposed Methodology](#proposed-methodology)
  - [Creating Lookup Dictionaries](#creating-lookup-dictionaries)
  - [Four-Point Perspective Transformation](#four-point-perspective-transformation)
  - [Traffic Sign Recognition](#traffic-sign-recognition)
  - [Traffic Sign Detection](#traffic-sign-detection)
  - [Display and Read Image Functions](#display-and-read-image-functions)
- [Installation & Usage](#installation--usage)
- [Summary](#summary)
- [Acknowledgments](#acknowledgments)

## 🛠 Techniques and Algorithms
### 📌 Image Segmentation
✅ Converts images to HSV color space for easier segmentation.  
✅ Defines color thresholds to isolate traffic sign colors (e.g., red and blue).  
✅ Uses masks, morphological operations (dilation, erosion) to refine results.  

### 📌 Edge Detection
✅ Applies **Canny Edge Detection** to identify traffic sign contours.  
✅ Helps define shapes for recognition and classification.  

### 📌 Shape Recognition
✅ Detects geometric shapes (circles, triangles, rectangles) using:
- **Hough Transform** (detecting lines & circles)
- **Template Matching** (comparing predefined shapes)
- **Contour Analysis** (boundary detection)

### 📌 Traffic Sign Classification
✅ Uses OpenCV for:
- **Feature Extraction:** SIFT, SURF, ORB.
- **Template Matching:** Compares segmented traffic signs against a database.

## 📌 Proposed Methodology
### 🔹 Creating Lookup Dictionaries
✅ Defines 'Red' and 'Red_lower' dictionaries to map binary patterns to traffic signs for quick recognition.

### 🔹 Four-Point Perspective Transformation
🔸 Steps:
1. Identify the top-left, top-right, bottom-right, and bottom-left corners.
2. Compute the width and height based on distances between points.
3. Define destination points for transformed perspective.
4. Use OpenCV's `getPerspectiveTransform` and `warpPerspective`.

### 🔹 Traffic Sign Recognition
🔹 Steps:
1. **Invert image colors**
2. **Divide the image into blocks** (left, center, right, top)
3. **Calculate white pixel fractions** in each block
4. **Generate a binary pattern** for recognition
5. **Match pattern** to predefined dictionaries

### 🔹 Traffic Sign Detection
🔹 Steps:
1. Read and resize image (250x250 pixels).
2. Convert to HSV color space.
3. Segment based on red/blue color ranges.
4. Apply morphological transformations (opening, closing).
5. Detect and filter contours to find the largest contour.
6. Transform perspective and recognize the sign.
7. Annotate and save the result.

### 🔹 Display and Read Image Functions
✅ Reads and displays images in a grid layout using Matplotlib.  
✅ Processes images from a folder, detects traffic signs, and saves results.  

## 📥 Installation & Usage
### 1️⃣ Clone the repository
```bash
git clone https://github.com/khoi272/Image_processing_project.git
```
### 2️⃣ Navigate to the project folder
```bash
cd Image_processing_project
```
### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
### 4️⃣ Run the main script
```bash
python main.py
```

## ✅ Summary
🚀 This project integrates multiple computer vision techniques to detect and recognize traffic signs:
- **Lookup Dictionaries:** Predefined patterns for recognition.
- **Perspective Transformation:** Adjusting viewpoint of detected signs.
- **Sign Recognition:** Identifying segmented traffic signs.
- **Sign Detection:** Extracting and processing traffic signs from images.
- **Image Display and Processing:** Managing input/output images.
- **Main Execution:** Orchestrating the recognition pipeline.

This methodology ensures accurate and efficient traffic sign detection and classification for automated systems.

## 🙏 Acknowledgments
Special thanks to **Dr. Trinh Hung Cuong** for his guidance and support throughout this project.

---
📌 **Author:** Ngo Minh Khoi  
📌 **University:** Ton Duc Thang University  
📌 **Course:** Digital Image Processing  

Feel free to contribute or suggest improvements! 🚀
