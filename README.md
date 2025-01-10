# Object Detection Web Application

This project is a containerized web application for object detection and visualization. It leverages a TensorFlow Lite (TFLite) model for efficient object detection and uses Flask as the backend framework. Users can upload images via a user-friendly web interface to receive real-time detection results, including bounding boxes, class labels, and confidence scores.

---

## Key Features

### **Object Detection**
- Integrated a lightweight TensorFlow Lite model for efficient object detection.
- Processes user-uploaded images to identify objects, bounding boxes, class IDs, and confidence scores.

### **Visualization**
- Detection results are visually enhanced by overlaying bounding boxes and labels on uploaded images.
- Uses OpenCV and PIL for image processing and visualization.

### **Web Interface**
- Designed a clean and intuitive HTML interface for image upload and output display.

### **Containerization**
- Application is containerized using Docker, ensuring portability and environment consistency across platforms.

---

## Tech Stack

### **Backend**
- Flask (Python)

### **Machine Learning**
- TensorFlow Lite

### **Frontend**
- HTML, CSS

### **Visualization**
- OpenCV
- PIL (Python Imaging Library)

### **Containerization**
- Docker

### **File Management**
- Image upload and result storage with folder-based segregation for organized file handling.

---

## Key Accomplishments
- Developed a complete end-to-end object detection pipeline: from model inference to result visualization.
- Enabled real-time detection and visualization for user-uploaded images with minimal latency.
- Successfully containerized the application using Docker for seamless deployment.

---

## Impact
- Delivered an efficient and scalable solution for object detection with real-time feedback.
- Demonstrated expertise in:
  - Machine learning model integration.
  - Backend development using Flask.
  - Deployment strategies with Docker.

---

## How to Run the Application

### **Pre-requisites**
- Docker installed on your system.
- TensorFlow Lite model file (`.tflite`) ready for deployment.

### **Steps**
1. Clone this repository:
   ```bash
   git clone https://github.com/vedantsalunke29/Object-Detection
   cd Object-Detection
2. Build the Docker image:
   ```bash
   docker build -t object-detection-app .
   
3. Run the Docker container:
   ```bash
   docker run -p 5001:5001 object-detection-app
   
4. Access the web application:
- Open a browser and navigate to http://localhost:5001.

5. Upload an image and view detection results.




   
