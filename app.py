import os
import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, request, jsonify, render_template
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = './static/uploads'
RESULT_FOLDER = './static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

MODEL_PATH = 'model/model.tflite'

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Error: model.tflite not found.")

class ObjectDetector:
    def __init__(self, model_path):
        self.model = tf.lite.Interpreter(model_path=model_path)
        self.model.allocate_tensors()

    def detect(self, image_np):
        input_details = self.model.get_input_details()
        output_details = self.model.get_output_details()

        input_shape = input_details[0]['shape']
        image_resized = cv2.resize(image_np, (input_shape[2], input_shape[1]))
        image_resized = np.expand_dims(image_resized, axis=0).astype(np.uint8)

        self.model.set_tensor(input_details[0]['index'], image_resized)
        self.model.invoke()

        output_data = [self.model.get_tensor(output['index']) for output in output_details]
        return output_data

def postprocess_result(output_data):
    boxes = output_data[0][0]  
    class_ids = output_data[1][0]  
    scores = output_data[2][0] 

    detections = []
    for i in range(len(scores)):
        if scores[i] > 0.5: 
            detections.append({
                'bounding_box': boxes[i],
                'class_id': class_ids[i],
                'score': scores[i]
            })
    return detections

def extract_labels(output_data, label_map):
    """
    Extract labels and confidence from detection results.
    
    Args:
        output_data (list): Output data from the model containing bounding boxes, class IDs, and scores.
        label_map (dict): Mapping of class IDs to class names.
        
    Returns:
        list: List of dictionaries containing labels, bounding boxes, and confidence scores.
    """
    boxes = output_data[0][0]  
    class_ids = output_data[1][0] 
    scores = output_data[2][0]  

    results = []
    for i, score in enumerate(scores):
        if score > 0.5:
            label = label_map.get(int(class_ids[i]), "Unknown")
            confidence = int(score * 100)
            box = boxes[i]

            result = {
                "label": label,
                "confidence": confidence,
                "bounding_box": {
                    "left": box[1],  
                    "top": box[0],   
                    "right": box[3],
                    "bottom": box[2] 
                }
            }
            results.append(result)

    return results

def visualize(image_np, detections):
    height, width, _ = image_np.shape
    for detection in detections:
        box = detection['bounding_box']
        ymin, xmin, ymax, xmax = box

        left = int(xmin * width)
        top = int(ymin * height)
        right = int(xmax * width)
        bottom = int(ymax * height)

        cv2.rectangle(image_np, (left, top), (right, bottom), (0, 255, 0), 2)
        label = f"Class {int(detection['class_id'])} ({detection['score']:.2f})"
        cv2.putText(image_np, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image_np

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    image = Image.open(filepath).convert('RGB')
    image_np = np.array(image)

    detector = ObjectDetector(model_path=MODEL_PATH)
    output_data = detector.detect(image_np)
    detections = postprocess_result(output_data)

    result_image = visualize(image_np, detections)

    result_path = os.path.join(app.config['RESULT_FOLDER'], file.filename)
    cv2.imwrite(result_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))

    return render_template('result.html', result_image=f'/static/results/{file.filename}', detections=detections)

@app.route('/static/<path:path>')
def serve_static(path):
    return app.send_static_file(path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
