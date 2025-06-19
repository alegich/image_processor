from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import os
import glob
from datetime import datetime
import io
import zipfile
import face_recognition

app = Flask(__name__)

OUTPUT_DIR = "processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in request'}), 400

    files = request.files.getlist('image')
    if not files:
        return jsonify({'error': 'No images uploaded'}), 400
    
    actions = request.args.get('actions', 'gray,equalize,denoise').split(',')

    processed_files = []

    for file in files:
        image_data = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        if img is None:
            continue

        # Convert to grayscale and enhance
        if 'gray' in actions:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if 'equalize' in actions:
            img = cv2.equalizeHist(img)
        if 'denoise' in actions:
            img = cv2.fastNlMeansDenoising(img, h=10)
        if 'clahe' in actions:
            img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
        if 'blur' in actions:
            img = cv2.GaussianBlur(img, (3, 3), 0)
        if 'faces' in actions:
            img = detect_faces(img)

        # Save processed image to memory (not disk)
        filename = f"snap_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        _, buffer = cv2.imencode('.jpg', img)
        processed_files.append((filename, buffer.tobytes()))

    if not processed_files:
        return jsonify({'error': 'No valid images processed'}), 400

    # Create in-memory zip file
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zipf:
        for fname, content in processed_files:
            zipf.writestr(fname, content)

    zip_buffer.seek(0)

    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name='processed_images.zip'
    )

def is_grayscale(image):
    return len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1)

def detect_faces(image):
    # convert to RGB for face detection
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Detect face locations: [(top, right, bottom, left), ...]
    face_locations = face_recognition.face_locations(image_rgb)
    print(f"Found {len(face_locations)} face(s)")
    if is_grayscale(image):
        # If the image is grayscale, convert it to BGR for drawing
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    draw_dashes(image, face_locations)
    return image

def draw_dashed_box(image, top_left, bottom_right, color=(0, 255, 0), thickness=2, dash_length=10):
    """Draw a dashed rectangle by splitting lines into dashes."""
    x1, y1 = top_left
    x2, y2 = bottom_right

    def draw_dashed_line(p1, p2):
        dist = ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2) ** 0.5
        dashes = int(dist / dash_length)
        for i in range(0, dashes, 2):
            start = (
                int(p1[0] + (p2[0] - p1[0]) * i / dashes),
                int(p1[1] + (p2[1] - p1[1]) * i / dashes),
            )
            end = (
                int(p1[0] + (p2[0] - p1[0]) * (i + 1) / dashes),
                int(p1[1] + (p2[1] - p1[1]) * (i + 1) / dashes),
            )
            cv2.line(image, start, end, color, thickness)

    # Top edge
    draw_dashed_line((x1, y1), (x2, y1))
    # Right edge
    draw_dashed_line((x2, y1), (x2, y2))
    # Bottom edge
    draw_dashed_line((x2, y2), (x1, y2))
    # Left edge
    draw_dashed_line((x1, y2), (x1, y1))

def draw_dashes(image, face_locations):
    # Draw dashed boxes
    for (top, right, bottom, left) in face_locations:
        draw_dashed_box(image, (left, top), (right, bottom))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)
