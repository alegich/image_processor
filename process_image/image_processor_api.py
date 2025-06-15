from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import os
import glob
from datetime import datetime
import io
import zipfile

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

    processed_files = []

    for file in files:
        image_data = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        if img is None:
            continue

        # Convert to grayscale and enhance
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.equalizeHist(gray)

        # Save processed image to memory (not disk)
        filename = f"snap_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        _, buffer = cv2.imencode('.jpg', enhanced)
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)
