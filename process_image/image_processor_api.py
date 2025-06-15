from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import os
import glob
from datetime import datetime

app = Flask(__name__)

OUTPUT_DIR = "processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in request'}), 400

    file = request.files['image']
    image_data = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Improve contrast with histogram equalization
    enhanced = cv2.equalizeHist(gray)

    filename = f"snap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    filepath = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(filepath, enhanced)

    cleanup_snapshots()

    return send_file(filepath, mimetype='image/jpeg')

def cleanup_snapshots(keep=5):
    pattern = os.path.join(OUTPUT_DIR, "snap_*.jpg")

    # Get a list of files matching the pattern, sorted by modification time (descending)
    files = sorted(
        glob.glob(pattern),
        key=os.path.getmtime,
        reverse=True
    )

    # Keep only the newest 'keep' files
    files_to_delete = files[keep:]

    for file_path in files_to_delete:
        os.remove(file_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)

