import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import uuid
import random

# --- Import your custom ML modules ---
from modules.yolo_inference import detect_aircraft
from modules.radar_classifier import classify_modulation
from modules.trajectory_predictor import predict_trajectory

# --- Flask App Initialization ---
app = Flask(__name__)

# Configure the folder to store temporary user uploads
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Store processed data between requests. In a real app, you'd use a database or session.
plane_data_storage = {}

# --- Helper Function ---
def generate_initial_history(start_pos):
    """Creates a synthetic history for the trajectory model."""
    history = []
    # start_pos is expected to be [x, y, z, vx, vy]
    start_x, start_y, start_z, start_vx, start_vy = start_pos
    for i in range(10):
        # Create a linear path leading up to the starting point
        hist_x = start_x - (10 - i) * start_vx
        hist_y = start_y - (10 - i) * start_vy
        history.append([hist_x, hist_y, start_z, start_vx, start_vy])
    return history

# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the main upload page."""
    # Clear any old data when the user starts over
    plane_data_storage.clear()
    return render_template('index.html')

# In app.py, replace the entire upload_files function

# In app.py, replace the entire upload_files function

# In app.py, replace the entire upload_files function

# In app.py, replace the entire upload_files function

# In app.py, replace the entire upload_files function

@app.route('/upload', methods=['POST'])
def upload_files():
    """
    Handles uploads with the corrected URL path for images.
    """
    plane_data_storage.clear()
    image_files = request.files.getlist('image_files')
    npy_files = request.files.getlist('npy_files')

    if not image_files:
        return redirect(request.url)

    npy_paths = {}
    for npy_file in npy_files:
        if npy_file and npy_file.filename:
            filename = secure_filename(npy_file.filename)
            base_name, _ = os.path.splitext(filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            npy_file.save(file_path)
            npy_paths[base_name.lower()] = file_path

    for img_file in image_files:
        if not (img_file and img_file.filename):
            continue
        try:
            filename = secure_filename(img_file.filename)
            base_name, _ = os.path.splitext(filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img_file.save(image_path)
            
            npy_path = npy_paths.get(base_name.lower())
            img = Image.open(image_path)
            yolo_results = detect_aircraft(img)
            
            iq_data = np.load(npy_path) if npy_path else None
            modulation = classify_modulation(iq_data)
            
            plane_id = str(uuid.uuid4())
            crop_filename = f"{plane_id}.png"
            crop_path = os.path.join(app.config['UPLOAD_FOLDER'], crop_filename)
            
            class_name = "Unidentified"
            if len(yolo_results.boxes) > 0:
                best_box = max(yolo_results.boxes, key=lambda box: box.conf)
                class_name = yolo_results.names[int(best_box.cls)]
                coords = best_box.xyxy[0].cpu().numpy().astype(int)
                cropped_img = img.crop(coords)
                cropped_img.save(crop_path)
            else:
                img.save(crop_path)

            # --- CORRECTED LINE ---
            # Use forward slashes for the web URL path
            web_image_path = f'uploads/{crop_filename}'

            plane_data_storage[plane_id] = {
                'id': plane_id, 'name': class_name, 'modulation': modulation,
                'image_path': web_image_path
            }
        except Exception as e:
            print(f"ERROR processing file {img_file.filename}: {e}")
            continue
            
    return render_template('results.html', planes=list(plane_data_storage.values()))

# In app.py, replace the simulation function

@app.route('/simulation', methods=['POST'])
def simulation():
    """
    Sends the INITIAL STATE of each aircraft to the simulation page.
    """
    threat_ids = set(request.form.getlist('threats'))
    plane_ids = request.form.getlist('plane_ids')
    
    initial_sim_data = []

    for i, plane_id in enumerate(plane_ids):
        plane_details = plane_data_storage.get(plane_id)
        if plane_details:
            is_threat = plane_id in threat_ids
            
            # Define the starting state [x, y, z, vx, vy] for JavaScript
            initial_state = {
                'x': 20.0 + (i * 60),
                'y': 50.0 + random.uniform(-20, 20),
                'z': 10.0,
                'vx': 1.5 + random.uniform(-0.2, 0.2),
                'vy': 1.0 + random.uniform(-0.2, 0.2),
            }
            
            initial_sim_data.append({
                'id': plane_id,
                'name': plane_details['name'],
                'modulation': plane_details['modulation'],
                'is_threat': is_threat,
                'initial_state': initial_state # Send initial state, not full path
            })

    return render_template('simulation.html', initial_sim_data=initial_sim_data)

# This block allows the script to be run directly to start the server
if __name__ == '__main__':
    app.run(debug=True)