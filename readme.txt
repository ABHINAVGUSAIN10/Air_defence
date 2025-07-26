AI R Defence System
This project is a web-based simulation of an air defence system that uses computer vision and deep learning models to identify, track, and simulate engagement with potential aerial threats. The entire user interface is built using the Flask web framework.

Features
    Aircraft Detection & Classification: Upload aircraft images and use a YOLOv8 model to detect and classify the aircraft type (e.g., A10, B2).

    Radar Signal Analysis: Upload corresponding .npy I/Q data files to classify the aircraft's radar signal modulation using a custom RadarCNN model.

    Trajectory Prediction: An LSTM model pre-calculates a realistic flight path for each detected aircraft.

    Interactive Web UI: A clean, multi-page web interface built with Flask allows users to upload files, view analysis results, and mark threats.

    Live Radar Simulation: A dynamic canvas-based animation displays a live radar screen with a sweeping effect, a central military base, and all detected aircraft.

    Threat Engagement: Aircraft marked as "threats" are automatically engaged with a missile animation once they enter the radar's range.

    Live Status Panel: A side panel displays real-time information for each aircraft, including its name, status (Active/Destroyed), and dynamically updating coordinates.

Project Structure
The project is organized into distinct directories for data, models, scripts, and the web application itself.

AIR_DEFENCE/
├── app.py                      # Main Flask application entry point
├── requirements.txt
├── README.md
|
├── data/                       # All datasets for training and testing
│   ├── detection/              # Image data for the YOLOv8 model
│   ├── iq_data/                # Sample I/Q .npy files for inference
│   ├── radar_signal/           # Class definitions for the radar model
│   └── trajectory/             # Raw and processed trajectory data
|
├── models/                     # Python class definitions and trained model weights
│   ├── best_radar_cnn.pth
│   ├── radar_cnn.py
│   ├── trajectory_lstm.pth
│   └── trajectory_model.py
|
├── modules/                    # Helper modules that load models for inference in the app
│   ├── radar_classifier.py
│   ├── trajectory_predictor.py
│   └── yolo_inference.py
|
├── scripts/                    # Standalone scripts for training and evaluation
│   ├── train_detection.py
│   ├── train_radar_classifier.py
│   ├── train_trajectory.py
│   ├── evaluate_radar_classifier.py
│   └── ... (and other evaluation/utility scripts)
|
├── static/                     # Web assets (CSS, JavaScript, and user uploads)
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── simulation.js
│   └── uploads/
|
└── templates/                  # HTML templates for the Flask front-end
    ├── index.html
    ├── results.html
    └── simulation.html


Setup and Installation
Follow these steps to set up the project environment on your local machine.

Prerequisites
This project requires Python 3.8+ and the Pip package manager.

Steps:

1:    Clone the repository:
      Open your terminal, navigate to where you want to store the project, and run:

      git clone <your-repository-url>
      cd AIR_DEFENCE

2:    Create and activate a virtual environment:
      It is highly recommended to use a virtual environment to manage project dependencies.

      On Windows:

      python -m venv venv
      .\venv\Scripts\activate

      On macOS & Linux:

      python3 -m venv venv
      source venv/bin/activate    

3:    Install required libraries:
      With your virtual environment activated, install all dependencies from the requirements.txt file:

      pip install -r requirements.txt

4:    Add Model Weights:
      
      Place your trained model files (e.g., best.pt, best_radar_cnn.pth, trajectory_lstm.pth) inside the models/ directory.
      Important: You must verify that the hardcoded file paths inside each script in the modules/ folder (yolo_inference.py, radar_classifier.py, etc.) point to the correct location of 
      your model files.

How to Run
Navigate to the project root (AIR_DEFENCE) and ensure your virtual environment is activated.

Run the Flask application:

            python app.py

Open the UI in Your Browser:
        After running the command, the terminal will show that the server is active. Open your web browser and go to the URL provided, which is usually:
        http://1227.0.0.1:5000

How to Use
1: Upload Files: On the main page, upload one or more aircraft images (.jpg, .png). For each image, upload the corresponding .npy file with the exact same base name (e.g., plane1.jpg and plane1.npy).

2: Analyze Results: You will be taken to the "Analysis Results" page. Here you can see the aircraft type and radar modulation for each detected plane.

3: Mark Threats: Use the checkboxes to mark any aircraft you want to designate as a threat.

4: Run Simulation: Click the "Run Simulation" button to launch the live radar display and watch the engagement.

