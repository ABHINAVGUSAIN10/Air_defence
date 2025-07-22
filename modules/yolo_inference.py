from ultralytics import YOLO
from PIL import Image

# --- IMPORTANT: Update this path to your model's weights file ---
MODEL_PATH = r"C:\Users\Abhinav Gusain\Documents\Air_defence\results\detection_outputs\weights\best.pt"

# Load the model once when the application starts
model = YOLO(MODEL_PATH)

def detect_aircraft(image: Image.Image):
    """
    Runs YOLOv8 detection on the input PIL Image and returns the result object.
    
    Args:
        image (PIL.Image.Image): The input image.
        
    Returns:
        An ultralytics Results object containing detections.
    """
    # Run prediction with stricter thresholds to avoid duplicate detections
    results = model(image, imgsz=640, conf=0.25, iou=0.45)
    
    # The model returns a list; we return the first result object for the single image
    return results[0]