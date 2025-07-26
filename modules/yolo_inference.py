from ultralytics import YOLO
from PIL import Image


MODEL_PATH = r"C:\Users\Abhinav Gusain\Documents\Air_defence\results\detection_outputs\weights\best.pt"

# Load the model once when the application starts
model = YOLO(MODEL_PATH)

def detect_aircraft(image: Image.Image):
    
    
    results = model(image, imgsz=640, conf=0.25, iou=0.45)
    
    
    return results[0]
