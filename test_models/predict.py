import os
from ultralytics import YOLO
from PIL import Image

# Load the trained model
model = YOLO(r"C:\Users\Abhinav Gusain\Documents\Air_defence\results\detection_outputs\weights\best.pt")  

# Path to folder containing test images
test_folder = r"C:\Users\Abhinav Gusain\Documents\Air_defence\test_images"

# Output folder for annotated predictions
output_folder = r"C:\Users\Abhinav Gusain\Documents\Air_defence\test_models\predictions"
os.makedirs(output_folder, exist_ok=True)


for file in os.listdir(test_folder):
    if file.endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(test_folder, file)

        
        results = model(image_path)

        
        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            print(f" Aircraft detected in: {file}")
        else:
            print(f" No aircraft detected in: {file}")

        
        save_path = os.path.join(output_folder, file)
        result.save(filename=save_path)
