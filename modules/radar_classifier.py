import torch
import numpy as np
from models.radar_cnn import RadarCNN # Assumes models folder is accessible

# --- IMPORTANT: Update these paths to your model and class files ---
MODEL_PATH = r'C:\Users\Abhinav Gusain\Documents\Air_defence\models\best_radar_cnn.pth'
CLASSES_PATH = r'C:\Users\Abhinav Gusain\Documents\Air_defence\data\radar_signal\classes-fixed.txt'

# Load the model and class labels once when the application starts
model = RadarCNN()
# Add weights_only=True to fix the warning
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True))
model.eval()

with open(CLASSES_PATH) as f:
    classes = [line.strip() for line in f]

def classify_modulation(iq_array: np.ndarray):
    """
    Takes a raw I/Q numpy array and returns the predicted modulation class.
    
    Args:
        iq_array (np.ndarray): The I/Q data of shape (2, 1024).
        
    Returns:
        str: The name of the predicted modulation type or an error message.
    """
    if not isinstance(iq_array, np.ndarray):
        return "I/Q data not provided"

    if iq_array.shape != (2, 1024):
        return f"Invalid I/Q shape: {iq_array.shape}"

    # Prepare tensor for the model
    iq_tensor = torch.tensor(iq_array, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        outputs = model(iq_tensor)
        predicted_class_index = torch.argmax(outputs, dim=1).item()

    return classes[predicted_class_index]