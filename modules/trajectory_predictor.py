import torch
import numpy as np
from models.trajectory_model import TrajectoryPredictor # Assumes models folder is accessible

# --- IMPORTANT: Update this path to your model's weights file ---
MODEL_PATH = r'C:\Users\Abhinav Gusain\Documents\Air_defence\models\trajectory_lstm.pth'

# Load the model once when the application starts
model = TrajectoryPredictor()
# Add weights_only=True to fix the warning
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True))
model.eval()

def predict_trajectory(sequence: list):
    """
    Predicts a sequence of future coordinates from a history of states.
    
    Args:
        sequence (list): A list of states, where each state is [x, y, z, vx, vy].
        
    Returns:
        list: A list of predicted future states, each as [x, y, z].
    """
    # Convert input to the correct tensor shape [1, seq_len, 5]
    seq_array = np.array(sequence, dtype=np.float32)
    if seq_array.ndim != 2 or seq_array.shape[1] != 5:
        raise ValueError("Input sequence must be a list of lists, with inner lists of length 5.")
    
    input_tensor = torch.tensor(seq_array).unsqueeze(0)

    with torch.no_grad():
        # Model output shape is [1, pred_len, 3] for x, y, z
        output = model(input_tensor)
        # Remove the batch dimension and convert to a standard Python list
        prediction = output[0].numpy().tolist()

    return prediction