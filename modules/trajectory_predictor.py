import torch
import numpy as np
from models.trajectory_model import TrajectoryPredictor 


MODEL_PATH = r'C:\Users\Abhinav Gusain\Documents\Air_defence\models\trajectory_lstm.pth'


model = TrajectoryPredictor()

model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True))
model.eval()

def predict_trajectory(sequence: list):
   
   
    seq_array = np.array(sequence, dtype=np.float32)
    if seq_array.ndim != 2 or seq_array.shape[1] != 5:
        raise ValueError("Input sequence must be a list of lists, with inner lists of length 5.")
    
    input_tensor = torch.tensor(seq_array).unsqueeze(0)

    with torch.no_grad():
        
        output = model(input_tensor)
        
        prediction = output[0].numpy().tolist()

    return prediction
