import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))
from trajectory_dataset import TrajectoryDataset

from trajectory_model import TrajectoryPredictor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# -------- Load saved model --------
model_path = r"C:\Users\Abhinav Gusain\Documents\Air_defence\models\trajectory_lstm.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TrajectoryPredictor()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# -------- Load test dataset --------
test_path = r"C:\Users\Abhinav Gusain\Documents\Air_defence\data\trajectory\7days1\processed_data\test"
test_dataset = TrajectoryDataset(test_path)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# -------- Evaluation metrics --------
mse_total, mae_total = 0, 0
all_preds, all_targets = [], []

with torch.no_grad():
    for input_seq, target_seq in test_loader:
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)

        output = model(input_seq, pred_len=10)

        pred_np = output.cpu().numpy().reshape(-1, 3)
        target_np = target_seq.cpu().numpy().reshape(-1, 3)

        all_preds.append(pred_np)
        all_targets.append(target_np)

# -------- Final metrics --------
all_preds = np.vstack(all_preds)
all_targets = np.vstack(all_targets)

mse = mean_squared_error(all_targets, all_preds)
mae = mean_absolute_error(all_targets, all_preds)
rmse = np.sqrt(mse)

print(f"📊 Evaluation on Test Set:")
print(f"✅ MAE  (Mean Absolute Error): {mae:.4f}")
print(f"✅ MSE  (Mean Squared Error): {mse:.4f}")
print(f"✅ RMSE (Root Mean Squared Error): {rmse:.4f}")
