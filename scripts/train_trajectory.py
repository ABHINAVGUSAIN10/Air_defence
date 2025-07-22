import sys
import os
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from trajectory_dataset import TrajectoryDataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.trajectory_model import TrajectoryPredictor

train_path = r"C:\Users\Abhinav Gusain\Documents\Air_defence\data\trajectory\7days1\processed_data\train"
full_dataset = TrajectoryDataset(train_path)

# Split 80% train, 20% validation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

model = TrajectoryPredictor()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for epoch in range(25):
    # Training
    model.train()
    total_loss = 0
    for input_seq, target_seq in train_loader:
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)

        optimizer.zero_grad()
        output = model(input_seq, pred_len=10)
        loss = loss_fn(output, target_seq)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for input_seq, target_seq in val_loader:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            output = model(input_seq, pred_len=10)
            loss = loss_fn(output, target_seq)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # After training is complete
torch.save(model.state_dict(), "C:/Users/Abhinav Gusain/Documents/Air_defence/models/trajectory_lstm.pth")
print("✅ Trained model saved to models/trajectory_lstm.pth")

