import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import h5py
import os
import argparse
from sklearn.model_selection import train_test_split

# Adjust import paths based on project structure
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.radar_dataset import RadarDataset
from models.radar_cnn import RadarCNN

def train(args):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    with h5py.File(args.data_path, 'r') as f:
        num_samples = len(f['X'])
    
    indices = np.arange(num_samples)
    # Split indices into training and validation sets (80/20 split)
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
    
    train_dataset = RadarDataset(args.data_path, train_indices)
    val_dataset = RadarDataset(args.data_path, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples.")

    # --- Model, Loss, Optimizer ---
    model = RadarCNN(num_classes=24).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    best_val_loss = float('inf')
    
    # --- Training Loop ---
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for i, (signals, labels) in enumerate(train_loader):
            signals, labels = signals.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        train_loss = running_loss / len(train_loader)
        
        # --- Validation Loop ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for signals, labels in val_loader:
                signals, labels = signals.to(device), labels.to(device)
                outputs = model(signals)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%\n")
        
        # Save the model with the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save(model.state_dict(), args.save_path)
            print(f"Model saved to {args.save_path}\n")

    train_dataset.close()
    val_dataset.close()
    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Radar Signal Classifier")
    parser.add_argument('--data_path', type=str, default='data/radar_signal/GOLD_XYZ_OSC.0001_1024.hdf5', help='Path to the HDF5 data file')
    parser.add_argument('--save_path', type=str, default='models/best_radar_cnn.pth', help='Path to save the best model weights')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    
    # In a real shell, you would run this script from the root 'air_defense_ai' directory
    # For IDEs, you might need to adjust default paths or run configurations
    args = parser.parse_args()
    
    # Make paths relative to the project root
    if not os.path.isabs(args.data_path):
        args.data_path = os.path.join(os.path.dirname(__file__), '..', args.data_path)
    if not os.path.isabs(args.save_path):
        args.save_path = os.path.join(os.path.dirname(__file__), '..', args.save_path)

    train(args)