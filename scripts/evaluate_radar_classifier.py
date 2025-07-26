import torch
from torch.utils.data import DataLoader
import numpy as np
import h5py
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.radar_dataset import RadarDataset
from models.radar_cnn import RadarCNN


CLASSES = [
    'OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
    '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
    '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
    'FM', 'GMSK', 'OQPSK'
]

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    with h5py.File(args.data_path, 'r') as f:
        num_samples = len(f['X'])
    indices = np.arange(num_samples)
    _, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
    
    test_dataset = RadarDataset(args.data_path, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Evaluating on {len(test_dataset)} samples.")

    # --- Load Model ---
    model = RadarCNN(num_classes=len(CLASSES)).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    all_labels = []
    all_predictions = []
   
    snr_results = defaultdict(lambda: {'correct': 0, 'total': 0})

    with torch.no_grad():
        for signals, labels, snrs in test_loader:
            signals = signals.to(device)
            outputs = model(signals)
            _, predicted = torch.max(outputs.data, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            
            
            for i in range(len(snrs)):
                snr_val = int(snrs[i].item())
                snr_results[snr_val]['total'] += 1
                if predicted[i].cpu() == labels[i]:
                    snr_results[snr_val]['correct'] += 1

    
    print("\n" + "="*50)
    print("Overall Classification Report")
    print("="*50)
    print(classification_report(all_labels, all_predictions, target_names=CLASSES, digits=3))
    
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Overall Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = "evaluation_confusion_matrix.png"
    plt.savefig(cm_path)
    print(f"\nOverall confusion matrix saved to {cm_path}")
    plt.close() 

    
    print("\n" + "="*50)
    print("Accuracy per Signal-to-Noise Ratio (SNR)")
    print("="*50)
    
    snr_values = sorted(snr_results.keys())
    acc_by_snr = []
    for snr in snr_values:
        accuracy = (snr_results[snr]['correct'] / snr_results[snr]['total']) * 100
        acc_by_snr.append(accuracy)
        print(f"SNR: {snr:3d} dB | Accuracy: {accuracy:6.2f}% ({snr_results[snr]['correct']}/{snr_results[snr]['total']})")
        
   
    plt.figure(figsize=(12, 7))
    plt.plot(snr_values, acc_by_snr, marker='o', linestyle='-', color='b')
    plt.title('Model Accuracy vs. SNR')
    plt.xlabel('Signal-to-Noise Ratio (dB)')
    plt.ylabel('Classification Accuracy (%)')
    plt.grid(True, which='both', linestyle='--')
    plt.ylim(0, 100)
    plt.xticks(snr_values)
    plt.tight_layout()
    snr_plot_path = "accuracy_vs_snr.png"
    plt.savefig(snr_plot_path)
    print(f"\nAccuracy vs. SNR plot saved to {snr_plot_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Radar Signal Classifier")
    parser.add_argument('--data_path', type=str, default='data/radar_signal/GOLD_XYZ_OSC.0001_1024.hdf5', help='Path to the HDF5 data file')
    parser.add_argument('--model_path', type=str, default='models/best_radar_cnn.pth', help='Path to the trained model weights')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for evaluation')
    
    args = parser.parse_args()

    if not os.path.isabs(args.data_path):
        args.data_path = os.path.join(os.path.dirname(__file__), '..', args.data_path)
    if not os.path.isabs(args.model_path):
        args.model_path = os.path.join(os.path.dirname(__file__), '..', args.model_path)
    
    evaluate(args)
