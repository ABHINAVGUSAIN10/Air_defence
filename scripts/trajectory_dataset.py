import os
import pandas as pd
import torch
from torch.utils.data import Dataset

class TrajectoryDataset(Dataset):
    def __init__(self, folder_path, input_len=20, pred_len=10):
        self.samples = []
        self.input_len = input_len
        self.pred_len = pred_len

        for file in os.listdir(folder_path):  # ← this MUST be inside __init__
            if file.endswith(".txt"):
                path = os.path.join(folder_path, file)
                try:
                    df = pd.read_csv(path, delim_whitespace=True, header=None)
                    df.columns = ["Frame #", "Aircraft ID", "x (km)", "y (km)", "z (km)", "windx (m/s)", "windy (m/s)"]
                except Exception as e:
                    print(f"❌ Failed to process {file}: {e}")
                    continue

                for aid, group in df.groupby("Aircraft ID"):
                    group = group.sort_values("Frame #")
                    arr = group[["x (km)", "y (km)", "z (km)", "windx (m/s)", "windy (m/s)"]].values

                    total_len = input_len + pred_len
                    if len(arr) >= total_len:
                        for i in range(len(arr) - total_len + 1):
                            input_seq = arr[i : i + input_len]
                            target_seq = arr[i + input_len : i + total_len, :3]  # x,y,z only
                            self.samples.append((input_seq, target_seq))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_seq, target_seq = self.samples[idx]
        return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(target_seq, dtype=torch.float32)


# Example usage:
if __name__ == '__main__':
    train_path = r"C:\Users\Abhinav Gusain\Documents\Air_defence\data\trajectory\7days1\processed_data\train"
    dataset = TrajectoryDataset(train_path)

    print("Total training sequences:", len(dataset))
    sample_input, sample_output = dataset[0]
    print("Input shape:", sample_input.shape)   # Expect: [20, 5]
    print("Target shape:", sample_output.shape) # Expect: [10, 3]
