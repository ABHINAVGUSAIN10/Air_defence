import torch
import torch.nn as nn

class TrajectoryPredictor(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, num_layers=2, output_size=3):
        super(TrajectoryPredictor, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, pred_len=10):
        batch_size = input_seq.size(0)

        # Encode the input sequence
        _, (hidden, cell) = self.encoder(input_seq)

        # Initialize decoder input (start with last known position)
        decoder_input = input_seq[:, -1, :3].unsqueeze(1)  # [B, 1, 3]

        outputs = []

        for _ in range(pred_len):
            out, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            pred = self.fc_out(out)
            outputs.append(pred)
            decoder_input = pred  # feedback for next step

        return torch.cat(outputs, dim=1)  # [B, pred_len, 3]
