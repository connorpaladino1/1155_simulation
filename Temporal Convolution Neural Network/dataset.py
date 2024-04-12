import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels, sequence_length):
        self.data = data
        self.labels = labels
        self.sequence_length = sequence_length
        self.sequences, self.sequence_labels = self._create_sequences()

    def _create_sequences(self):
        sequences = []
        sequence_labels = []
        for start in range(len(self.data) - self.sequence_length):
            end = start + self.sequence_length
            sequences.append(self.data[start:end])
            sequence_labels.append(self.labels[end-1])
        return sequences, sequence_labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence, label = self.sequences[idx], self.sequence_labels[idx]
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
