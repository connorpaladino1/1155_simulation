import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import TimeSeriesDataset
from Temporal_Convolution_Network import TCNAnomalyDetector
from data import data_info
import matplotlib.pyplot as plt

# Load dataset
sequence_length = 20
train_data, train_labels = data_info['train_data'], data_info['train_labels']
test_data, test_labels = data_info['test_data'], data_info['test_labels']

# Create datasets for training and validation
train_dataset = TimeSeriesDataset(train_data, train_labels, sequence_length)
val_dataset = TimeSeriesDataset(test_data, test_labels, sequence_length)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model, loss function, and optimizer
model = TCNAnomalyDetector()
criterion = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

epochs = [100]
epoch_cumulative = 0

train_accuracies = []
val_accuracies = []

for index, e in enumerate(epochs):
    train_losses = []
    val_losses = []
    
    # Training loop
    for epoch in range(e):
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0
        for inputs, targets in train_dataloader:
            # inputs = inputs.permute(0, 2, 1)
            optimizer.zero_grad()        
            # outputs = model(inputs.unsqueeze(1))                        
            outputs = model(inputs)
            outputs = outputs[:, -1, 0]
            loss = criterion(outputs, targets.float())        
            train_loss += loss.item()        
            loss.backward()
            optimizer.step()
            
            # Calculate training accuracy
            predicted_train = (outputs > 0.5).float()
            correct_train += (predicted_train == targets).sum().item()
            total_train += targets.size(0)
            
        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)        
        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)

        # Validation loop
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                # inputs = inputs.permute(0, 2, 1)
                # outputs = model(inputs.unsqueeze(1))
                outputs = model(inputs)
                outputs = outputs[:, -1, 0]                
                val_loss += criterion(outputs, targets).item()
                
                # Calcualte validation accuracy
                predicted_val = (outputs > 0.5).float()
                correct_val += (predicted_val == targets).sum().item()
                total_val += targets.size(0)
                
        val_accuracy = 100 * correct_val / total_val
        val_accuracies.append(val_accuracy)
        val_loss /= len(val_dataloader)
        val_losses.append(val_loss)
        
        # Print both train and validation loss
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%')
            
    epoch_cumulative += e
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Training and Validation Loss for Epochs {e}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')    
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.savefig(f'C:/Users/Proto/Desktop/Neural Network/training_validation_loss_{index}.png')
    plt.close()
    
    print("\n")

# Save the model
torch.save(model.state_dict(), 'tcn_anomaly_detector.pth')
