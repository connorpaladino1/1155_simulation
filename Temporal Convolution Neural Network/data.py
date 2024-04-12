import numpy as np
import matplotlib.pyplot as plt

# Load data from text file
data_file = 'C:/Users/Proto/Desktop/Neural Network/Data_New_2.txt'

with open(data_file, 'r') as file:
    data_string = file.read()
    data_string = data_string.strip('[]')
    data_rows = data_string.split('],[')
    data = np.array([list(map(float, row.split(','))) for row in data_rows])

# data = np.loadtxt(data_file, delimiter=',')

time_steps = np.linspace(0, 2*np.pi, len(data))

# Label data
labels = np.zeros(len(data))

# Split data into training and testing sets
split_index = int(0.7 * len(data))
train_data = data[:split_index]
train_labels = labels[:split_index]
test_data = data[split_index:]
test_labels = labels[split_index:]

data_info = {'train_data': train_data, 'train_labels': train_labels,
             'test_data': test_data, 'test_labels': test_labels}

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(time_steps[:split_index], train_data[:, 0], label='Training Data', color='blue')
plt.plot(time_steps[split_index:], test_data[:, 0], label='Test Data', color='green')
plt.scatter(time_steps, data[:, 0], label='Data Points', color='red')
plt.title('Synthetic Time-Series Data with Outliers')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.savefig('C:/Users/Proto/Desktop/Neural Network/data.png')
plt.close()
