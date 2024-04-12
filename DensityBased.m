% Load Data
data = load('Data_New.txt'); % Load data from a text file

% Set parameters
epsilon = 0.5;  % Adjust epsilon as needed
min_points = 10; % Adjust min_points as needed

% Perform DBSCAN
[cluster_ids, outliers] = dbscan(data, epsilon, min_points);

% Plot the clusters
figure;
gscatter(data(:,1), data(:,2), cluster_ids);
hold on;
title('DBSCAN Clustering with Outliers');
xlabel('X');
ylabel('Y');
legend('Anomalies', 'Cluster 1', 'Cluster 2');

% Count the number of red dots (anomalies)
num_anomalies = sum(cluster_ids == -1);

% Display the number of anomalies
disp(['Number of anomalies detected: ', num2str(num_anomalies)]);
