% real_time_fall_detection_flexible.m

function realTimeFallDetection()
    % Load the trained models
    load('svmModel.mat', 'svmModel');
    load('logisticModel.mat', 'logisticModel');

    % Find the latest dataset file
    files = dir('realistic_dataset_with_*.csv');
    [~, idx] = max([files.datenum]);
    latestFile = files(idx).name;

    % Load the custom dataset
    dataset = csvread(latestFile);
    timestamps = dataset(:, 1); % The first column is timestamps
    accelData = dataset(:, 2:4); % Accelerometer data (x, y, z)
    gyroData = dataset(:, 5:7); % Gyroscope data (x, y, z)
    labels = dataset(:, end);   % The last column

    % Verify the number of falls and non-falls
    numFalls = sum(labels == 1);
    numNonFalls = sum(labels == 0);
    fprintf('Loaded dataset with %d falls and %d non-falls\n', numFalls, numNonFalls);

    % Convert timestamps to hours and minutes for better readability
    formattedTimestamps = datetime(timestamps, 'ConvertFrom', 'posixtime', 'Epoch', '1970-01-01', 'Format', 'HH:mm:ss');

    % Compute magnitudes of accelerometer and gyroscope data
    accelMagnitude = sqrt(sum(accelData.^2, 2));
    gyroMagnitude = sqrt(sum(gyroData.^2, 2));

    % Initialize figure for real-time visualization
    figure('Name', 'Fall Detection Visualization', 'Position', [100, 100, 1200, 600]); % Increase figure size
    hold on;
    title('Fall Detection Visualization');
    xlabel('Time');
    ylabel('Magnitude');
    grid on; % Add grid lines

    % Plot accelerometer and gyroscope magnitudes
    hAccel = plot(formattedTimestamps, accelMagnitude, 'b-', 'DisplayName', 'Accel Magnitude', 'LineWidth', 1.5);
    hGyro = plot(formattedTimestamps, gyroMagnitude, 'g-', 'DisplayName', 'Gyro Magnitude', 'LineWidth', 1.5);

    % Initialize red dots for detected falls
    fallMarkersAccel = plot(nan, nan, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'DisplayName', 'Detected Falls Accel');
    fallMarkersGyro = plot(nan, nan, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'DisplayName', 'Detected Falls Gyro');
    
    % Adjust the legend position
    legend('show', 'Location', 'northwest');

    fallTimestamps = [];
    accelFallYData = [];
    gyroFallYData = [];

    for i = 1:size(dataset, 1)
        % Extract current data point
        currentData = dataset(i, 2:end-1);

        % Predict using the SVM model
        svmPrediction = predict(svmModel, currentData);

        % Predict using the Logistic Regression model
        logisticPrediction = round(predict(logisticModel, currentData));

        % Check if a fall is detected
        if svmPrediction == 1 || logisticPrediction == 1
            % Sound alert
            beep;

            % Record the timestamp and data of the fall
            fallTimestamps = [fallTimestamps; formattedTimestamps(i)];
            accelFallYData = [accelFallYData; accelMagnitude(i)];
            gyroFallYData = [gyroFallYData; gyroMagnitude(i)];

            % Display alert message
            disp(['Fall detected at timestamp: ', datestr(formattedTimestamps(i), 'HH:MM:SS')]);
        end
        
        % Update the plots with new data points
        set(hAccel, 'YData', accelMagnitude);
        set(hGyro, 'YData', gyroMagnitude);
        set(fallMarkersAccel, 'XData', fallTimestamps, 'YData', accelFallYData);
        set(fallMarkersGyro, 'XData', fallTimestamps, 'YData', gyroFallYData);

        % Pause to simulate real-time processing
        pause(0.01); % Adjust the pause duration as needed
    end

    hold off;

    % Display total number of falls detected
    fprintf('Total number of falls detected: %d\n', length(fallTimestamps));
    disp('Timestamps of detected falls:');
    disp(datestr(fallTimestamps, 'HH:MM:SS'));

    % Evaluate models and plot confusion matrices
    evaluateModelsWithCustomDataset(latestFile);
end

function evaluateModelsWithCustomDataset(latestFile)
    % Load the trained models
    load('svmModel.mat', 'svmModel');
    load('logisticModel.mat', 'logisticModel');

    % Load the custom dataset
    dataset = csvread(latestFile);
    labels = dataset(:, end);   % The last column

    % Combine accelerometer and gyroscope data
    data = dataset(:, 2:7);

    % Predict using the SVM model
    svmPredictions = predict(svmModel, data);

    % Predict using the Logistic Regression model
    logisticPredictions = round(predict(logisticModel, data));

    % Compute confusion matrices
    svmConfusionMatrix = confusionmat(labels, svmPredictions);
    logisticConfusionMatrix = confusionmat(labels, logisticPredictions);

    % Plot confusion matrices side by side
    figure('Name', 'Model Performance Evaluation', 'Position', [100, 100, 1200, 600]); % Increase figure size

    % SVM Confusion Matrix
    subplot(1, 2, 1);
    confusionchart(svmConfusionMatrix);
    title('Confusion Matrix for SVM');

    % Logistic Regression Confusion Matrix
    subplot(1, 2, 2);
    confusionchart(logisticConfusionMatrix);
    title('Confusion Matrix for Logistic Regression');
    
    % Calculate and display accuracy for both models
    svmAccuracy = sum(svmPredictions == labels) / length(labels);
    logisticAccuracy = sum(logisticPredictions == labels) / length(labels);
    
    fprintf('SVM Accuracy: %.2f%%\n', svmAccuracy * 100);
    fprintf('Logistic Regression Accuracy: %.2f%%\n', logisticAccuracy * 100);
end

% Example usage:
% generateRealisticFallDataset(800, 100);
% realTimeFallDetection();
