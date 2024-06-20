% generate_realistic_fall_dataset.m

function generateRealisticFallDataset(numNonFalls, numFalls)
    % Generate synthetic timestamps (in seconds over 24 hours)
    timestamps = sort(randperm(86400, numNonFalls + numFalls))'; % 86400 seconds in 24 hours

    % Generate walking pattern for non-fall data (accelerometer and gyroscope)
    nonFallAccelData = 0.5 * randn(numNonFalls, 3) + 1; % Simulating normal walking
    nonFallGyroData = 0.2 * randn(numNonFalls, 3) + 0.5; % Simulating normal walking

    % Generate spikes for fall data
    fallAccelData = 5 * randn(numFalls, 3) + 10; % Simulating falls
    fallGyroData = 2 * randn(numFalls, 3) + 5; % Simulating falls

    % Combine non-fall and fall data
    accelData = [nonFallAccelData; fallAccelData];
    gyroData = [nonFallGyroData; fallGyroData];
    labels = [zeros(numNonFalls, 1); ones(numFalls, 1)];

    % Shuffle data
    shuffleIdx = randperm(length(timestamps));
    timestamps = timestamps(shuffleIdx);
    accelData = accelData(shuffleIdx, :);
    gyroData = gyroData(shuffleIdx, :);
    labels = labels(shuffleIdx, :);

    % Combine accelerometer, gyroscope data, and timestamps
    dataset = [timestamps, accelData, gyroData, labels];

    % Save the dataset to a CSV file
    csvFileName = sprintf('realistic_dataset_with_%d_falls.csv', numFalls);
    csvwrite(csvFileName, dataset);

    disp(['Realistic dataset with ', num2str(numFalls), ' falls and ', num2str(numNonFalls), ' non-falls generated and saved as ', csvFileName]);
end

% Example usage:
% generateRealisticFallDataset(800, 100);
