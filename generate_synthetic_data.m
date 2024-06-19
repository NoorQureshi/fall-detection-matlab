% Generate synthetic accelerometer and gyroscope data
num_samples = 10000; % Total number of samples
fall_duration = 20; % Duration of the fall spike
num_falls = round(0.01 * num_samples); % 1% falls
fall_indices = randperm(num_samples - fall_duration, num_falls); % Random fall indices

% Initialize accelerometer data with normal activities (random noise)
accel_x = randn(num_samples, 1);
accel_y = randn(num_samples, 1);
accel_z = randn(num_samples, 1);

% Initialize gyroscope data with normal activities (random noise)
gyro_x = randn(num_samples, 1);
gyro_y = randn(num_samples, 1);
gyro_z = randn(num_samples, 1);

% Simulate activities dynamically
for i = 1:num_samples
    if ismember(i, fall_indices) % Simulate falls (1% chance)
        accel_x(i:i+fall_duration) = accel_x(i:i+fall_duration) + 20 * randn(fall_duration + 1, 1); % Very large spike in x direction
        accel_y(i:i+fall_duration) = accel_y(i:i+fall_duration) + 20 * randn(fall_duration + 1, 1); % Very large spike in y direction
        accel_z(i:i+fall_duration) = accel_z(i:i+fall_duration) + 20 * randn(fall_duration + 1, 1); % Very large spike in z direction
        gyro_x(i:i+fall_duration) = gyro_x(i:i+fall_duration) + 15 * randn(fall_duration + 1, 1); % Large spike in x direction
        gyro_y(i:i+fall_duration) = gyro_y(i:i+fall_duration) + 15 * randn(fall_duration + 1, 1); % Large spike in y direction
        gyro_z(i:i+fall_duration) = gyro_z(i:i+fall_duration) + 15 * randn(fall_duration + 1, 1); % Large spike in z direction
    else % Simulate normal activities and other abnormalities
        if rand < 0.02 % 2% chance of walking (small spikes)
            accel_x(i) = accel_x(i) + 3 * randn; % Small spike in x direction
            accel_y(i) = accel_y(i) + 3 * randn; % Small spike in y direction
            accel_z(i) = accel_z(i) + 3 * randn; % Small spike in z direction
        elseif rand < 0.01 % 1% chance of speeding (larger spikes)
            accel_x(i) = accel_x(i) + 10 * randn; % Larger spike in x direction
            accel_y(i) = accel_y(i) + 10 * randn; % Larger spike in y direction
            accel_z(i) = accel_z(i) + 10 * randn; % Larger spike in z direction
        elseif rand < 0.005 % 0.5% chance of subtle abnormality (medium spikes)
            accel_x(i) = accel_x(i) + 7 * randn; % Medium spike in x direction
            accel_y(i) = accel_y(i) + 7 * randn; % Medium spike in y direction
            accel_z(i) = accel_z(i) + 7 * randn; % Medium spike in z direction
            gyro_x(i) = gyro_x(i) + 5 * randn; % Medium spike in x direction
            gyro_y(i) = gyro_y(i) + 5 * randn; % Medium spike in y direction
            gyro_z(i) = gyro_z(i) + 5 * randn; % Medium spike in z direction
        end
    end
end

% Create a table for the dataset
timestamp = (1:num_samples)';
fall_data = table(timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z);

% Save the data to a CSV file
filename = 'realistic_fall_data.csv';
writetable(fall_data, filename);
disp(['Realistic synthetic dataset saved to ', filename]);
