function fall_detection()
    % Main Fall Detection Code

    % Define the filename for the realistic synthetic dataset
    filename = 'realistic_fall_data.csv';

    % Load the synthetic dataset
    fall_data = readtable(filename);

    % Calculate the magnitude of acceleration and gyroscope
    fall_data.accel_magnitude = sqrt(fall_data.accel_x.^2 + fall_data.accel_y.^2 + fall_data.accel_z.^2);
    fall_data.gyro_magnitude = sqrt(fall_data.accel_x.^2 + fall_data.accel_y.^2 + fall_data.accel_z.^2);

    % Define labels (1 for fall, 0 for no fall)
    fall_threshold_accel = 30; % Adjusted threshold for accelerometer to reduce false positives
    fall_threshold_gyro = 20; % Adjusted threshold for gyroscope to reduce false positives
    fall_data.label = double(fall_data.accel_magnitude > fall_threshold_accel & fall_data.gyro_magnitude > fall_threshold_gyro);

    % Define window size for feature extraction
    window_size = 100;

    features = [];
    labels = [];

    for i = 1:window_size:(height(fall_data) - window_size)
        window_data = fall_data(i:(i + window_size - 1), :);

        % Calculate features for the window
        mean_accel = mean(window_data{:, {'accel_x', 'accel_y', 'accel_z'}});
        std_accel = std(window_data{:, {'accel_x', 'accel_y', 'accel_z'}});
        mean_gyro = mean(window_data{:, {'gyro_x', 'gyro_y', 'gyro_z'}});
        std_gyro = std(window_data{:, {'gyro_x', 'gyro_y', 'gyro_z'}});

        % Magnitude of acceleration and gyroscope
        accel_magnitude = sqrt(sum(window_data{:, {'accel_x', 'accel_y', 'accel_z'}}.^2, 2));
        mean_accel_magnitude = mean(accel_magnitude);
        std_accel_magnitude = std(accel_magnitude);

        gyro_magnitude = sqrt(sum(window_data{:, {'gyro_x', 'gyro_y', 'gyro_z'}}.^2, 2));
        mean_gyro_magnitude = mean(gyro_magnitude);
        std_gyro_magnitude = std(gyro_magnitude);

        % Combine features into a single row
        feature_row = [mean_accel, std_accel, mean_gyro, std_gyro, mean_accel_magnitude, std_accel_magnitude, mean_gyro_magnitude, std_gyro_magnitude];
        features = [features; feature_row];

        % Assign label (majority label in the window)
        labels = [labels; mode(window_data.label)];
    end

    % Convert to table
    features = array2table(features, 'VariableNames', {'mean_accel_x', 'mean_accel_y', 'mean_accel_z', ...
        'std_accel_x', 'std_accel_y', 'std_accel_z', 'mean_gyro_x', 'mean_gyro_y', 'mean_gyro_z', ...
        'std_gyro_x', 'std_gyro_y', 'std_gyro_z', 'mean_accel_magnitude', 'std_accel_magnitude', ...
        'mean_gyro_magnitude', 'std_gyro_magnitude'});
    features.label = labels;

    % Split data into training, validation, and testing sets
    cv = cvpartition(height(features), 'HoldOut', 0.3);
    train_data = features(training(cv), :);
    test_val_data = features(test(cv), :);

    cv2 = cvpartition(height(test_val_data), 'HoldOut', 0.5);
    val_data = test_val_data(training(cv2), :);
    test_data = test_val_data(test(cv2), :);

    % Separate predictors and labels
    train_X = train_data{:, 1:end-1};
    train_Y = train_data.label;
    val_X = val_data{:, 1:end-1};
    val_Y = val_data.label;
    test_X = test_data{:, 1:end-1};
    test_Y = test_data.label;

    % Train SVM model
    svm_model = fitcsvm(train_X, train_Y, 'KernelFunction', 'linear', 'Standardize', true);

    % Increase iteration limit for logistic regression
    options = statset('glmfit');
    options.MaxIter = 10000; % Further increase the iteration limit

    % Train Logistic Regression model with increased iteration limit
    logistic_model = fitglm(train_X, train_Y, 'Distribution', 'binomial', 'Link', 'logit', 'Options', options);

    % Validate models
    svm_val_predictions = predict(svm_model, val_X);
    svm_val_accuracy = sum(svm_val_predictions == val_Y) / length(val_Y);
    fprintf('SVM Validation Accuracy: %.2f%%\n', svm_val_accuracy * 100);

    logistic_val_predictions = round(predict(logistic_model, val_X));
    logistic_val_accuracy = sum(logistic_val_predictions == val_Y) / length(val_Y);
    fprintf('Logistic Regression Validation Accuracy: %.2f%%\n', logistic_val_accuracy * 100);

    % Test models
    svm_test_predictions = predict(svm_model, test_X);
    svm_test_accuracy = sum(svm_test_predictions == test_Y) / length(test_Y);
    fprintf('SVM Test Accuracy: %.2f%%\n', svm_test_accuracy * 100);

    logistic_test_predictions = round(predict(logistic_model, test_X));
    logistic_test_accuracy = sum(logistic_test_predictions == test_Y) / length(test_Y);
    fprintf('Logistic Regression Test Accuracy: %.2f%%\n', logistic_test_accuracy * 100);

    % Evaluate models with confusion matrices
    figure;
    subplot(2, 2, 1);
    confusionchart(val_Y, svm_val_predictions);
    title('SVM Validation Confusion Matrix');

    subplot(2, 2, 2);
    confusionchart(val_Y, logistic_val_predictions);
    title('Logistic Regression Validation Confusion Matrix');

    subplot(2, 2, 3);
    confusionchart(test_Y, svm_test_predictions);
    title('SVM Test Confusion Matrix');

    subplot(2, 2, 4);
    confusionchart(test_Y, logistic_test_predictions);
    title('Logistic Regression Test Confusion Matrix');

    % Visualize detected falls
    figure;
    hold on;
    plot(fall_data.timestamp, fall_data.accel_magnitude);
    plot(fall_data.timestamp, fall_data.gyro_magnitude);
    scatter(fall_data.timestamp(fall_data.label == 1), fall_data.accel_magnitude(fall_data.label == 1), 'r', 'filled');
    scatter(fall_data.timestamp(fall_data.label == 1), fall_data.gyro_magnitude(fall_data.label == 1), 'r', 'filled');
    xlabel('Timestamp');
    ylabel('Magnitude');
    legend('Accel Magnitude', 'Gyro Magnitude', 'Detected Falls');
    title('Fall Detection Visualization');
    hold off;

    % Check for detected falls and display details
    fall_indices = find(fall_data.label == 1);
    if ~isempty(fall_indices)
        fall_data_subset = fall_data(fall_indices, :);
        display_fall_details(fall_data_subset);
%        send_email_alert(fall_data_subset);
        play_sound_alert();
    end
end

function display_fall_details(fall_data)
    fprintf('Falls Detected:\n');
    for i = 1:height(fall_data)
        timestamp = fall_data.timestamp(i);
        accel_magnitude = fall_data.accel_magnitude(i);
        gyro_magnitude = fall_data.gyro_magnitude(i);
        fprintf('Time: %s, Accel Magnitude: %.2f, Gyro Magnitude: %.2f\n', ...
                datestr(seconds(timestamp), 'HH:MM:SS'), accel_magnitude, gyro_magnitude);
    end
end

function send_email_alert(fall_data)
    % Set up email preferences
    setpref('Internet', 'SMTP_Server', 'smtp.gmail.com');
    setpref('Internet', 'E_mail', 'your_email@gmail.com');
    setpref('Internet', 'SMTP_Username', 'your_email@gmail.com');
    setpref('Internet', 'SMTP_Password', 'your_password');

    % Configure SMTP properties
    props = java.lang.System.getProperties;
    props.setProperty('mail.smtp.auth', 'true');
    props.setProperty('mail.smtp.starttls.enable', 'true');
    props.setProperty('mail.smtp.port', '587');

    % Convert fall data to string
    fall_info = '';
    for i = 1:height(fall_data)
        fall_info = strcat(fall_info, sprintf('Time: %s, Accel Magnitude: %.2f, Gyro Magnitude: %.2f\n', ...
            datestr(seconds(fall_data.timestamp(i)), 'HH:MM:SS'), fall_data.accel_magnitude(i), fall_data.gyro_magnitude(i)));
    end

    % Email content
    subject = 'Fall Detection Alert';
    message = sprintf('A fall has been detected:\n\n%s', fall_info);

    % Send email
    sendmail('recipient_email@example.com', subject, message);
end

function play_sound_alert()
    % Generate a simple beep sound
    fs = 44100;  % Sample rate
    t = 0:1/fs:0.5;  % Time vector
    y = sin(2*pi*1000*t);  % Generate a 1 kHz tone
    sound(y, fs);  % Play the sound
end
