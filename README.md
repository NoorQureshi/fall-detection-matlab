# Real-Time Fall Detection using SVM and Logistic Regression

This MATLAB project demonstrates a real-time fall detection system using accelerometer and gyroscope data. The system utilizes Support Vector Machines (SVM) and Logistic Regression models to detect falls and provides real-time visualization.

# Introduction
This project implements a fall detection system that monitors accelerometer and gyroscope data to detect falls. The system uses machine learning models (SVM and Logistic Regression) for detection and provides real-time alerts and visualization.

# Dataset Generation
The script `generateRealisticFallDataset.m` generates a synthetic dataset of accelerometer and gyroscope data with labeled falls.

### Usage

To generate a dataset with a specified number of non-fall and fall samples, run:

```MATLP
generateRealisticFallDataset(numNonFalls, numFalls);
```
For example, to generate a dataset with 800 non-falls and 100 falls:

```MATLAB
generateRealisticFallDataset(800, 100);
```

## Real-Time Fall Detection
The script `generateRealisticFallDataset.m` performs real-time fall detection using the generated dataset. It loads the trained models, processes the data in real-time, and provides alerts and visualization.

### Usage
To run the real-time fall detection:

```MATLP
realTimeFallDetection(();
```
The script will:
1. Load the latest dataset.
2. Load the trained SVM and Logistic Regression models.
3. Process the data in real-time, highlighting detected falls with red markers and providing sound alerts.
4. Display the accelerometer and gyroscope data as continuous lines.

## Model Evaluation
The script `real_timefall_detection_flexible.m` also evaluates the performance of the SVM and Logistic Regression models by plotting confusion matrices and calculating accuracy.
### Usage
To evaluate the models:

```MATLAB
realTimeFallDetection();
```
The evaluation results will be displayed in a separate figure, showing confusion matrices and accuracy for both models.

### Usage
Clone the Repository

```
git clone https://github.com/YourUsername/fall-detection-matlab.git
cd fall-detection-matlab
```

Generate Dataset

* Open MATLAB and run:


```generateRealisticFallDataset(800, 100);```

Run Real-Time Fall Detection

* Open MATLAB and run:

```realTimeFallDetection();```

* Requirements
MATLAB R2019b or later
Statistics and Machine Learning Toolbox
