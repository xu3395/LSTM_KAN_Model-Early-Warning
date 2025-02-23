# Earthquake Prediction Model Using LSTM and KAN

This project implements a hybrid deep learning model combining LSTM networks and KAN (Kolmogorov-Arnold Networks) for Secondary collapse of RC frame building prediction using PyTorch. The code supports comprehensive data processing, model training, prediction evaluation, and multidimensional visualization.
## Features

- **LSTM**: A deep learning architecture for sequential data, used to process time-series data.
- **KAN**: A custom layer designed for improved feature extraction and representation.
- **Data Augmentation**: Noise is added to the training data to increase robustness.
- **Model Evaluation**: Metrics like RMSE, MAE, and R² are used for evaluating the model's performance.
- **Early Stopping**: Stops training when the model's performance on the validation set starts to deteriorate.
- **Visualization**: Several visualizations for training loss, residuals, and error distributions.

## Key Features
- **📈 Two-stage hybrid architecture**: Deep integration of LSTM and KAN networks
- **🛠️ Advanced data processing**: Sliding window normalization + standardization + data augmentation
- **📊 Multidimensional evaluation**: RMSE/MAE/R² metrics + 8 visualization analysis methods
- **⚙️ Adaptive training mechanisms**: Dynamic learning rate adjustment + early stopping
- **🔍 Interpretability analysis**: KAN activation function visualization

- 
## Project Structure
-  ├── main.py                 # Main entry
- ├── best_lstm_kan_model.pth # Best model checkpoint
- ├── train_predictions.csv   # Training set predictions
-  ├── test_predictions.csv    # Testing set predictions
-  ├── results.json            # Evaluation metrics
-  ├── Learning_Curve.png      # Training process visualization
-  └── Feature_*.png           # Dimension-wise prediction plots



## Installation

1. Clone the repository:
   ```bash
   git clone  https://github.com/xu3395/LSTM_KAN_Model-Early-Warning.git
