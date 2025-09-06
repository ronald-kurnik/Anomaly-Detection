# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 12:17:13 2025

@author: Ron
"""

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, ReLU, Dropout, Conv1DTranspose
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler 
import matplotlib.pyplot as plt

# --- 1. Data Import and Preparation (from classification_1d_Input.py) ---

# Load data from HDF5 file
hdf5_file_name = 'exported_data_from_matlab.h5'
imported_python_data = []

try:
    with h5py.File(hdf5_file_name, 'r') as f:
        data_group = f['/my_cell_data_group']
        dataset_names = sorted(data_group.keys())

        for name in dataset_names:
            raw_array = data_group[name][()]
            # Transpose if the data format is (channels, timesteps) to (timesteps, channels)
            if raw_array.ndim == 2 and raw_array.shape[0] == 3 and raw_array.shape[1] > 3:
                imported_array = raw_array.T
            else:
                imported_array = raw_array

            imported_python_data.append(imported_array.astype(np.float32))

    print(f"Successfully loaded {len(imported_python_data)} data samples from {hdf5_file_name}")
except FileNotFoundError:
    print(f"Error: HDF5 file '{hdf5_file_name}' not found. Please ensure MATLAB script has run and created it.")
    exit()
except KeyError:
    print(f"Error: HDF5 group '/my_cell_data_group' not found in '{hdf5_file_name}'. Check MATLAB export script's group name.")
    exit()
except Exception as e:
    print(f"An error occurred during HDF5 loading: {e}")
    exit()


# Find the maximum length among all sequences and confirm number of channels
if imported_python_data:
    max_sequence_length = max(arr.shape[0] for arr in imported_python_data)
    num_channels = imported_python_data[0].shape[1]
else:
    print("Error: No data imported to determine sequence length and channels.")
    exit()

print(f"Maximum sequence length found: {max_sequence_length}")
print(f"Number of channels detected: {num_channels}")

# Manually pad sequences to a common maximum length (pre-padding, value=0.0)
padded_data_list = []
for i, sequence in enumerate(imported_python_data):
    if sequence.ndim != 2 or sequence.shape[1] != num_channels:
        print(f"Warning: Sample {i} has unexpected shape {sequence.shape}. Expected (timesteps, {num_channels}). Skipping or check data source.")
        continue

    current_length = sequence.shape[0]
    if current_length < max_sequence_length:
        pad_length = max_sequence_length - current_length
        padded_sequence = np.pad(sequence, ((pad_length, 0), (0, 0)),
                                 mode='constant', constant_values=0.0)
    else:
        padded_sequence = sequence[:max_sequence_length, :]

    padded_data_list.append(padded_sequence)

X_all = np.array(padded_data_list, dtype=np.float32)
print(f"Padded data (X_all) final shape: {X_all.shape}")

# Split data into training and validation sets (90-10)

num_observations = X_all.shape[0]
X_train_raw, X_validation_raw = train_test_split(X_all, test_size=0.1, random_state=42, shuffle=True)

print(f"X_train_raw shape: {X_train_raw.shape}")
print(f"X_validation_raw shape: {X_validation_raw.shape}")

# --- Z-score Normalization using StandardScaler ---

# Reshape data for StandardScaler: (num_samples * timesteps, num_channels)
# StandardScaler expects 2D array (n_samples, n_features)
# Normalize each channel independently across all timesteps and samples.
# So, flatten X_train_raw along the time dimension, but keep channels.
X_train_flat = X_train_raw.reshape(-1, num_channels)

scaler = StandardScaler()
scaler.fit(X_train_flat) # Fit ONLY on training data

# Transform all relevant datasets
X_train_normalized = scaler.transform(X_train_flat).reshape(X_train_raw.shape)
X_validation_normalized = scaler.transform(X_validation_raw.reshape(-1, num_channels)).reshape(X_validation_raw.shape)

print(f"X_train_normalized shape: {X_train_normalized.shape}")
print(f"X_validation_normalized shape: {X_validation_normalized.shape}")

# --- 2. Define Network Architecture ---

num_downsamples = 2
filter_size = 7
num_filters_base = 16 # Base number of filters, will be multiplied
dropout_prob = 0.4

# Input layer
input_seq = Input(shape=(max_sequence_length, num_channels))

x = input_seq
# Instantiate the normalization layer
normalizer = tf.keras.layers.Normalization(axis=-1)

# Adapt the layer to the training data. This is a one-time step.
# It calculates the mean and variance for each feature (channel).
normalizer.adapt(X_train_raw)

print("Normalization layer adapted to training data.")

# Input layer
input_seq = Input(shape=(max_sequence_length, num_channels))

# Add the pre-adapted normalization layer directly after the input
x = normalizer(input_seq)

# Encoder (Downsampling)
for i in range(num_downsamples):
    # MATLAB: (numDownsamples+1-i)*numFilters
    current_filters = (num_downsamples + 1 - i) * num_filters_base
    x = Conv1D(filters=current_filters, kernel_size=filter_size, padding='same', strides=2)(x)
    x = ReLU()(x)
    x = Dropout(dropout_prob)(x)

# Decoder (Upsampling)
for i in range(1, num_downsamples + 1): # i from 1 to num_downsamples
    # MATLAB: i*numFilters
    current_filters = i * num_filters_base
    x = Conv1DTranspose(filters=current_filters, kernel_size=filter_size, padding='same', strides=2)(x)
    x = ReLU()(x)
    x = Dropout(dropout_prob)(x)

# Output layer
output_seq = Conv1DTranspose(filters=num_channels, kernel_size=filter_size, padding='same')(x)

# Create the autoencoder model
autoencoder = Model(inputs=input_seq, outputs=output_seq)

# Compile the model
autoencoder.compile(optimizer=Adam(), loss='mse')

autoencoder.summary()

# --- 3. Train Network ---

# Training options (mimicking MATLAB's trainingOptions)
epochs = 120
batch_size = 32 # Keras typically uses batch_size, MATLAB uses mini-batch size implicitly
# Validation data is both input and target for autoencoder
validation_data = (X_validation_raw, X_validation_raw)
validation_data_normalized = (X_validation_normalized, X_validation_normalized)

print("\nStarting training...")
history = autoencoder.fit(
    X_train_normalized, X_train_normalized, # Use the normalized training data
    epochs=epochs,
    batch_size=batch_size,
    validation_data=validation_data_normalized, # Use the normalized validation data
    shuffle=True, # Shuffle every epoch
    verbose=0 # Display training progress
)

print("\nTraining finished.")

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Autoencoder Training Progress')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend()
plt.grid(True)
plt.show()

# --- 4. Test Network and Identify Anomalous Sequences ---

# Make predictions on validation data to establish baseline RMSE
Y_validation_pred = autoencoder.predict(X_validation_raw)

# Calculate RMSE for each validation sequence
# RMSE calculation: sqrt(mean(squared_error))
def calculate_rmse_per_sequence(original, reconstructed):
    # original and reconstructed are (timesteps, channels)
    squared_error = np.square(original - reconstructed)
    # Sum over channels, then mean over timesteps, then sqrt
    return np.sqrt(np.mean(squared_error, axis=(0, 1))) # Mean over all elements in sequence

err_validation = []
for i in range(X_validation_raw.shape[0]):
    rmse_val = calculate_rmse_per_sequence(X_validation_raw[i], Y_validation_pred[i])
    err_validation.append(rmse_val)

err_validation = np.array(err_validation)

# Visualize the RMSE values in a histogram for representative samples
plt.figure(figsize=(10, 6))
plt.hist(err_validation, bins=30, edgecolor='black')
plt.xlabel("Root Mean Square Error (RMSE)")
plt.ylabel("Frequency")
plt.title("RMSE for Representative Samples (Validation Data)")
plt.grid(True)
plt.show()

# Determine the maximum RMSE from the validation data as a baseline
RMSEbaseline = np.max(err_validation)
print(f"\nBaseline RMSE (max from validation data): {RMSEbaseline:.4f}")

# Create new data with anomalies (mimicking MATLAB's manual anomaly injection)
X_new = np.copy(X_validation_raw) # Start with a copy of validation data

num_anomalous_sequences = 20
# Randomly select indices to modify
np.random.seed(42) # for reproducibility
idx_anomalous = np.random.choice(X_new.shape[0], num_anomalous_sequences, replace=False)

for i in idx_anomalous:
    X_sample = X_new[i]
    # Define a patch (e.g., time steps 50 to 60)
    idx_patch_start = 50
    idx_patch_end = 60
    # Ensure patch is within sequence bounds
    idx_patch_end = min(idx_patch_end, X_sample.shape[0])
    idx_patch_start = min(idx_patch_start, idx_patch_end)

    if idx_patch_start < idx_patch_end:
        # Apply anomaly: 4 * abs(original_patch)
        X_sample[idx_patch_start:idx_patch_end, :] = 4 * np.abs(X_sample[idx_patch_start:idx_patch_end, :])
        X_new[i] = X_sample # Update the modified sample

# Make predictions on the new data
X_new_normalized = scaler.transform(X_new.reshape(-1, num_channels)).reshape(X_new.shape)
Y_new_pred = autoencoder.predict(X_new_normalized)

# Calculate RMSE for the new data
err_new = []
for i in range(X_new.shape[0]):
    rmse_val = calculate_rmse_per_sequence(X_new[i], Y_new_pred[i])
    err_new.append(rmse_val)

err_new = np.array(err_new)

# Visualize the RMSE values for new samples
plt.figure(figsize=(10, 6))
plt.hist(err_new, bins=30, edgecolor='black', label='New Samples RMSE')
plt.axvline(x=RMSEbaseline, color='r', linestyle='--', label=f'Baseline RMSE ({RMSEbaseline:.4f})')
plt.xlabel("Root Mean Square Error (RMSE)")
plt.ylabel("Frequency")
plt.title("RMSE for New Samples (with injected anomalies)")
plt.legend()
plt.grid(True)
plt.show()

# Identify the top 10 sequences with the largest RMSE values
idx_top_anomalous = np.argsort(err_new)[::-1][:10] # Sort descending and take top 10
print(f"\nIndices of top 10 most anomalous sequences (0-indexed): {idx_top_anomalous}")

# Visualize the sequence with the largest RMSE value and its reconstruction
top_anomaly_idx_in_X_new = idx_top_anomalous[0] # Get the index in X_new
X_top_anomaly = X_new[top_anomaly_idx_in_X_new]
Y_top_anomaly_pred = Y_new_pred[top_anomaly_idx_in_X_new]

plt.figure(figsize=(12, 8))
plt.suptitle(f"Sequence {top_anomaly_idx_in_X_new} (Largest RMSE) - Original vs. Reconstructed")
for i in range(num_channels):
    plt.subplot(num_channels, 1, i + 1)
    plt.plot(X_top_anomaly[:, i], label='Original')
    plt.plot(Y_top_anomaly_pred[:, i], '--', label='Reconstructed')
    plt.ylabel(f"Channel {i + 1}")
    plt.grid(True)
    if i == 0:
        plt.legend(loc='upper right')
    if i == num_channels - 1:
        plt.xlabel("Time Step")
plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to prevent suptitle overlap
plt.show()

# --- 5. Identify Anomalous Regions within a Sequence ---

# Select the top anomalous sequence for detailed region analysis
X_selected = X_top_anomaly
Y_selected_pred = Y_top_anomaly_pred

# Calculate element-wise RMSE (error per time step per channel)
# For MATLAB's rmse(Y,X,2), it calculates RMSE along dimension 2 (channels),
# resulting in (timesteps, 1) error.
# In NumPy, this means sqrt(mean(squared_diff, axis=1))
RMSE_per_timestep = np.sqrt(np.mean(np.square(X_selected - Y_selected_pred), axis=1))

# Set the time step window size and threshold
window_size = 7
# Threshold: 10% above the maximum error value identified using validation data
threshold = 1.5 * RMSEbaseline

# Identify anomalous windows
idx_anomaly = np.zeros(X_selected.shape[0], dtype=bool)
for t in range(X_selected.shape[0] - window_size + 1):
    idx_window = slice(t, t + window_size) # Python slice for the window
    # Check if all RMSE values within the window are above the threshold
    if np.all(RMSE_per_timestep[idx_window] > threshold):
        idx_anomaly[idx_window] = True

print(f"\nNumber of anomalous time steps detected: {np.sum(idx_anomaly)}")

# Display the sequence in a plot and highlight the anomalous regions
plt.figure(figsize=(12, 8))
plt.suptitle("Anomaly Detection - Highlighted Anomalous Regions")
for i in range(num_channels):
    plt.subplot(num_channels, 1, i + 1)
    plt.plot(X_selected[:, i], label='Input')
    
    # Create an array to plot only anomalous parts
    X_anomalous_channel = np.full_like(X_selected[:, i], np.nan)
    X_anomalous_channel[idx_anomaly] = X_selected[idx_anomaly, i]
    
    plt.plot(X_anomalous_channel, 'r', linewidth=3, label='Anomalous')
    
    plt.ylabel(f"Channel {i + 1}")
    plt.grid(True)
    if i == 0:
        plt.legend(loc='upper right')
    if i == num_channels - 1:
        plt.xlabel("Time Step")
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()

