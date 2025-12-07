import os
import pandas as pd
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
# import wandb
# from wandb.integration.keras import WandbMetricsLogger 
import joblib
import keras
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout, BatchNormalization, LayerNormalization, Add, Attention, Flatten


LSTM_n = True



early_stop = EarlyStopping(
    monitor='val_loss',      # what to watch
    patience=25,              # stop after 5 epochs with no improvement
    restore_best_weights=True
)

raw_neighboors = [
[0,1,2,3,4,5],
[0,1,2,3,5],
[0,1,2,4,5],
[0,1,3,4,5],
[0,2,3,4,5],
[0,1,2,3,4,5]
]

max_len = max(len(n) for n in raw_neighboors)
padded_neighboors = [n + [-1]*(max_len - len(n)) for n in raw_neighboors]
neighbors_tensor = tf.constant(padded_neighboors, dtype=tf.int32)

def masked_sparse_cce_with_neighbors(y_true, y_pred, penalty=5.0):
    """
    Sparse categorical cross-entropy with neighbor penalty.
    If predicted class is not in neighbors of the true class, apply a penalty.
    """
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
    y_true = tf.cast(y_true, tf.int32)

    batch_size = tf.shape(y_pred)[0]

    # Standard sparse categorical cross-entropy
    batch_indices = tf.range(batch_size)
    indices = tf.stack([batch_indices, y_true], axis=1)
    true_probs = tf.gather_nd(y_pred, indices)
    base_loss = -tf.math.log(true_probs)

    # Predicted classes
    y_pred_classes = tf.argmax(y_pred, axis=1, output_type=tf.int32)

    # Gather neighbors for each true label
    neighbors_of_true = tf.gather(neighbors_tensor, y_true)  # shape (batch_size, max_neighbors)

    # Create a mask to ignore padding (-1)
    mask_valid = neighbors_of_true >= 0
    match = tf.equal(y_pred_classes[:, tf.newaxis], neighbors_of_true)
    match_masked = tf.logical_and(match, mask_valid)

    # Apply penalty if prediction not in neighbors
    not_in_neighbors = tf.logical_not(tf.reduce_any(match_masked, axis=1))
    penalty_factor = tf.where(not_in_neighbors, penalty, 1.0)

    loss = base_loss * penalty_factor
    return tf.reduce_mean(loss)

def balance_classes(X, Y, Yclass):

    unique_classes, counts = np.unique(Yclass, return_counts=True)
    min_count = min(counts)

    X_balanced = []
    Y_balanced = []
    Yclass_balanced = []

    for cls in unique_classes:
        idx = np.where(Yclass == cls)[0]       # indices for this class
        np.random.shuffle(idx)                 # shuffle indices
        if cls == 0:
            selected_idx = idx[:2*min_count] 
        else:
            selected_idx = idx[:min_count]         # take only min_count samples
        X_balanced.append(X[selected_idx])
        Y_balanced.append(Y[selected_idx])
        Yclass_balanced.append(Yclass[selected_idx])

    # Concatenate back into arrays
    X_balanced = np.concatenate(X_balanced, axis=0)
    Y_balanced = np.concatenate(Y_balanced, axis=0)
    Yclass_balanced = np.concatenate(Yclass_balanced, axis=0)

    # Shuffle the balanced dataset
    perm = np.random.permutation(len(Yclass_balanced))
    X_balanced = X_balanced[perm]
    Y_balanced = Y_balanced[perm]
    Yclass_balanced = Yclass_balanced[perm]

    return X_balanced, Y_balanced, Yclass_balanced

def return_values(df):
    data = df.to_numpy()
    X, y, yclass = [], [], []
    for i in range(1,len(data)-1):
        if data[i,1] == -100:
            mean_value = 0.5 * (data[i-1,1]+data[i+1,1])
        else:
            mean_value =data[i,1]
        y.append(mean_value)

        if mean_value>threshhold:
            yclass.append(int(data[i,0]))
        else:
            yclass.append(int(0))
        
        R = data[i, 2:features+2]
        X.append(R)
    return np.array(X), np.array(y), np.array(yclass)

def feature_extraction(df, timewindow):
    data = df.to_numpy()
    R = data[:, 2:features+2]

    n_ch = R.shape[1]
    features_arr = np.zeros((len(data), 4 * n_ch))  # 5 features per channel

    for i in range(len(data) - timewindow):
        R_window = R[i:i+timewindow, :]

        mean = np.mean(R_window, axis=0)                
        rms = np.sqrt(np.mean(R_window**2, axis=0))      
        ptp = np.ptp(R_window, axis=0)                  
        peak = np.max(np.abs(R_window), axis=0)          
        crest_factor = peak / rms                       
        Ri = R[i+timewindow, :]                                    

        # Concatenate all feature arrays into a single row
        features_arr[i+timewindow, :] = np.hstack([rms, ptp, crest_factor, Ri])

    return features_arr

def create_sequences_tw(df, time_steps, timewindow):
    features_arr = np.array(feature_extraction(df, timewindow))
    data = df.to_numpy()
    X, y, yclass = [], [], []
    for i in range(time_steps+2,len(data)-1):
        if data[i,1] == -100:
            mean_value = 0.5 * (data[i-1,1]+data[i+1,1])
        else:
            mean_value =data[i,1]
        y.append(mean_value)

        if mean_value>threshhold:
            yclass.append(int(data[i,0]))
        else:
            yclass.append(int(0))
        

        X.append(features_arr[i-time_steps:i,:])

    return np.array(X), np.array(y), np.array(yclass)


def create_sequences(df, time_steps):
    data = df.to_numpy()
    X, y, yclass = [], [], []
    for i in range(time_steps+2,len(data)-1):
        if data[i,1] == -100:
            mean_value = 0.5 * (data[i-1,1]+data[i+1,1])
        else:
            mean_value =data[i,1]
        y.append(mean_value)

        if mean_value>threshhold:
            yclass.append(int(data[i,0]))
        else:
            yclass.append(int(0))
        
        R = data[i-time_steps:i, 2:features+2]
        DR = R - np.mean(R, axis=0)

        X.append(R)
        #X.append(np.concatenate([R, DR], axis=1))
    return np.array(X), np.array(y), np.array(yclass)

def training_inputs(folder_path):
    X, Y = None, None
    Yclass = None


    file_list = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_list.append(os.path.join(root, file))

    for file in file_list:
        print(file)
        clean_lines = []
        with open(file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if ">" in line:
                    _, values = line.split(">", 1)
                    values = values.replace("�", "").strip()
                    clean_lines.append(values)

        csv_data = "\n".join(clean_lines)

        df = pd.read_csv(StringIO(csv_data), header=None)
        df = df.drop(df.index[-1])
        
        
        if LSTM_n:
            # x, y, yclass = create_sequences(df, time_step)
            x, y, yclass = create_sequences_tw(df, time_step,timewindow)
        else:
            x, y, yclass = return_values(df)

        if X is None:
            X, Y, Yclass = x, y, yclass
        else:
            X = np.concatenate((X, x), axis=0)
            Y = np.concatenate((Y, y), axis=0)
            Yclass = np.concatenate((Yclass, yclass), axis=0)

    return X, Y, Yclass

folder_path = '/home/papaveneti/ros_ws/src/sensor_package/sensor_package/config/data'

all_dfs = []
time_step = 15
timewindow = 5
max_sense = 1000
min_sense = -20
Nepochs =400
Nx, Ny = 2, 3
features = Nx*Ny


threshhold= 50

if LSTM_n:
    save_dir = "/home/papaveneti/ros_ws/src/sensor_package/sensor_package/config/Models/LSTM"
else:
    save_dir = "/home/papaveneti/ros_ws/src/sensor_package/sensor_package/config/Models/MLP"
os.makedirs(save_dir, exist_ok=True)



X, Y, Yclass = training_inputs(folder_path)

n_classes = len(np.unique(Yclass))

X_train, X_test, Y_train, Y_test, y_train, y_test = train_test_split(
    X, Y, Yclass, test_size=0.3, random_state=42, shuffle=True
)

# Initialize scalers
x_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaler = MinMaxScaler()



# Transform both train and test

if LSTM_n:
    nsamples, timesteps, nfeatures = X_train.shape

    # Flatten to 2D for scaling
    X_train_2d = X_train.reshape(-1, nfeatures)
    X_test_2d  = X_test.reshape(-1, nfeatures)

    # Fit scalers
    x_scaler.fit(X_train_2d)
    y_scaler.fit(Y_train.reshape(-1, 1))

    # Transform and reshape back
    X_train_scaled = x_scaler.transform(X_train_2d).reshape(nsamples, timesteps, nfeatures)
    X_test_scaled  = x_scaler.transform(X_test_2d).reshape(X_test.shape[0], timesteps, nfeatures)

    y_train_scaled = y_scaler.transform(Y_train.reshape(-1, 1))
    y_test_scaled  = y_scaler.transform(Y_test.reshape(-1, 1))
else:
    nsamples, nfeatures = X_train.shape
    X_train_2d = X_train.reshape(-1, features)
    X_test_2d  = X_test.reshape(-1, features)
    x_scaler.fit(X_train_2d)
    y_scaler.fit(Y_train.reshape(-1, 1))
    # Transform both train and test
    X_train_scaled = x_scaler.transform(X_train_2d).reshape(nsamples, nfeatures)
    X_test_scaled  = x_scaler.transform(X_test_2d).reshape(X_test.shape[0], nfeatures)
    y_train_scaled = y_scaler.transform(Y_train.reshape(-1, 1))
    y_test_scaled  = y_scaler.transform(Y_test.reshape(-1, 1))

if not LSTM_n:
    model = Sequential([
        Input(shape=(features,)), 
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(n_classes, activation='softmax')
    ])
else:
    model = Sequential([
        Input(shape=(timesteps, nfeatures)),
        LSTM(64, return_sequences=True),  # LSTM layer
        Dropout(0.3),
        LSTM(32),                         # Second LSTM layer (outputs last timestep only)
        Dropout(0.3),
        Dense(32, activation='sigmoid'),
        Dropout(0.3),
        Dense(n_classes, activation='softmax')
    ])

model.summary()

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    #loss=masked_sparse_cce_with_neighbors,
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    metrics=['accuracy']
)

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=Nepochs,
    batch_size=64,
    verbose=1,
    callbacks=[early_stop]
)

# Evaluate on integer labels
loss, acc = model.evaluate(X_test_scaled, y_test)

# Predict probabilities (softmax outputs)
y_pred = model.predict(X_test_scaled)

# Convert probabilities → predicted class index
y_pred_classes = np.argmax(y_pred, axis=1)

# Ground truth is already integer labels
y_true = y_test

# Build confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(save_dir, "Position_confusionMatrix"))
plt.close()

print("SavingModel")
keras.saving.save_model(model, os.path.join(save_dir, "model_position.keras"))

print("SavingScalers")
joblib.dump(x_scaler, os.path.join(save_dir, "x_scaler_position.save"))