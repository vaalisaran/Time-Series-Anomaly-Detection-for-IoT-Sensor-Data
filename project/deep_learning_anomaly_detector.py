# Implement deep learning based anomaly detection (Autoencoder & LSTM)

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
import pickle
import warnings
warnings.filterwarnings('ignore')

class DeepLearningAnomalyDetector:
    def __init__(self, contamination=0.05):
       
        self.contamination = contamination
        self.models = {}
        self.predictions = {}
        self.reconstruction_errors = {}
        self.thresholds = {}
        
    def build_autoencoder(self, input_dim, encoding_dim=32, hidden_layers=[64, 32]):
       
        print("\n" + "="*60)
        print("BUILDING AUTOENCODER")
        print("="*60)
        print(f"Input dimension: {input_dim}")
        print(f"Encoding dimension: {encoding_dim}")
        print(f"Hidden layers: {hidden_layers}")
        
        # Encoder
        encoder_input = layers.Input(shape=(input_dim,))
        x = encoder_input
        
        for units in hidden_layers:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
            
        encoded = layers.Dense(encoding_dim, activation='relu', name='encoding')(x)
        
        # Decoder
        x = encoded
        for units in reversed(hidden_layers):
            x = layers.Dense(units, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
            
        decoded = layers.Dense(input_dim, activation='linear')(x)
        
        # Autoencoder model
        autoencoder = Model(encoder_input, decoded, name='autoencoder')
        
        # Compile
        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print(f"Total parameters: {autoencoder.count_params():,}")
        print("="*60)
        
        return autoencoder
    
    def train_autoencoder(self, X_train, X_val, epochs=50, batch_size=128):
        
        print("\nTraining Autoencoder...")
        
        # Build model
        autoencoder = self.build_autoencoder(
            input_dim=X_train.shape[1],
            encoding_dim=32,
            hidden_layers=[128, 64, 32]
        )
        
        # Callbacks
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # Train
        history = autoencoder.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, X_val),
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        self.models['autoencoder'] = autoencoder
        
        return autoencoder, history
    
    def predict_autoencoder(self, X, model, percentile=95):
        
        print("\n" + "="*60)
        print("AUTOENCODER PREDICTION")
        print("="*60)
        
        # Get reconstructions
        reconstructions = model.predict(X, verbose=0)
        
        # Calculate reconstruction error (MSE per sample)
        errors = np.mean(np.square(X - reconstructions), axis=1)
        
        # Determine threshold based on percentile
        threshold = np.percentile(errors, percentile)
        self.thresholds['autoencoder'] = threshold
        
        # Predict anomalies
        predictions = (errors > threshold).astype(int)
        
        self.predictions['autoencoder'] = predictions
        self.reconstruction_errors['autoencoder'] = errors
        
        n_anomalies = predictions.sum()
        print(f"Threshold (MSE): {threshold:.6f}")
        print(f"Detected {n_anomalies} anomalies ({n_anomalies/len(predictions)*100:.2f}%)")
        print("="*60)
        
        return predictions, errors
    
    def build_lstm_autoencoder(self, timesteps, n_features, encoding_dim=32):
        
        print("\n" + "="*60)
        print("BUILDING LSTM AUTOENCODER")
        print("="*60)
        print(f"Timesteps: {timesteps}")
        print(f"Features per timestep: {n_features}")
        print(f"Encoding dimension: {encoding_dim}")
        
        # Encoder
        encoder_inputs = layers.Input(shape=(timesteps, n_features))
        
        # LSTM layers
        x = layers.LSTM(128, return_sequences=True)(encoder_inputs)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(64, return_sequences=True)(x)
        x = layers.Dropout(0.2)(x)
        encoded = layers.LSTM(encoding_dim, return_sequences=False, name='encoding')(x)
        
        # Decoder
        x = layers.RepeatVector(timesteps)(encoded)
        x = layers.LSTM(encoding_dim, return_sequences=True)(x)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(64, return_sequences=True)(x)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(128, return_sequences=True)(x)
        decoded = layers.TimeDistributed(layers.Dense(n_features))(x)
        
        # Model
        lstm_autoencoder = Model(encoder_inputs, decoded, name='lstm_autoencoder')
        
        # Compile
        lstm_autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print(f"Total parameters: {lstm_autoencoder.count_params():,}")
        print("="*60)
        
        return lstm_autoencoder
    
    def create_sequences(self, X, timesteps=10):
        
        sequences = []
        for i in range(len(X) - timesteps + 1):
            sequences.append(X[i:i+timesteps])
        return np.array(sequences)
    
    def train_lstm_autoencoder(self, X_train, X_val, timesteps=10, epochs=30, batch_size=64):
        
        print("\nTraining LSTM Autoencoder...")
        
        # Create sequences
        X_train_seq = self.create_sequences(X_train, timesteps)
        X_val_seq = self.create_sequences(X_val, timesteps)
        
        print(f"Training sequences shape: {X_train_seq.shape}")
        print(f"Validation sequences shape: {X_val_seq.shape}")
        
        # Build model
        lstm_autoencoder = self.build_lstm_autoencoder(
            timesteps=timesteps,
            n_features=X_train.shape[1],
            encoding_dim=32
        )
        
        # Callbacks
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-6
        )
        
        # Train
        history = lstm_autoencoder.fit(
            X_train_seq, X_train_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val_seq, X_val_seq),
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        self.models['lstm_autoencoder'] = lstm_autoencoder
        
        return lstm_autoencoder, history
    
    def predict_lstm_autoencoder(self, X, model, timesteps=10, percentile=95):
    
        print("\n" + "="*60)
        print("LSTM AUTOENCODER PREDICTION")
        print("="*60)
        
        # Create sequences
        X_seq = self.create_sequences(X, timesteps)
        
        # Get reconstructions
        reconstructions = model.predict(X_seq, verbose=0)
        
        # Calculate reconstruction error per sequence
        errors = np.mean(np.square(X_seq - reconstructions), axis=(1, 2))
        
        # Pad errors to match original length
        errors_full = np.zeros(len(X))
        errors_full[timesteps-1:] = errors
        errors_full[:timesteps-1] = errors[0]  # Fill initial values
        
        # Determine threshold
        threshold = np.percentile(errors, percentile)
        self.thresholds['lstm_autoencoder'] = threshold
        
        # Predict anomalies
        predictions_full = (errors_full > threshold).astype(int)
        
        self.predictions['lstm_autoencoder'] = predictions_full
        self.reconstruction_errors['lstm_autoencoder'] = errors_full
        
        n_anomalies = predictions_full.sum()
        print(f"Threshold (MSE): {threshold:.6f}")
        print(f"Detected {n_anomalies} anomalies ({n_anomalies/len(predictions_full)*100:.2f}%)")
        print("="*60)
        
        return predictions_full, errors_full
    
    def save_models(self, path_prefix='models/'):
        """Save trained models"""
        import os
        os.makedirs('models', exist_ok=True)
        
        if 'autoencoder' in self.models:
            self.models['autoencoder'].save(f'{path_prefix}autoencoder.h5')
            print(f"Autoencoder saved to {path_prefix}autoencoder.h5")
            
        if 'lstm_autoencoder' in self.models:
            self.models['lstm_autoencoder'].save(f'{path_prefix}lstm_autoencoder.h5')
            print(f"LSTM Autoencoder saved to {path_prefix}lstm_autoencoder.h5")
            
        # Save thresholds
        with open(f'{path_prefix}thresholds.pkl', 'wb') as f:
            pickle.dump(self.thresholds, f)
        print(f"Thresholds saved to {path_prefix}thresholds.pkl")

if __name__ == "__main__":
    # Load featured data
    df = pd.read_csv('sensor_data_features.csv')
    
    # Prepare features
    exclude_cols = ['timestamp', 'is_anomaly']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].values
    
    # Split data
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
    
    # Initialize detector
    detector = DeepLearningAnomalyDetector(contamination=0.05)
    
    # Train and predict with Autoencoder
    autoencoder, ae_history = detector.train_autoencoder(X_train, X_val, epochs=50, batch_size=128)
    ae_pred, ae_errors = detector.predict_autoencoder(X, autoencoder, percentile=95)
    
    # Train and predict with LSTM Autoencoder
    lstm_ae, lstm_history = detector.train_lstm_autoencoder(X_train, X_val, timesteps=10, epochs=30)
    lstm_pred, lstm_errors = detector.predict_lstm_autoencoder(X, lstm_ae, timesteps=10, percentile=95)
    
    # Save models
    detector.save_models()
    
    # Add predictions to dataframe
    df['pred_autoencoder'] = ae_pred
    df['pred_lstm_autoencoder'] = lstm_pred
    df['error_autoencoder'] = ae_errors
    df['error_lstm_autoencoder'] = lstm_errors
    
    # Save results
    df.to_csv('results_deep_learning.csv', index=False)
    print("\nResults saved to 'results_deep_learning.csv'")