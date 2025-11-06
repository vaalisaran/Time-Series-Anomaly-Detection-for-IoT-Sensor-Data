

import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Install with: pip install torch")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Autoencoder(nn.Module):
    
    
    def __init__(self, input_dim, encoding_dim=32):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class LSTMAutoencoder(nn.Module):
    
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(LSTMAutoencoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=0.2
        )
        
        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(
            hidden_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=0.2
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        # Encoder
        _, (hidden, cell) = self.encoder_lstm(x)
        
        # Repeat hidden state for decoder
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        
        # Decoder
        decoder_output, _ = self.decoder_lstm(decoder_input, (hidden, cell))
        
        # Output
        output = self.output_layer(decoder_output)
        
        return output


class DeepLearningAnomalyDetector:
    
    def __init__(self, model_type='autoencoder', input_dim=10, encoding_dim=32,
                 hidden_dim=64, num_layers=2, device=None):
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")
        
        self.model_type = model_type
        self.input_dim = input_dim
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_type == 'autoencoder':
            self.model = Autoencoder(input_dim, encoding_dim)
        elif model_type == 'lstm_autoencoder':
            self.model = LSTMAutoencoder(input_dim, hidden_dim, num_layers)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model.to(self.device)
        self.threshold = None
        
        logger.info(f"Initialized {model_type} on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
    
    def prepare_data(self, X, sequence_length=10):
        
        if self.model_type == 'autoencoder':
            X_tensor = torch.FloatTensor(X)
            dataset = TensorDataset(X_tensor, X_tensor)
        elif self.model_type == 'lstm_autoencoder':
            # Create sequences
            sequences = []
            for i in range(len(X) - sequence_length + 1):
                sequences.append(X[i:i+sequence_length])
            
            X_tensor = torch.FloatTensor(np.array(sequences))
            dataset = TensorDataset(X_tensor, X_tensor)
        
        return dataset
    
    def train(self, X, epochs=50, batch_size=32, learning_rate=0.001, 
              validation_split=0.2, sequence_length=10):
        
        logger.info(f"Training {self.model_type}...")
        logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")
        
        # Prepare data
        dataset = self.prepare_data(X, sequence_length)
        
        # Split into train and validation
        train_size = int((1 - validation_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        logger.info("Training completed")
        return train_losses, val_losses
    
    def compute_reconstruction_error(self, X, sequence_length=10):
        
        self.model.eval()
        
        dataset = self.prepare_data(X, sequence_length)
        loader = DataLoader(dataset, batch_size=32)
        
        errors = []
        
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                batch_errors = torch.mean((outputs - batch_y) ** 2, dim=tuple(range(1, len(outputs.shape))))
                errors.extend(batch_errors.cpu().numpy())
        
        return np.array(errors)
    
    def set_threshold(self, X, percentile=95, sequence_length=10):
        
        errors = self.compute_reconstruction_error(X, sequence_length)
        self.threshold = np.percentile(errors, percentile)
        logger.info(f"Threshold set to {self.threshold:.6f} ({percentile}th percentile)")
        return self.threshold
    
    def predict(self, X, sequence_length=10):
        
        if self.threshold is None:
            raise ValueError("Threshold not set. Call set_threshold() first.")
        
        errors = self.compute_reconstruction_error(X, sequence_length)
        predictions = np.where(errors > self.threshold, -1, 1)
        
        return predictions
    
    def get_anomaly_scores(self, X, sequence_length=10):
        
        errors = self.compute_reconstruction_error(X, sequence_length)
        # Negate so lower scores indicate anomalies (consistent with other methods)
        return -errors


if __name__ == "__main__":
    # Example usage
    if TORCH_AVAILABLE:
        X = np.random.randn(1000, 10)
        
        detector = DeepLearningAnomalyDetector(
            model_type='autoencoder',
            input_dim=10,
            encoding_dim=5
        )
        
        train_losses, val_losses = detector.train(X, epochs=20)
        detector.set_threshold(X, percentile=95)
        predictions = detector.predict(X)
        
        print(f"Detected {np.sum(predictions == -1)} anomalies")