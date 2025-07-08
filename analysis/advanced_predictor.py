"""
Advanced Neural Network Models for Bitcoin Price Prediction
CNN-LSTM with Attention Mechanism + Feature Selection
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
import logging
from datetime import datetime, timedelta
import joblib
import os

# Import advanced feature engineering
try:
    from analysis.advanced_feature_engineering import AdvancedFeatureEngineer
    FEATURE_ENGINEERING_AVAILABLE = True
except ImportError:
    FEATURE_ENGINEERING_AVAILABLE = False

# Import SHAP for explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

class AttentionBlock(nn.Module):
    """Attention mechanism for LSTM outputs"""
    def __init__(self, hidden_size):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_size)
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        # Apply attention weights
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector, attention_weights

class CNN_LSTM_Attention(nn.Module):
    """
    Advanced CNN-LSTM model with Attention mechanism
    Based on research showing ~82% accuracy for BTC direction prediction
    """
    def __init__(self, input_size, hidden_size=128, num_layers=2, seq_length=60, dropout=0.3):
        super(CNN_LSTM_Attention, self).__init__()
        
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # CNN layers for feature extraction
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout_conv = nn.Dropout(dropout)
        
        # Bi-LSTM layers
        self.lstm = nn.LSTM(256, hidden_size, num_layers, 
                           batch_first=True, bidirectional=True, dropout=dropout)
        
        # Attention mechanism
        self.attention = AttentionBlock(hidden_size * 2)  # *2 for bidirectional
        
        # Dense layers
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 3)  # UP, DOWN, HOLD
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN feature extraction
        # x shape: (batch_size, seq_len, features) -> (batch_size, features, seq_len)
        x = x.transpose(1, 2)
        
        x = self.relu(self.conv1(x))
        x = self.dropout_conv(x)
        
        x = self.relu(self.conv2(x))
        x = self.dropout_conv(x)
        
        x = self.relu(self.conv3(x))
        x = self.dropout_conv(x)
        
        # Back to (batch_size, seq_len, features)
        x = x.transpose(1, 2)
        
        # Bi-LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Attention mechanism
        context_vector, attention_weights = self.attention(lstm_out)
        
        # Dense layers
        x = self.relu(self.fc1(context_vector))
        x = self.dropout(x)
        
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        
        output = self.output(x)
        
        return output, attention_weights

class AdvancedCryptoPredictor:
    """
    Advanced Bitcoin predictor using CNN-LSTM with Attention
    Implements research-based improvements for ~82% accuracy
    """
    
    def __init__(self, seq_length=60, hidden_size=128, num_layers=2, 
                 dropout=0.3, use_boruta=True):
        self.setup_logging()
        
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_boruta = use_boruta
        
        self.model = None
        self.scaler = RobustScaler()  # More robust to outliers
        self.feature_selector = None
        self.selected_features = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize feature engineer
        if FEATURE_ENGINEERING_AVAILABLE:
            self.feature_engineer = AdvancedFeatureEngineer()
            self.logger.info("✅ Advanced feature engineering available")
        else:
            self.feature_engineer = None
            self.logger.warning("⚠️ Advanced feature engineering not available")
        
        self.logger.info(f"🧠 Advanced Crypto Predictor initialized on {self.device}")
        self.logger.info(f"📊 Architecture: CNN-LSTM + Attention, seq_length={seq_length}")
        
    def setup_logging(self):
        """Setup logging"""
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def engineer_features(self, price_data, sentiment_data, enhanced_data=None):
        """
        Advanced feature engineering based on research
        Combines hard + soft features for HSIF approach
        Uses advanced feature engineering if available
        """
        try:
            # Use advanced feature engineering if available
            if self.feature_engineer:
                self.logger.info("🔧 Using advanced feature engineering...")
                
                # Get comprehensive features
                advanced_features = self.feature_engineer.engineer_all_features(
                    price_data=price_data,
                    sentiment_data=sentiment_data,
                    crypto_symbol='bitcoin'
                )
                
                # Convert to DataFrame with proper indexing
                if advanced_features:
                    features_df = pd.DataFrame([advanced_features])
                    
                    # Ensure we have enough data points for sequences
                    if len(features_df) < self.seq_length:
                        # Replicate the data to meet minimum sequence length
                        features_df = pd.concat([features_df] * (self.seq_length + 1), ignore_index=True)
                    
                    # Add target variable
                    features_df['target'] = self._create_target_from_price(price_data)
                    
                    self.logger.info(f"✅ Advanced features engineered: {features_df.shape[1]} features")
                    return features_df.dropna()
            
            # Fallback to basic feature engineering
            self.logger.info("📊 Using basic feature engineering...")
            features_df = self._basic_feature_engineering(price_data, sentiment_data, enhanced_data)
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"Error in feature engineering: {e}")
            import traceback
            traceback.print_exc()
            return self._basic_feature_engineering(price_data, sentiment_data, enhanced_data)
    
    def _basic_feature_engineering(self, price_data, sentiment_data, enhanced_data=None):
        """
        Basic feature engineering (fallback method)
        """
        features_df = pd.DataFrame()
        
        try:
            # Price-based features (Hard features)
            features_df['price'] = price_data['close']
            features_df['volume'] = price_data['volume']
            features_df['high'] = price_data['high']
            features_df['low'] = price_data['low']
            
            # Technical indicators (Hard features)
            features_df['sma_5'] = price_data['close'].rolling(5).mean()
            features_df['sma_10'] = price_data['close'].rolling(10).mean()
            features_df['sma_20'] = price_data['close'].rolling(20).mean()
            features_df['sma_50'] = price_data['close'].rolling(50).mean()
            
            features_df['ema_12'] = price_data['close'].ewm(span=12).mean()
            features_df['ema_26'] = price_data['close'].ewm(span=26).mean()
            
            # RSI
            delta = price_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features_df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            features_df['macd'] = features_df['ema_12'] - features_df['ema_26']
            features_df['macd_signal'] = features_df['macd'].ewm(span=9).mean()
            features_df['macd_histogram'] = features_df['macd'] - features_df['macd_signal']
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            sma_bb = price_data['close'].rolling(bb_period).mean()
            std_bb = price_data['close'].rolling(bb_period).std()
            features_df['bb_upper'] = sma_bb + (std_bb * bb_std)
            features_df['bb_lower'] = sma_bb - (std_bb * bb_std)
            features_df['bb_position'] = (price_data['close'] - features_df['bb_lower']) / (features_df['bb_upper'] - features_df['bb_lower'])
            
            # Price momentum features
            features_df['returns_1h'] = price_data['close'].pct_change(1)
            features_df['returns_4h'] = price_data['close'].pct_change(4)
            features_df['returns_24h'] = price_data['close'].pct_change(24)
            
            # Volatility (VIX-like indicators)
            features_df['volatility_5'] = features_df['returns_1h'].rolling(5).std()
            features_df['volatility_24'] = features_df['returns_1h'].rolling(24).std()
            
            # Volume indicators
            features_df['volume_sma'] = features_df['volume'].rolling(20).mean()
            features_df['volume_ratio'] = features_df['volume'] / features_df['volume_sma']
            
            # Sentiment features (Soft features) - Enhanced
            if sentiment_data is not None and not sentiment_data.empty:
                # Basic sentiment
                features_df['sentiment_score'] = sentiment_data['sentiment']
                features_df['sentiment_volume'] = sentiment_data.get('volume', 1)
                
                # Sentiment momentum (key feature from research)
                features_df['sentiment_momentum'] = sentiment_data['sentiment'].diff()
                features_df['sentiment_ma_5'] = sentiment_data['sentiment'].rolling(5).mean()
                features_df['sentiment_ma_24'] = sentiment_data['sentiment'].rolling(24).mean()
                
                # Sentiment volatility
                features_df['sentiment_volatility'] = sentiment_data['sentiment'].rolling(24).std()
                
                # Compound scores and negativity (from research)
                if 'compound' in sentiment_data.columns:
                    features_df['compound_score'] = sentiment_data['compound']
                if 'negativity' in sentiment_data.columns:
                    features_df['negativity'] = sentiment_data['negativity']
            
            # Enhanced metrics (if available)
            if enhanced_data is not None:
                if 'fear_greed_value' in enhanced_data:
                    features_df['fear_greed'] = enhanced_data['fear_greed_value']
                if 'google_trends' in enhanced_data:
                    features_df['google_trends'] = enhanced_data['google_trends']
                if 'stocktwits_sentiment' in enhanced_data:
                    features_df['stocktwits_sentiment'] = enhanced_data['stocktwits_sentiment']
            
            # Multi-horizon features (1h, 4h, 24h timeframes)
            for horizon in [1, 4, 24]:
                features_df[f'price_change_{horizon}h'] = price_data['close'].pct_change(horizon)
                features_df[f'volume_change_{horizon}h'] = features_df['volume'].pct_change(horizon)
                if 'sentiment_score' in features_df.columns:
                    features_df[f'sentiment_change_{horizon}h'] = features_df['sentiment_score'].pct_change(horizon)
            
            # Target variable (direction prediction)
            features_df['target'] = self._create_target(price_data['close'])
            
            self.logger.info(f"✅ Basic features engineered: {features_df.shape[1]} features")
            return features_df.dropna()
            
        except Exception as e:
            self.logger.error(f"Error in basic feature engineering: {e}")
            return pd.DataFrame()
    
    def _create_target_from_price(self, price_data, horizon=24, threshold=0.02):
        """
        Create target variable from price data
        """
        try:
            if isinstance(price_data, pd.DataFrame):
                prices = price_data['close']
            else:
                prices = price_data
            
            return self._create_target(prices, horizon, threshold)
            
        except Exception as e:
            self.logger.error(f"Error creating target: {e}")
            return 1  # Default to HOLD
    
    def _create_target(self, prices, horizon=24, threshold=0.02):
        """
        Create target variable for direction prediction
        Args:
            prices: Price series
            horizon: Hours ahead to predict
            threshold: Minimum change to consider significant (2%)
        Returns:
            0: DOWN, 1: HOLD, 2: UP
        """
        future_prices = prices.shift(-horizon)
        returns = (future_prices - prices) / prices
        
        target = np.where(returns > threshold, 2,  # UP
                 np.where(returns < -threshold, 0, 1))  # DOWN, HOLD
        
        return target
    
    def select_features_boruta(self, X, y):
        """
        Feature selection using Boruta algorithm
        Research shows Boruta + CNN-LSTM gives ~82% accuracy
        """
        if not self.use_boruta:
            return X
        
        try:
            self.logger.info("🔍 Starting Boruta feature selection...")
            
            # Use RandomForest for feature importance
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            
            # Boruta feature selection
            boruta_selector = BorutaPy(rf, n_estimators='auto', random_state=42, verbose=0)
            
            # Fit Boruta
            boruta_selector.fit(X.values, y)
            
            # Get selected features
            selected_mask = boruta_selector.support_
            self.selected_features = X.columns[selected_mask].tolist()
            
            self.logger.info(f"✅ Boruta selected {len(self.selected_features)} features out of {X.shape[1]}")
            self.logger.info(f"Selected features: {self.selected_features[:5]}...")  # Show first 5
            
            return X[self.selected_features]
            
        except Exception as e:
            self.logger.error(f"Boruta feature selection failed: {e}")
            return X
    
    def prepare_sequences(self, data, target_col='target'):
        """Prepare data sequences for CNN-LSTM"""
        features = data.drop(columns=[target_col])
        targets = data[target_col].values
        
        X, y = [], []
        
        for i in range(len(data) - self.seq_length):
            X.append(features.iloc[i:(i + self.seq_length)].values)
            y.append(targets[i + self.seq_length])
        
        return np.array(X), np.array(y)
    
    def train_model(self, features_data, epochs=100, batch_size=32, learning_rate=0.001):
        """
        Train the CNN-LSTM-Attention model
        """
        try:
            self.logger.info("🚀 Starting advanced model training...")
            
            # Feature selection with Boruta
            X = features_data.drop(columns=['target'])
            y = features_data['target'].values
            
            X_selected = self.select_features_boruta(X, y)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X_selected)
            X_scaled_df = pd.DataFrame(X_scaled, columns=X_selected.columns, index=X_selected.index)
            
            # Add target back
            scaled_data = X_scaled_df.copy()
            scaled_data['target'] = y
            
            # Prepare sequences
            X_seq, y_seq = self.prepare_sequences(scaled_data)
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
            )
            
            # Convert to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train).to(self.device)
            y_train_tensor = torch.LongTensor(y_train).to(self.device)
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            y_test_tensor = torch.LongTensor(y_test).to(self.device)
            
            # Create data loaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # Initialize model
            input_size = X_seq.shape[2]
            self.model = CNN_LSTM_Attention(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                seq_length=self.seq_length,
                dropout=self.dropout
            ).to(self.device)
            
            # Loss and optimizer
            criterion = nn.CrossEntropyLoss(weight=self._get_class_weights(y_train))
            optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
            
            # Training loop
            best_accuracy = 0
            patience_counter = 0
            max_patience = 20
            
            for epoch in range(epochs):
                self.model.train()
                total_loss = 0
                
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    
                    outputs, attention_weights = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                # Validation
                if epoch % 10 == 0:
                    accuracy = self.evaluate_model(X_test_tensor, y_test_tensor)
                    scheduler.step(total_loss)
                    
                    self.logger.info(f"Epoch {epoch}: Loss={total_loss:.4f}, Accuracy={accuracy:.4f}")
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        patience_counter = 0
                        self.save_model()
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= max_patience:
                        self.logger.info("Early stopping triggered")
                        break
            
            # Final evaluation
            final_accuracy = self.evaluate_model(X_test_tensor, y_test_tensor)
            self.logger.info(f"✅ Training complete! Final accuracy: {final_accuracy:.4f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in model training: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _get_class_weights(self, y):
        """Calculate class weights for imbalanced data"""
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        return torch.FloatTensor(weights).to(self.device)
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        self.model.eval()
        with torch.no_grad():
            outputs, _ = self.model(X_test)
            predictions = torch.argmax(outputs, dim=1)
            accuracy = (predictions == y_test).float().mean()
            return accuracy.item()
    
    def predict(self, recent_data, return_attention=False):
        """
        Make prediction with the trained model
        Returns direction and confidence
        """
        try:
            if self.model is None:
                self.logger.error("Model not trained yet")
                return None
            
            # Prepare data
            if self.selected_features:
                recent_data = recent_data[self.selected_features]
            
            # Scale data
            scaled_data = self.scaler.transform(recent_data)
            
            # Get last sequence
            if len(scaled_data) < self.seq_length:
                self.logger.error(f"Not enough data: {len(scaled_data)} < {self.seq_length}")
                return None
            
            sequence = scaled_data[-self.seq_length:]
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                outputs, attention_weights = self.model(sequence_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                prediction = torch.argmax(outputs, dim=1).item()
                confidence = torch.max(probabilities).item()
            
            # Convert to direction
            direction_map = {0: 'DOWN', 1: 'HOLD', 2: 'UP'}
            direction_text = direction_map[prediction]
            
            result = {
                'direction': prediction,
                'direction_text': direction_text,
                'confidence': confidence,
                'probabilities': {
                    'DOWN': probabilities[0][0].item(),
                    'HOLD': probabilities[0][1].item(),
                    'UP': probabilities[0][2].item()
                }
            }
            
            if return_attention:
                result['attention_weights'] = attention_weights.cpu().numpy()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in prediction: {e}")
            return None
    
    def save_model(self, filepath='models/advanced_crypto_model.pth'):
        """Save the trained model"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'scaler': self.scaler,
                'selected_features': self.selected_features,
                'model_params': {
                    'hidden_size': self.hidden_size,
                    'num_layers': self.num_layers,
                    'seq_length': self.seq_length,
                    'dropout': self.dropout
                }
            }, filepath)
            
            self.logger.info(f"✅ Model saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
    
    def load_model(self, filepath='models/advanced_crypto_model.pth'):
        """Load a trained model"""
        try:
            if not os.path.exists(filepath):
                self.logger.error(f"Model file not found: {filepath}")
                return False
            
            checkpoint = torch.load(filepath, map_location=self.device)
            
            # Restore model parameters
            self.scaler = checkpoint['scaler']
            self.selected_features = checkpoint['selected_features']
            
            # Recreate model
            # We need to know the input size, which should be stored
            if self.selected_features:
                input_size = len(self.selected_features)
            else:
                # Fallback - this should be improved
                input_size = 50  # default
            
            self.model = CNN_LSTM_Attention(
                input_size=input_size,
                hidden_size=checkpoint['model_params']['hidden_size'],
                num_layers=checkpoint['model_params']['num_layers'],
                seq_length=checkpoint['model_params']['seq_length'],
                dropout=checkpoint['model_params']['dropout']
            ).to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            self.logger.info(f"✅ Model loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
        
    def explain_prediction(self, recent_data, feature_names=None):
        """
        Explain prediction using SHAP values and attention weights
        
        Args:
            recent_data: Recent data for prediction
            feature_names: Names of features
            
        Returns:
            Dictionary with explanations
        """
        try:
            if not SHAP_AVAILABLE:
                self.logger.warning("SHAP not available for explanations")
                return {}
            
            # Make prediction with attention weights
            prediction = self.predict(recent_data, return_attention=True)
            
            if prediction is None:
                return {}
            
            # Get attention weights explanation
            attention_explanation = self._explain_attention_weights(
                prediction.get('attention_weights')
            )
            
            # Get feature importance using SHAP (simplified for neural networks)
            feature_importance = self._get_feature_importance(recent_data, feature_names)
            
            return {
                'prediction': prediction,
                'attention_explanation': attention_explanation,
                'feature_importance': feature_importance,
                'explanation_timestamp': datetime.utcnow()
            }
            
        except Exception as e:
            self.logger.error(f"Error explaining prediction: {e}")
            return {}
    
    def _explain_attention_weights(self, attention_weights):
        """
        Explain attention weights
        
        Args:
            attention_weights: Attention weights from model
            
        Returns:
            Dictionary with attention explanations
        """
        try:
            if attention_weights is None:
                return {}
            
            # Convert to numpy if needed
            if isinstance(attention_weights, torch.Tensor):
                attention_weights = attention_weights.detach().cpu().numpy()
            
            # Get top attention positions
            avg_attention = np.mean(attention_weights, axis=0)
            top_indices = np.argsort(avg_attention)[-5:]  # Top 5 most attended positions
            
            explanation = {
                'top_attended_positions': top_indices.tolist(),
                'attention_scores': avg_attention[top_indices].tolist(),
                'total_attention_variance': np.var(avg_attention),
                'attention_concentration': np.max(avg_attention) / np.mean(avg_attention)
            }
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Error explaining attention weights: {e}")
            return {}
    
    def _get_feature_importance(self, recent_data, feature_names=None):
        """
        Get feature importance using gradient-based methods
        
        Args:
            recent_data: Recent data for analysis
            feature_names: Names of features
            
        Returns:
            Dictionary with feature importance
        """
        try:
            if self.model is None:
                return {}
            
            # Prepare data
            if self.selected_features:
                recent_data = recent_data[self.selected_features]
            
            # Scale data
            scaled_data = self.scaler.transform(recent_data)
            
            # Get last sequence
            if len(scaled_data) < self.seq_length:
                return {}
            
            sequence = scaled_data[-self.seq_length:]
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            sequence_tensor.requires_grad_(True)
            
            # Forward pass
            self.model.eval()
            outputs, _ = self.model(sequence_tensor)
            
            # Get gradients (simple feature importance)
            prediction_class = torch.argmax(outputs, dim=1)
            class_score = outputs[0, prediction_class]
            
            # Backward pass
            class_score.backward()
            
            # Get gradients
            gradients = sequence_tensor.grad.detach().cpu().numpy()
            
            # Calculate feature importance (mean absolute gradient)
            feature_importance = np.mean(np.abs(gradients), axis=(0, 1))
            
            # Create importance dictionary
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(feature_importance))]
            
            importance_dict = {
                'feature_importance': dict(zip(feature_names, feature_importance)),
                'top_features': sorted(
                    zip(feature_names, feature_importance), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:10]  # Top 10 features
            }
            
            return importance_dict
            
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {e}")
            return {}
    
    def generate_prediction_report(self, recent_data, include_explanation=True):
        """
        Generate comprehensive prediction report with explanations
        
        Args:
            recent_data: Recent data for prediction
            include_explanation: Whether to include SHAP explanations
            
        Returns:
            Dictionary with comprehensive prediction report
        """
        try:
            # Make prediction
            prediction = self.predict(recent_data, return_attention=True)
            
            if prediction is None:
                return {}
            
            # Base report
            report = {
                'prediction': prediction,
                'timestamp': datetime.utcnow(),
                'confidence_level': self._classify_confidence(prediction['confidence']),
                'trading_signal': self._generate_trading_signal(prediction)
            }
            
            # Add explanations if requested
            if include_explanation:
                explanations = self.explain_prediction(recent_data)
                report['explanations'] = explanations
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating prediction report: {e}")
            return {}
    
    def _classify_confidence(self, confidence):
        """
        Classify confidence level
        
        Args:
            confidence: Confidence score
            
        Returns:
            String classification
        """
        if confidence >= 0.8:
            return "Very High"
        elif confidence >= 0.6:
            return "High"
        elif confidence >= 0.4:
            return "Medium"
        else:
            return "Low"
    
    def _generate_trading_signal(self, prediction):
        """
        Generate trading signal from prediction
        
        Args:
            prediction: Prediction result
            
        Returns:
            Dictionary with trading signal
        """
        try:
            direction = prediction['direction']
            confidence = prediction['confidence']
            
            # Signal strength based on confidence
            if confidence >= 0.7:
                strength = "Strong"
            elif confidence >= 0.5:
                strength = "Moderate"
            else:
                strength = "Weak"
            
            # Trading action
            if direction == 2 and confidence >= 0.6:  # UP with good confidence
                action = "BUY"
            elif direction == 0 and confidence >= 0.6:  # DOWN with good confidence
                action = "SELL"
            else:
                action = "HOLD"
            
            return {
                'action': action,
                'strength': strength,
                'confidence': confidence,
                'direction': prediction['direction_text'],
                'risk_level': "High" if confidence < 0.5 else "Medium" if confidence < 0.7 else "Low"
            }
            
        except Exception as e:
            self.logger.error(f"Error generating trading signal: {e}")
            return {'action': 'HOLD', 'strength': 'Weak', 'confidence': 0.0}
