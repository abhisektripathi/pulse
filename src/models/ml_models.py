"""
Advanced ML Models for Predictive System Health Platform

This module contains sophisticated machine learning models for:
- Anomaly Detection (Isolation Forest, LOF, Autoencoder)
- Failure Prediction (Random Forest, XGBoost, LSTM)
- Performance Forecasting (Prophet, ARIMA, Neural Networks)
- Resource Prediction (Linear Regression, SVR, Gradient Boosting)
- Cascade Risk Assessment (Graph Neural Networks, Bayesian Networks)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import joblib
import pickle
import logging

# Traditional ML
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor, 
    IsolationForest, GradientBoostingClassifier,
    GradientBoostingRegressor, ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR, OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Time Series
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Advanced ML
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not available. Install with: pip install lightgbm")

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        Dense, LSTM, GRU, Dropout, BatchNormalization,
        Conv1D, MaxPooling1D, Flatten, Input, Concatenate
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. Install with: pip install tensorflow")

# Bayesian
try:
    from pymc import Model as PyMCModel
    from pymc import Normal, Bernoulli, Beta, sample, traceplot
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    logging.warning("PyMC not available. Install with: pip install pymc")

logger = logging.getLogger(__name__)


class BaseMLModel:
    """Base class for all ML models."""
    
    def __init__(self, model_name: str, model_type: str):
        self.model_name = model_name
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.training_history = {}
        self.feature_names = []
        self.model_metadata = {}
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Train the model."""
        raise NotImplementedError
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        raise NotImplementedError
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities (for classification models)."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        raise NotImplementedError
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        raise NotImplementedError
        
    def save_model(self, filepath: str):
        """Save model to disk."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'training_history': self.training_history,
            'feature_names': self.feature_names,
            'model_metadata': self.model_metadata
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
        
    def load_model(self, filepath: str):
        """Load model from disk."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_trained = model_data['is_trained']
        self.training_history = model_data['training_history']
        self.feature_names = model_data['feature_names']
        self.model_metadata = model_data['model_metadata']
        logger.info(f"Model loaded from {filepath}")


class AnomalyDetectionModel(BaseMLModel):
    """Advanced anomaly detection using multiple algorithms."""
    
    def __init__(self, algorithm: str = "isolation_forest", **kwargs):
        super().__init__("anomaly_detector", "anomaly_detection")
        self.algorithm = algorithm
        self.contamination = kwargs.get('contamination', 0.1)
        self._initialize_model(**kwargs)
        
    def _initialize_model(self, **kwargs):
        """Initialize the anomaly detection model."""
        if self.algorithm == "isolation_forest":
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=kwargs.get('random_state', 42),
                n_estimators=kwargs.get('n_estimators', 100)
            )
        elif self.algorithm == "local_outlier_factor":
            self.model = LocalOutlierFactor(
                contamination=self.contamination,
                n_neighbors=kwargs.get('n_neighbors', 20),
                novelty=True
            )
        elif self.algorithm == "one_class_svm":
            self.model = OneClassSVM(
                kernel=kwargs.get('kernel', 'rbf'),
                nu=self.contamination,
                gamma=kwargs.get('gamma', 'scale')
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
            
        self.scaler = StandardScaler()
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs):
        """Train the anomaly detection model."""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        if self.algorithm == "local_outlier_factor":
            self.model.fit(X_scaled)
        else:
            self.model.fit(X_scaled)
            
        self.is_trained = True
        logger.info(f"Anomaly detection model trained with {X.shape[0]} samples")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies (1 for normal, -1 for anomaly)."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly scores."""
        X_scaled = self.scaler.transform(X)
        if hasattr(self.model, 'decision_function'):
            scores = self.model.decision_function(X_scaled)
            # Convert to probability-like scores (0-1, higher = more anomalous)
            scores = 1 - (scores - scores.min()) / (scores.max() - scores.min())
            return scores.reshape(-1, 1)
        else:
            # For models without decision_function, use predict
            predictions = self.predict(X)
            return (predictions == -1).astype(float).reshape(-1, 1)
            
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate anomaly detection performance."""
        predictions = self.predict(X)
        scores = self.predict_proba(X).flatten()
        
        # Convert labels: 1 = normal, 0 = anomaly
        y_binary = (y == 1).astype(int)
        pred_binary = (predictions == 1).astype(int)
        
        return {
            'accuracy': accuracy_score(y_binary, pred_binary),
            'precision': precision_score(y_binary, pred_binary, zero_division=0),
            'recall': recall_score(y_binary, pred_binary, zero_division=0),
            'f1_score': f1_score(y_binary, pred_binary, zero_division=0),
            'auc': roc_auc_score(y_binary, scores) if len(np.unique(y_binary)) > 1 else 0.5
        }


class FailurePredictionModel(BaseMLModel):
    """Advanced failure prediction using ensemble methods."""
    
    def __init__(self, algorithm: str = "random_forest", **kwargs):
        super().__init__("failure_predictor", "failure_prediction")
        self.algorithm = algorithm
        self._initialize_model(**kwargs)
        
    def _initialize_model(self, **kwargs):
        """Initialize the failure prediction model."""
        if self.algorithm == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', None),
                random_state=kwargs.get('random_state', 42),
                class_weight=kwargs.get('class_weight', 'balanced')
            )
        elif self.algorithm == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 3),
                random_state=kwargs.get('random_state', 42)
            )
        elif self.algorithm == "xgboost" and XGBOOST_AVAILABLE:
            self.model = xgb.XGBClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 3),
                random_state=kwargs.get('random_state', 42),
                eval_metric='logloss'
            )
        elif self.algorithm == "lightgbm" and LIGHTGBM_AVAILABLE:
            self.model = lgb.LGBMClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 3),
                random_state=kwargs.get('random_state', 42),
                class_weight='balanced'
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
            
        self.scaler = StandardScaler()
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Train the failure prediction model."""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        # Store feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.model_metadata['feature_importance'] = dict(
                zip(self.feature_names, self.model.feature_importances_)
            )
            
        self.is_trained = True
        logger.info(f"Failure prediction model trained with {X.shape[0]} samples")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict failure (0 = no failure, 1 = failure)."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict failure probabilities."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate failure prediction performance."""
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)[:, 1]  # Probability of failure
        
        return {
            'accuracy': accuracy_score(y, predictions),
            'precision': precision_score(y, predictions, zero_division=0),
            'recall': recall_score(y, predictions, zero_division=0),
            'f1_score': f1_score(y, predictions, zero_division=0),
            'auc': roc_auc_score(y, probabilities) if len(np.unique(y)) > 1 else 0.5
        }


class PerformancePredictionModel(BaseMLModel):
    """Performance prediction using time series and regression models."""
    
    def __init__(self, algorithm: str = "random_forest", **kwargs):
        super().__init__("performance_predictor", "performance_prediction")
        self.algorithm = algorithm
        self._initialize_model(**kwargs)
        
    def _initialize_model(self, **kwargs):
        """Initialize the performance prediction model."""
        if self.algorithm == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', None),
                random_state=kwargs.get('random_state', 42)
            )
        elif self.algorithm == "gradient_boosting":
            self.model = GradientBoostingRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 3),
                random_state=kwargs.get('random_state', 42)
            )
        elif self.algorithm == "xgboost" and XGBOOST_AVAILABLE:
            self.model = xgb.XGBRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 3),
                random_state=kwargs.get('random_state', 42)
            )
        elif self.algorithm == "svr":
            self.model = SVR(
                kernel=kwargs.get('kernel', 'rbf'),
                C=kwargs.get('C', 1.0),
                gamma=kwargs.get('gamma', 'scale')
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
            
        self.scaler = StandardScaler()
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Train the performance prediction model."""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        # Store feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.model_metadata['feature_importance'] = dict(
                zip(self.feature_names, self.model.feature_importances_)
            )
            
        self.is_trained = True
        logger.info(f"Performance prediction model trained with {X.shape[0]} samples")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict performance scores."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate performance prediction."""
        predictions = self.predict(X)
        
        return {
            'mse': mean_squared_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'mae': mean_absolute_error(y, predictions),
            'r2_score': r2_score(y, predictions)
        }


class TimeSeriesModel(BaseMLModel):
    """Time series forecasting for system metrics."""
    
    def __init__(self, algorithm: str = "arima", **kwargs):
        super().__init__("time_series_predictor", "time_series")
        self.algorithm = algorithm
        self.order = kwargs.get('order', (1, 1, 1))
        self.seasonal_order = kwargs.get('seasonal_order', (1, 1, 1, 12))
        self._initialize_model(**kwargs)
        
    def _initialize_model(self, **kwargs):
        """Initialize the time series model."""
        self.model = None  # Will be initialized during fit
        self.scaler = StandardScaler()
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Train the time series model."""
        # For time series, X should be timestamps and y should be values
        if self.algorithm == "arima":
            self.model = ARIMA(y, order=self.order)
            self.model = self.model.fit()
        elif self.algorithm == "sarima":
            self.model = SARIMAX(y, order=self.order, seasonal_order=self.seasonal_order)
            self.model = self.model.fit(disp=False)
        elif self.algorithm == "exponential_smoothing":
            self.model = ExponentialSmoothing(
                y, 
                trend=kwargs.get('trend', 'add'),
                seasonal=kwargs.get('seasonal', 'add'),
                seasonal_periods=kwargs.get('seasonal_periods', 12)
            ).fit()
            
        self.is_trained = True
        logger.info(f"Time series model trained with {len(y)} observations")
        
    def predict(self, X: np.ndarray, steps: int = 1) -> np.ndarray:
        """Predict future values."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        if self.algorithm in ["arima", "sarima"]:
            return self.model.forecast(steps=steps)
        elif self.algorithm == "exponential_smoothing":
            return self.model.forecast(steps=steps)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
            
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate time series model performance."""
        predictions = self.predict(X, steps=len(y))
        
        return {
            'mse': mean_squared_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'mae': mean_absolute_error(y, predictions),
            'mape': np.mean(np.abs((y - predictions) / y)) * 100
        }


class DeepLearningModel(BaseMLModel):
    """Deep learning models for complex pattern recognition."""
    
    def __init__(self, model_type: str = "lstm", **kwargs):
        super().__init__("deep_learning_predictor", "deep_learning")
        self.model_type = model_type
        self.sequence_length = kwargs.get('sequence_length', 10)
        self._initialize_model(**kwargs)
        
    def _initialize_model(self, **kwargs):
        """Initialize the deep learning model."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for deep learning models")
            
        self.model = None  # Will be built during fit
        self.scaler = MinMaxScaler()
        
    def _build_lstm_model(self, input_shape: Tuple[int, int], output_size: int = 1):
        """Build LSTM model for time series prediction."""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(output_size)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
        
    def _build_autoencoder(self, input_shape: int):
        """Build autoencoder for anomaly detection."""
        # Encoder
        input_layer = Input(shape=(input_shape,))
        encoded = Dense(64, activation='relu')(input_layer)
        encoded = Dense(32, activation='relu')(encoded)
        encoded = Dense(16, activation='relu')(encoded)
        
        # Decoder
        decoded = Dense(32, activation='relu')(encoded)
        decoded = Dense(64, activation='relu')(decoded)
        decoded = Dense(input_shape, activation='sigmoid')(decoded)
        
        model = Model(input_layer, decoded)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse'
        )
        
        return model
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Train the deep learning model."""
        if self.model_type == "lstm":
            # Reshape data for LSTM (samples, timesteps, features)
            X_reshaped = X.reshape((X.shape[0], self.sequence_length, -1))
            self.model = self._build_lstm_model(
                input_shape=(self.sequence_length, X_reshaped.shape[2]),
                output_size=1
            )
            
            # Train model
            history = self.model.fit(
                X_reshaped, y,
                epochs=kwargs.get('epochs', 50),
                batch_size=kwargs.get('batch_size', 32),
                validation_split=0.2,
                callbacks=[
                    EarlyStopping(patience=10, restore_best_weights=True),
                    ReduceLROnPlateau(patience=5, factor=0.5)
                ],
                verbose=kwargs.get('verbose', 1)
            )
            
            self.training_history = history.history
            
        elif self.model_type == "autoencoder":
            self.model = self._build_autoencoder(X.shape[1])
            
            # Train autoencoder
            history = self.model.fit(
                X, X,  # Autoencoder learns to reconstruct input
                epochs=kwargs.get('epochs', 50),
                batch_size=kwargs.get('batch_size', 32),
                validation_split=0.2,
                callbacks=[
                    EarlyStopping(patience=10, restore_best_weights=True)
                ],
                verbose=kwargs.get('verbose', 1)
            )
            
            self.training_history = history.history
            
        self.is_trained = True
        logger.info(f"Deep learning model trained with {X.shape[0]} samples")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with deep learning model."""
        if self.model_type == "lstm":
            X_reshaped = X.reshape((X.shape[0], self.sequence_length, -1))
            return self.model.predict(X_reshaped)
        elif self.model_type == "autoencoder":
            return self.model.predict(X)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate deep learning model performance."""
        predictions = self.predict(X)
        
        return {
            'mse': mean_squared_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'mae': mean_absolute_error(y, predictions)
        }


class EnsembleModel(BaseMLModel):
    """Ensemble model combining multiple algorithms."""
    
    def __init__(self, models: List[BaseMLModel], weights: Optional[List[float]] = None):
        super().__init__("ensemble_predictor", "ensemble")
        self.models = models
        self.weights = weights if weights else [1.0 / len(models)] * len(models)
        
        if len(self.models) != len(self.weights):
            raise ValueError("Number of models must match number of weights")
            
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Train all ensemble models."""
        for model in self.models:
            model.fit(X, y, **kwargs)
        self.is_trained = True
        logger.info(f"Ensemble model trained with {len(self.models)} sub-models")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
            
        # Weighted average for regression, weighted voting for classification
        if hasattr(self.models[0], 'predict_proba'):
            # Classification with probabilities
            probas = []
            for model in self.models:
                proba = model.predict_proba(X)
                probas.append(proba)
                
            weighted_proba = np.average(probas, axis=0, weights=self.weights)
            return np.argmax(weighted_proba, axis=1)
        else:
            # Regression
            weighted_pred = np.average(predictions, axis=0, weights=self.weights)
            return weighted_pred
            
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict ensemble probabilities."""
        probas = []
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                probas.append(proba)
            else:
                # Convert regression predictions to probabilities
                pred = model.predict(X)
                proba = np.column_stack([1 - pred, pred])  # Binary classification
                probas.append(proba)
                
        return np.average(probas, axis=0, weights=self.weights)
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate ensemble model performance."""
        predictions = self.predict(X)
        
        if len(np.unique(y)) <= 2:  # Classification
            return {
                'accuracy': accuracy_score(y, predictions),
                'precision': precision_score(y, predictions, zero_division=0),
                'recall': recall_score(y, predictions, zero_division=0),
                'f1_score': f1_score(y, predictions, zero_division=0)
            }
        else:  # Regression
            return {
                'mse': mean_squared_error(y, predictions),
                'rmse': np.sqrt(mean_squared_error(y, predictions)),
                'mae': mean_absolute_error(y, predictions),
                'r2_score': r2_score(y, predictions)
            }


class ModelFactory:
    """Factory for creating ML models."""
    
    @staticmethod
    def create_model(model_type: str, algorithm: str = None, **kwargs) -> BaseMLModel:
        """Create a model instance based on type and algorithm."""
        
        if model_type == "anomaly_detection":
            return AnomalyDetectionModel(algorithm=algorithm, **kwargs)
        elif model_type == "failure_prediction":
            return FailurePredictionModel(algorithm=algorithm, **kwargs)
        elif model_type == "performance_prediction":
            return PerformancePredictionModel(algorithm=algorithm, **kwargs)
        elif model_type == "time_series":
            return TimeSeriesModel(algorithm=algorithm, **kwargs)
        elif model_type == "deep_learning":
            return DeepLearningModel(model_type=algorithm, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
    @staticmethod
    def create_ensemble(models: List[BaseMLModel], weights: Optional[List[float]] = None) -> EnsembleModel:
        """Create an ensemble model."""
        return EnsembleModel(models, weights) 