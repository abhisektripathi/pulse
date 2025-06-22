"""
Model Trainer Service for Predictive System Health Platform

This service handles:
- Training new ML models with real data
- Model validation and evaluation
- Model versioning and deployment
- Hyperparameter optimization
- Model performance monitoring
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
import joblib
import os
from pathlib import Path

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_squared_error, accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Try to import optional dependencies
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not available. Install with: pip install mlflow")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Install with: pip install xgboost")

from src.models.ml_models import (
    ModelFactory, AnomalyDetectionModel, FailurePredictionModel,
    PerformancePredictionModel, TimeSeriesModel, DeepLearningModel, BaseMLModel
)
from src.models.metrics import StandardMetric
from src.models.health import HealthScore

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Service for training and managing ML models."""
    
    def __init__(self, models_dir: str = "models", mlflow_uri: str = "http://localhost:5000"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.mlflow_uri = mlflow_uri
        
        # Initialize MLflow if available
        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(mlflow_uri)
        
        # Model registry
        self.model_registry = {}
        self.training_history = {}
        
    def generate_synthetic_data(self, n_samples: int = 10000, seed: int = 42) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Generate synthetic training data for model development."""
        np.random.seed(seed)
        
        # Generate realistic system metrics
        cpu_usage = np.random.uniform(20, 95, n_samples)
        memory_usage = np.random.uniform(30, 90, n_samples)
        disk_usage = np.random.uniform(40, 85, n_samples)
        network_usage = np.random.uniform(10, 80, n_samples)
        error_rate = np.random.uniform(0, 15, n_samples)
        response_time = np.random.uniform(50, 2000, n_samples)
        throughput = np.random.uniform(100, 3000, n_samples)
        active_connections = np.random.uniform(10, 500, n_samples)
        
        # Create feature matrix
        X = np.column_stack([
            cpu_usage, memory_usage, disk_usage, network_usage,
            error_rate, response_time, throughput, active_connections
        ])
        
        # Generate target variables
        targets = {}
        
        # Anomaly labels (anomalies when multiple metrics are extreme)
        anomaly_labels = (
            (cpu_usage > 90) | (memory_usage > 85) |
            (error_rate > 10) | (response_time > 1500) |
            (disk_usage > 95) | (network_usage > 90)
        ).astype(int)
        targets['anomaly'] = anomaly_labels
        
        # Failure labels (failures when critical thresholds are exceeded)
        failure_labels = (
            (cpu_usage > 95) & (memory_usage > 90) |
            (error_rate > 12) & (response_time > 1800) |
            (disk_usage > 98) |
            (active_connections > 450)
        ).astype(int)
        targets['failure'] = failure_labels
        
        # Performance scores (0-100, lower is worse)
        performance_scores = 100 - (
            cpu_usage * 0.25 + memory_usage * 0.25 + 
            disk_usage * 0.15 + error_rate * 2.5 + 
            (response_time / 2000) * 15 + (network_usage * 0.2)
        )
        performance_scores = np.clip(performance_scores, 0, 100)
        targets['performance'] = performance_scores
        
        # Resource exhaustion (time to resource exhaustion in hours)
        resource_exhaustion = np.where(
            (cpu_usage > 80) | (memory_usage > 80) | (disk_usage > 80),
            np.random.uniform(1, 48, n_samples),  # Near exhaustion
            np.random.uniform(48, 720, n_samples)  # Safe
        )
        targets['resource_exhaustion'] = resource_exhaustion
        
        return X, targets
        
    def train_anomaly_detection_model(self, X: np.ndarray, y: np.ndarray, 
                                    algorithm: str = "isolation_forest", **kwargs) -> Union[AnomalyDetectionModel, BaseMLModel]:
        """Train anomaly detection model."""
        logger.info(f"Training anomaly detection model with {algorithm}")
        
        # Create model
        model = ModelFactory.create_model("anomaly_detection", algorithm, **kwargs)
        
        # Train model
        model.fit(X, y)
        
        # Evaluate model
        predictions = model.predict(X)
        scores = model.predict_proba(X).flatten()
        
        # Calculate metrics
        y_binary = (y == 1).astype(int)
        pred_binary = (predictions == 1).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_binary, pred_binary),
            'anomaly_detection_rate': np.mean(scores[y == 0]),
            'false_positive_rate': np.mean(scores[y == 1])
        }
        
        # Store model
        model_name = f"anomaly_detector_{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_path = self.models_dir / f"{model_name}.joblib"
        model.save_model(str(model_path))
        
        # Log to MLflow if available
        if MLFLOW_AVAILABLE:
            with mlflow.start_run():
                mlflow.log_params({
                    'algorithm': algorithm,
                    'n_samples': X.shape[0],
                    'n_features': X.shape[1],
                    **kwargs
                })
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(model.model, model_name)
            
        self.model_registry[model_name] = {
            'model': model,
            'path': model_path,
            'metrics': metrics,
            'created_at': datetime.now()
        }
        
        logger.info(f"Anomaly detection model trained successfully: {model_name}")
        return model
        
    def train_failure_prediction_model(self, X: np.ndarray, y: np.ndarray,
                                     algorithm: str = "random_forest", **kwargs) -> Union[FailurePredictionModel, BaseMLModel]:
        """Train failure prediction model."""
        logger.info(f"Training failure prediction model with {algorithm}")
        
        # Create model
        model = ModelFactory.create_model("failure_prediction", algorithm, **kwargs)
        
        # Train model
        model.fit(X, y)
        
        # Evaluate model
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, predictions),
            'failure_detection_rate': np.mean(probabilities[y == 1]),
            'false_alarm_rate': np.mean(probabilities[y == 0])
        }
        
        # Store model
        model_name = f"failure_predictor_{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_path = self.models_dir / f"{model_name}.joblib"
        model.save_model(str(model_path))
        
        # Log to MLflow if available
        if MLFLOW_AVAILABLE:
            with mlflow.start_run():
                mlflow.log_params({
                    'algorithm': algorithm,
                    'n_samples': X.shape[0],
                    'n_features': X.shape[1],
                    **kwargs
                })
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(model.model, model_name)
            
        self.model_registry[model_name] = {
            'model': model,
            'path': model_path,
            'metrics': metrics,
            'created_at': datetime.now()
        }
        
        logger.info(f"Failure prediction model trained successfully: {model_name}")
        return model
        
    def train_performance_prediction_model(self, X: np.ndarray, y: np.ndarray,
                                         algorithm: str = "random_forest", **kwargs) -> Union[PerformancePredictionModel, BaseMLModel]:
        """Train performance prediction model."""
        logger.info(f"Training performance prediction model with {algorithm}")
        
        # Create model
        model = ModelFactory.create_model("performance_prediction", algorithm, **kwargs)
        
        # Train model
        model.fit(X, y)
        
        # Evaluate model
        predictions = model.predict(X)
        
        metrics = {
            'mse': mean_squared_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'mae': np.mean(np.abs(y - predictions)),
            'r2_score': 1 - np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2)
        }
        
        # Store model
        model_name = f"performance_predictor_{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_path = self.models_dir / f"{model_name}.joblib"
        model.save_model(str(model_path))
        
        # Log to MLflow if available
        if MLFLOW_AVAILABLE:
            with mlflow.start_run():
                mlflow.log_params({
                    'algorithm': algorithm,
                    'n_samples': X.shape[0],
                    'n_features': X.shape[1],
                    **kwargs
                })
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(model.model, model_name)
            
        self.model_registry[model_name] = {
            'model': model,
            'path': model_path,
            'metrics': metrics,
            'created_at': datetime.now()
        }
        
        logger.info(f"Performance prediction model trained successfully: {model_name}")
        return model
        
    def hyperparameter_optimization(self, model_type: str, X: np.ndarray, y: np.ndarray,
                                  algorithm: str = "random_forest", cv_folds: int = 5) -> Dict[str, Any]:
        """Perform hyperparameter optimization."""
        logger.info(f"Performing hyperparameter optimization for {model_type}")
        
        if model_type == "failure_prediction":
            # Define parameter grids for different algorithms
            if algorithm == "random_forest":
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                base_model = RandomForestClassifier(random_state=42)
            elif algorithm == "xgboost" and XGBOOST_AVAILABLE:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                }
                base_model = xgb.XGBClassifier(random_state=42)
            else:
                raise ValueError(f"Unsupported algorithm for optimization: {algorithm}")
                
            # Perform grid search
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv_folds,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X, y)
            
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            
            logger.info(f"Best parameters: {best_params}")
            logger.info(f"Best score: {best_score}")
            
            return {
                'best_params': best_params,
                'best_score': best_score,
                'cv_results': grid_search.cv_results_
            }
        else:
            raise ValueError(f"Hyperparameter optimization not implemented for {model_type}")
            
    def train_all_models(self, n_samples: int = 10000) -> Dict[str, Any]:
        """Train all model types with synthetic data."""
        logger.info("Training all ML models")
        
        # Generate synthetic data
        X, targets = self.generate_synthetic_data(n_samples)
        
        results = {}
        
        # Train anomaly detection models
        anomaly_models = {}
        for algorithm in ["isolation_forest", "local_outlier_factor", "one_class_svm"]:
            try:
                model = self.train_anomaly_detection_model(X, targets['anomaly'], algorithm)
                anomaly_models[algorithm] = model
            except Exception as e:
                logger.error(f"Failed to train anomaly detection model with {algorithm}: {e}")
                
        results['anomaly_detection'] = anomaly_models
        
        # Train failure prediction models
        failure_models = {}
        for algorithm in ["random_forest", "gradient_boosting"]:
            try:
                model = self.train_failure_prediction_model(X, targets['failure'], algorithm)
                failure_models[algorithm] = model
            except Exception as e:
                logger.error(f"Failed to train failure prediction model with {algorithm}: {e}")
                
        results['failure_prediction'] = failure_models
        
        # Train performance prediction models
        performance_models = {}
        for algorithm in ["random_forest", "gradient_boosting", "svr"]:
            try:
                model = self.train_performance_prediction_model(X, targets['performance'], algorithm)
                performance_models[algorithm] = model
            except Exception as e:
                logger.error(f"Failed to train performance prediction model with {algorithm}: {e}")
                
        results['performance_prediction'] = performance_models
        
        # Perform hyperparameter optimization
        try:
            opt_results = self.hyperparameter_optimization("failure_prediction", X, targets['failure'])
            results['hyperparameter_optimization'] = opt_results
        except Exception as e:
            logger.error(f"Failed to perform hyperparameter optimization: {e}")
            
        logger.info("All models trained successfully")
        return results
        
    def load_best_models(self) -> Dict[str, Any]:
        """Load the best performing models from the registry."""
        best_models = {}
        
        for model_name, model_info in self.model_registry.items():
            model_type = model_name.split('_')[0]  # Extract model type from name
            
            if model_type not in best_models:
                best_models[model_type] = model_info
            else:
                # Compare metrics and keep the best one
                current_metrics = best_models[model_type]['metrics']
                new_metrics = model_info['metrics']
                
                # Simple comparison based on accuracy or similar metric
                if 'accuracy' in new_metrics and 'accuracy' in current_metrics:
                    if new_metrics['accuracy'] > current_metrics['accuracy']:
                        best_models[model_type] = model_info
                        
        return best_models
        
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get summary of all model performances."""
        summary = {
            'total_models': len(self.model_registry),
            'model_types': {},
            'best_models': self.load_best_models(),
            'training_history': self.training_history
        }
        
        # Group models by type
        for model_name, model_info in self.model_registry.items():
            model_type = model_name.split('_')[0]
            if model_type not in summary['model_types']:
                summary['model_types'][model_type] = []
            summary['model_types'][model_type].append({
                'name': model_name,
                'metrics': model_info['metrics'],
                'created_at': model_info['created_at']
            })
            
        return summary


# Example usage
if __name__ == "__main__":
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Train all models
    results = trainer.train_all_models(n_samples=5000)
    
    # Get performance summary
    summary = trainer.get_model_performance_summary()
    print(json.dumps(summary, indent=2, default=str)) 