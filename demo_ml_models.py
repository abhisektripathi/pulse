#!/usr/bin/env python3
"""
Demo script for Advanced ML Models in Predictive System Health Platform

This script demonstrates:
1. Training various ML models with synthetic data
2. Making predictions on system health metrics
3. Model performance evaluation
4. Feature importance analysis
5. Real-time prediction capabilities
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our ML models
from src.models.ml_models import (
    ModelFactory, AnomalyDetectionModel, FailurePredictionModel,
    PerformancePredictionModel, BaseMLModel
)
from services.prediction.model_trainer import ModelTrainer
from src.models.predictions import (
    PredictionRequest, PredictionResponse, PredictionType, RiskLevel
)
from src.models.metrics import StandardMetric

class MLModelDemo:
    """Demonstration class for ML models."""
    
    def __init__(self):
        self.trainer = ModelTrainer()
        self.models = {}
        self.demo_data = {}
        
    def generate_demo_scenarios(self) -> Dict[str, np.ndarray]:
        """Generate realistic demo scenarios for testing."""
        logger.info("Generating demo scenarios...")
        
        scenarios = {}
        
        # Scenario 1: Normal operation
        normal_metrics = np.array([
            [45, 60, 55, 30, 2, 200, 1500, 100],  # CPU, Memory, Disk, Network, Errors, Response, Throughput, Connections
            [50, 65, 58, 35, 1, 180, 1600, 120],
            [42, 58, 52, 28, 3, 220, 1400, 90],
            [48, 62, 56, 32, 2, 190, 1550, 110],
            [46, 59, 54, 31, 1, 210, 1480, 105]
        ])
        scenarios['normal'] = normal_metrics
        
        # Scenario 2: High CPU usage
        high_cpu_metrics = np.array([
            [92, 70, 60, 40, 5, 300, 1200, 150],
            [95, 75, 62, 45, 8, 400, 1000, 180],
            [89, 68, 58, 38, 6, 350, 1100, 160],
            [94, 72, 61, 42, 7, 380, 1050, 170],
            [91, 69, 59, 39, 5, 320, 1150, 155]
        ])
        scenarios['high_cpu'] = high_cpu_metrics
        
        # Scenario 3: Memory pressure
        memory_pressure_metrics = np.array([
            [60, 88, 65, 35, 4, 250, 1300, 130],
            [65, 92, 68, 38, 6, 280, 1200, 140],
            [58, 85, 62, 32, 3, 230, 1350, 125],
            [62, 90, 66, 36, 5, 260, 1250, 135],
            [64, 87, 64, 34, 4, 240, 1320, 128]
        ])
        scenarios['memory_pressure'] = memory_pressure_metrics
        
        # Scenario 4: Critical failure conditions
        critical_metrics = np.array([
            [98, 95, 98, 90, 15, 2000, 500, 480],
            [99, 97, 99, 95, 18, 2200, 400, 490],
            [97, 93, 97, 88, 14, 1900, 550, 475],
            [96, 94, 96, 87, 16, 2100, 450, 485],
            [98, 96, 98, 92, 17, 2050, 480, 488]
        ])
        scenarios['critical'] = critical_metrics
        
        # Scenario 5: Performance degradation
        performance_degradation = np.array([
            [70, 75, 70, 60, 8, 800, 800, 200],
            [75, 80, 72, 65, 10, 900, 700, 220],
            [68, 72, 68, 58, 7, 750, 850, 190],
            [72, 78, 71, 62, 9, 850, 750, 210],
            [73, 76, 69, 61, 8, 800, 800, 205]
        ])
        scenarios['performance_degradation'] = performance_degradation
        
        self.demo_data = scenarios
        logger.info(f"Generated {len(scenarios)} demo scenarios")
        return scenarios
        
    async def train_models(self):
        """Train all ML models."""
        logger.info("Training ML models...")
        
        try:
            # Train all models with synthetic data
            results = self.trainer.train_all_models(n_samples=5000)
            
            # Store the best models
            best_models = self.trainer.load_best_models()
            
            for model_type, model_info in best_models.items():
                self.models[model_type] = model_info['model']
                logger.info(f"Loaded {model_type} model with accuracy: {model_info['metrics'].get('accuracy', 'N/A')}")
                
            logger.info("✅ All models trained successfully")
            return results
            
        except Exception as e:
            logger.error(f"Failed to train models: {e}")
            return None
            
    def run_anomaly_detection_demo(self):
        """Demonstrate anomaly detection capabilities."""
        logger.info("🔍 Running Anomaly Detection Demo")
        
        if 'anomaly_detector' not in self.models:
            logger.warning("Anomaly detection model not available")
            return
            
        model = self.models['anomaly_detector']
        
        results = {}
        for scenario_name, metrics in self.demo_data.items():
            # Make predictions
            predictions = model.predict(metrics)
            scores = model.predict_proba(metrics).flatten()
            
            results[scenario_name] = {
                'predictions': predictions.tolist(),
                'anomaly_scores': scores.tolist(),
                'anomaly_detected': np.any(predictions == -1),
                'avg_anomaly_score': np.mean(scores)
            }
            
            logger.info(f"  {scenario_name}: Anomaly detected: {results[scenario_name]['anomaly_detected']}, "
                       f"Avg score: {results[scenario_name]['avg_anomaly_score']:.3f}")
            
        return results
        
    def run_failure_prediction_demo(self):
        """Demonstrate failure prediction capabilities."""
        logger.info("⚠️ Running Failure Prediction Demo")
        
        if 'failure_predictor' not in self.models:
            logger.warning("Failure prediction model not available")
            return
            
        model = self.models['failure_predictor']
        
        results = {}
        for scenario_name, metrics in self.demo_data.items():
            # Make predictions
            predictions = model.predict(metrics)
            probabilities = model.predict_proba(metrics)[:, 1]  # Probability of failure
            
            results[scenario_name] = {
                'predictions': predictions.tolist(),
                'failure_probabilities': probabilities.tolist(),
                'failure_predicted': np.any(predictions == 1),
                'avg_failure_probability': np.mean(probabilities),
                'max_failure_probability': np.max(probabilities)
            }
            
            risk_level = "LOW"
            if results[scenario_name]['max_failure_probability'] > 0.8:
                risk_level = "CRITICAL"
            elif results[scenario_name]['max_failure_probability'] > 0.6:
                risk_level = "HIGH"
            elif results[scenario_name]['max_failure_probability'] > 0.4:
                risk_level = "MEDIUM"
                
            logger.info(f"  {scenario_name}: Risk level: {risk_level}, "
                       f"Max failure prob: {results[scenario_name]['max_failure_probability']:.3f}")
            
        return results
        
    def run_performance_prediction_demo(self):
        """Demonstrate performance prediction capabilities."""
        logger.info("📊 Running Performance Prediction Demo")
        
        if 'performance_predictor' not in self.models:
            logger.warning("Performance prediction model not available")
            return
            
        model = self.models['performance_predictor']
        
        results = {}
        for scenario_name, metrics in self.demo_data.items():
            # Make predictions
            predictions = model.predict(metrics)
            
            results[scenario_name] = {
                'predicted_performance': predictions.tolist(),
                'avg_performance': np.mean(predictions),
                'min_performance': np.min(predictions),
                'performance_trend': 'stable'
            }
            
            # Determine performance trend
            if np.std(predictions) > 10:
                results[scenario_name]['performance_trend'] = 'volatile'
            elif np.mean(predictions) < 50:
                results[scenario_name]['performance_trend'] = 'degrading'
            elif np.mean(predictions) > 80:
                results[scenario_name]['performance_trend'] = 'excellent'
                
            logger.info(f"  {scenario_name}: Avg performance: {results[scenario_name]['avg_performance']:.1f}, "
                       f"Trend: {results[scenario_name]['performance_trend']}")
            
        return results
        
    def run_real_time_prediction_demo(self):
        """Demonstrate real-time prediction capabilities."""
        logger.info("⚡ Running Real-time Prediction Demo")
        
        # Simulate real-time data stream
        for i in range(10):
            # Generate random metrics
            metrics = np.random.uniform(20, 95, (1, 8))
            
            logger.info(f"  Timestamp {i+1}: Processing metrics...")
            
            results = {}
            
            # Anomaly detection
            if 'anomaly_detector' in self.models:
                anomaly_score = self.models['anomaly_detector'].predict_proba(metrics)[0, 0]
                results['anomaly_score'] = anomaly_score
                
            # Failure prediction
            if 'failure_predictor' in self.models:
                failure_prob = self.models['failure_predictor'].predict_proba(metrics)[0, 1]
                results['failure_probability'] = failure_prob
                
            # Performance prediction
            if 'performance_predictor' in self.models:
                performance = self.models['performance_predictor'].predict(metrics)[0]
                results['predicted_performance'] = performance
                
            # Log results
            logger.info(f"    Anomaly Score: {results.get('anomaly_score', 'N/A'):.3f}")
            logger.info(f"    Failure Probability: {results.get('failure_probability', 'N/A'):.3f}")
            logger.info(f"    Performance: {results.get('predicted_performance', 'N/A'):.1f}")
            
            # Simulate processing time
            time.sleep(0.5)
            
    def generate_prediction_report(self) -> Dict[str, Any]:
        """Generate a comprehensive prediction report."""
        logger.info("📋 Generating Prediction Report")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_summary': {},
            'scenario_analysis': {},
            'recommendations': []
        }
        
        # Model summary
        for model_type, model_info in self.trainer.model_registry.items():
            report['model_summary'][model_type] = {
                'metrics': model_info['metrics'],
                'created_at': model_info['created_at'].isoformat()
            }
            
        # Scenario analysis
        anomaly_results = self.run_anomaly_detection_demo()
        failure_results = self.run_failure_prediction_demo()
        performance_results = self.run_performance_prediction_demo()
        
        report['scenario_analysis'] = {
            'anomaly_detection': anomaly_results,
            'failure_prediction': failure_results,
            'performance_prediction': performance_results
        }
        
        # Generate recommendations
        recommendations = []
        
        # Check for critical scenarios
        if failure_results and 'critical' in failure_results:
            if failure_results['critical']['max_failure_probability'] > 0.8:
                recommendations.append("🚨 CRITICAL: Immediate intervention required - high failure probability detected")
                
        if anomaly_results and 'critical' in anomaly_results:
            if anomaly_results['critical']['avg_anomaly_score'] > 0.7:
                recommendations.append("⚠️ WARNING: Multiple anomalies detected in critical scenario")
                
        if performance_results and 'performance_degradation' in performance_results:
            if performance_results['performance_degradation']['avg_performance'] < 60:
                recommendations.append("📉 ALERT: Performance degradation detected - consider scaling resources")
                
        # Add general recommendations
        recommendations.extend([
            "🔍 Monitor anomaly scores continuously",
            "📊 Track performance trends over time",
            "⚡ Set up automated alerts for high-risk predictions",
            "🔄 Retrain models regularly with new data"
        ])
        
        report['recommendations'] = recommendations
        
        return report
        
    async def run_full_demo(self):
        """Run the complete ML model demonstration."""
        logger.info("🚀 Starting ML Model Demonstration")
        
        # Generate demo scenarios
        self.generate_demo_scenarios()
        
        # Train models
        training_results = await self.train_models()
        if not training_results:
            logger.error("Failed to train models. Exiting demo.")
            return
            
        # Run individual demos
        logger.info("\n" + "="*60)
        self.run_anomaly_detection_demo()
        
        logger.info("\n" + "="*60)
        self.run_failure_prediction_demo()
        
        logger.info("\n" + "="*60)
        self.run_performance_prediction_demo()
        
        logger.info("\n" + "="*60)
        self.run_real_time_prediction_demo()
        
        # Generate final report
        logger.info("\n" + "="*60)
        report = self.generate_prediction_report()
        
        # Save report
        with open('ml_demo_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        logger.info("✅ Demo completed! Report saved to ml_demo_report.json")
        
        return report


async def main():
    """Main function to run the ML model demo."""
    demo = MLModelDemo()
    await demo.run_full_demo()


if __name__ == "__main__":
    asyncio.run(main()) 