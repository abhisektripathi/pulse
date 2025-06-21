#!/usr/bin/env python3
"""
Data Simulator Service for Predictive System Health Platform

Generates realistic observability data for a payment processing system including:
- Infrastructure metrics (CPU, memory, disk, network)
- Application metrics (response time, throughput, error rates)
- Business metrics (payment success rates, transaction volumes)
- Anomalies and incidents for testing
"""

import asyncio
import json
import logging
import random
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
from kafka import KafkaProducer
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

from src.models.metrics import (
    StandardMetric, EntityInfo, BusinessContext, MetricType, MetricBatch
)
from src.models.entities import (
    SystemEntity, ServiceEntity, DatabaseEntity, HostEntity, EntityType, Environment
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PaymentSystemDataSimulator:
    """Simulates realistic observability data for a payment processing system."""
    
    def __init__(self, kafka_brokers: str, influxdb_url: str, simulation_interval: int = 30):
        self.kafka_brokers = kafka_brokers
        self.influxdb_url = influxdb_url
        self.simulation_interval = simulation_interval
        
        # Initialize clients
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=kafka_brokers,
            value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8')
        )
        
        self.influxdb_client = InfluxDBClient(
            url=influxdb_url,
            token="system-health-token",
            org="system-health"
        )
        self.write_api = self.influxdb_client.write_api(write_options=SYNCHRONOUS)
        
        # Define payment system entities
        self.entities = self._define_payment_system_entities()
        
        # Simulation state
        self.current_time = datetime.utcnow()
        self.anomaly_mode = False
        self.incident_active = False
        
    def _define_payment_system_entities(self) -> Dict[str, SystemEntity]:
        """Define the payment system architecture and entities."""
        
        entities = {}
        
        # Payment Gateway Service
        entities["payment-gateway"] = ServiceEntity(
            id="payment-gateway",
            name="Payment Gateway Service",
            type=EntityType.SERVICE,
            environment=Environment.PRODUCTION,
            criticality=5,
            team="payments-core",
            service_type="microservice",
            version="2.1.0",
            endpoints=["/api/v1/payments", "/api/v1/status"],
            health_check_url="/health",
            sla_target_ms=200,
            max_instances=10,
            min_instances=3,
            scaling_policy="cpu-based"
        )
        
        # Fraud Detection Service
        entities["fraud-detection"] = ServiceEntity(
            id="fraud-detection",
            name="Fraud Detection Service",
            type=EntityType.SERVICE,
            environment=Environment.PRODUCTION,
            criticality=4,
            team="risk-management",
            service_type="microservice",
            version="1.8.2",
            endpoints=["/api/v1/fraud/check", "/api/v1/fraud/analyze"],
            health_check_url="/health",
            sla_target_ms=500,
            max_instances=8,
            min_instances=2,
            scaling_policy="queue-based"
        )
        
        # Payment Database
        entities["payment-db"] = DatabaseEntity(
            id="payment-db",
            name="Payment Database Cluster",
            type=EntityType.DATABASE,
            environment=Environment.PRODUCTION,
            criticality=5,
            team="database-team",
            database_type="postgresql",
            version="14.5",
            host="payment-db-cluster",
            port=5432,
            database_name="payments",
            connection_pool_size=50,
            max_connections=200,
            backup_schedule="hourly",
            replication_factor=3
        )
        
        # Settlement Service
        entities["settlement-service"] = ServiceEntity(
            id="settlement-service",
            name="Settlement Processing Service",
            type=EntityType.SERVICE,
            environment=Environment.PRODUCTION,
            criticality=4,
            team="payments-core",
            service_type="microservice",
            version="1.5.1",
            endpoints=["/api/v1/settlement/process", "/api/v1/settlement/status"],
            health_check_url="/health",
            sla_target_ms=1000,
            max_instances=6,
            min_instances=2,
            scaling_policy="batch-based"
        )
        
        # Notification Service
        entities["notification-service"] = ServiceEntity(
            id="notification-service",
            name="Notification Service",
            type=EntityType.SERVICE,
            environment=Environment.PRODUCTION,
            criticality=3,
            team="platform-team",
            service_type="microservice",
            version="1.2.0",
            endpoints=["/api/v1/notifications/send", "/api/v1/notifications/status"],
            health_check_url="/health",
            sla_target_ms=300,
            max_instances=5,
            min_instances=1,
            scaling_policy="cpu-based"
        )
        
        # Host Infrastructure
        entities["host-payment-01"] = HostEntity(
            id="host-payment-01",
            name="Payment Host 01",
            type=EntityType.HOST,
            environment=Environment.PRODUCTION,
            criticality=4,
            team="infrastructure",
            hostname="payment-host-01.prod.company.com",
            ip_address="10.0.1.10",
            os_type="linux",
            os_version="Ubuntu 20.04",
            cpu_cores=16,
            memory_gb=64.0,
            disk_gb=1000.0,
            location="us-east-1",
            datacenter="primary"
        )
        
        entities["host-payment-02"] = HostEntity(
            id="host-payment-02",
            name="Payment Host 02",
            type=EntityType.HOST,
            environment=Environment.PRODUCTION,
            criticality=4,
            team="infrastructure",
            hostname="payment-host-02.prod.company.com",
            ip_address="10.0.1.11",
            os_type="linux",
            os_version="Ubuntu 20.04",
            cpu_cores=16,
            memory_gb=64.0,
            disk_gb=1000.0,
            location="us-east-1",
            datacenter="primary"
        )
        
        return entities
    
    def _generate_business_context(self, entity_id: str) -> BusinessContext:
        """Generate business context for an entity."""
        
        business_contexts = {
            "payment-gateway": BusinessContext(
                service_line="payments",
                revenue_impact=95.5,
                customer_impact=5,
                compliance_tag="PCI-DSS",
                sla_target=200.0,
                business_priority="critical"
            ),
            "fraud-detection": BusinessContext(
                service_line="risk-management",
                revenue_impact=85.0,
                customer_impact=4,
                compliance_tag="SOX",
                sla_target=500.0,
                business_priority="high"
            ),
            "payment-db": BusinessContext(
                service_line="payments",
                revenue_impact=98.0,
                customer_impact=5,
                compliance_tag="PCI-DSS",
                sla_target=50.0,
                business_priority="critical"
            ),
            "settlement-service": BusinessContext(
                service_line="payments",
                revenue_impact=90.0,
                customer_impact=4,
                compliance_tag="SOX",
                sla_target=1000.0,
                business_priority="high"
            ),
            "notification-service": BusinessContext(
                service_line="platform",
                revenue_impact=60.0,
                customer_impact=3,
                compliance_tag="GDPR",
                sla_target=300.0,
                business_priority="medium"
            )
        }
        
        return business_contexts.get(entity_id, BusinessContext(
            service_line="infrastructure",
            revenue_impact=70.0,
            customer_impact=3,
            business_priority="medium"
        ))
    
    def _generate_infrastructure_metrics(self, entity_id: str, timestamp: datetime) -> List[StandardMetric]:
        """Generate infrastructure metrics for hosts and databases."""
        
        metrics = []
        entity = self.entities[entity_id]
        
        if entity.type == EntityType.HOST:
            # Generate realistic CPU usage with some variation
            base_cpu = 45.0
            if self.anomaly_mode and entity_id == "host-payment-01":
                base_cpu = 85.0  # Simulate high CPU during anomaly
            
            cpu_usage = max(0, min(100, base_cpu + random.gauss(0, 5)))
            
            # Memory usage with gradual increase
            base_memory = 65.0
            memory_usage = max(0, min(100, base_memory + random.gauss(0, 3)))
            
            # Disk usage (slowly increasing)
            base_disk = 45.0 + (timestamp.hour * 0.1)  # Gradual increase
            disk_usage = max(0, min(100, base_disk + random.gauss(0, 1)))
            
            # Network I/O
            network_io = max(0, 100 + random.gauss(0, 20))
            
            metrics.extend([
                StandardMetric(
                    timestamp=timestamp,
                    source="prometheus",
                    metric_type=MetricType.INFRASTRUCTURE,
                    entity=EntityInfo(
                        id=entity_id,
                        name=entity.name,
                        type="host",
                        environment=entity.environment.value,
                        criticality=entity.criticality,
                        team=entity.team
                    ),
                    value=cpu_usage,
                    unit="percent",
                    business_info=self._generate_business_context(entity_id),
                    tags={"metric": "cpu_usage"}
                ),
                StandardMetric(
                    timestamp=timestamp,
                    source="prometheus",
                    metric_type=MetricType.INFRASTRUCTURE,
                    entity=EntityInfo(
                        id=entity_id,
                        name=entity.name,
                        type="host",
                        environment=entity.environment.value,
                        criticality=entity.criticality,
                        team=entity.team
                    ),
                    value=memory_usage,
                    unit="percent",
                    business_info=self._generate_business_context(entity_id),
                    tags={"metric": "memory_usage"}
                ),
                StandardMetric(
                    timestamp=timestamp,
                    source="prometheus",
                    metric_type=MetricType.INFRASTRUCTURE,
                    entity=EntityInfo(
                        id=entity_id,
                        name=entity.name,
                        type="host",
                        environment=entity.environment.value,
                        criticality=entity.criticality,
                        team=entity.team
                    ),
                    value=disk_usage,
                    unit="percent",
                    business_info=self._generate_business_context(entity_id),
                    tags={"metric": "disk_usage"}
                ),
                StandardMetric(
                    timestamp=timestamp,
                    source="prometheus",
                    metric_type=MetricType.INFRASTRUCTURE,
                    entity=EntityInfo(
                        id=entity_id,
                        name=entity.name,
                        type="host",
                        environment=entity.environment.value,
                        criticality=entity.criticality,
                        team=entity.team
                    ),
                    value=network_io,
                    unit="mbps",
                    business_info=self._generate_business_context(entity_id),
                    tags={"metric": "network_io"}
                )
            ])
        
        elif entity.type == EntityType.DATABASE:
            # Database-specific metrics
            active_connections = random.randint(20, 80)
            query_response_time = random.gauss(15, 5)  # ms
            connection_pool_usage = random.gauss(60, 10)
            
            metrics.extend([
                StandardMetric(
                    timestamp=timestamp,
                    source="prometheus",
                    metric_type=MetricType.INFRASTRUCTURE,
                    entity=EntityInfo(
                        id=entity_id,
                        name=entity.name,
                        type="database",
                        environment=entity.environment.value,
                        criticality=entity.criticality,
                        team=entity.team
                    ),
                    value=active_connections,
                    unit="connections",
                    business_info=self._generate_business_context(entity_id),
                    tags={"metric": "active_connections"}
                ),
                StandardMetric(
                    timestamp=timestamp,
                    source="prometheus",
                    metric_type=MetricType.INFRASTRUCTURE,
                    entity=EntityInfo(
                        id=entity_id,
                        name=entity.name,
                        type="database",
                        environment=entity.environment.value,
                        criticality=entity.criticality,
                        team=entity.team
                    ),
                    value=query_response_time,
                    unit="milliseconds",
                    business_info=self._generate_business_context(entity_id),
                    tags={"metric": "query_response_time"}
                ),
                StandardMetric(
                    timestamp=timestamp,
                    source="prometheus",
                    metric_type=MetricType.INFRASTRUCTURE,
                    entity=EntityInfo(
                        id=entity_id,
                        name=entity.name,
                        type="database",
                        environment=entity.environment.value,
                        criticality=entity.criticality,
                        team=entity.team
                    ),
                    value=connection_pool_usage,
                    unit="percent",
                    business_info=self._generate_business_context(entity_id),
                    tags={"metric": "connection_pool_usage"}
                )
            ])
        
        return metrics
    
    def _generate_application_metrics(self, entity_id: str, timestamp: datetime) -> List[StandardMetric]:
        """Generate application metrics for services."""
        
        metrics = []
        entity = self.entities[entity_id]
        
        if entity.type == EntityType.SERVICE:
            # Base response time varies by service
            base_response_times = {
                "payment-gateway": 120,
                "fraud-detection": 250,
                "settlement-service": 800,
                "notification-service": 150
            }
            
            base_response_time = base_response_times.get(entity_id, 200)
            
            # Simulate anomalies
            if self.anomaly_mode and entity_id == "payment-gateway":
                base_response_time = 800  # High latency during anomaly
            
            response_time = max(0, base_response_time + random.gauss(0, 20))
            
            # Throughput (requests per second)
            base_throughput = {
                "payment-gateway": 1500,
                "fraud-detection": 800,
                "settlement-service": 200,
                "notification-service": 300
            }
            
            throughput = max(0, base_throughput.get(entity_id, 500) + random.gauss(0, 50))
            
            # Error rate
            base_error_rate = 0.5  # 0.5%
            if self.anomaly_mode and entity_id == "payment-gateway":
                base_error_rate = 8.0  # High error rate during anomaly
            
            error_rate = max(0, min(100, base_error_rate + random.gauss(0, 0.2)))
            
            # Success rate (inverse of error rate)
            success_rate = 100 - error_rate
            
            # Queue depth
            queue_depth = random.randint(0, 50)
            if self.anomaly_mode and entity_id == "payment-gateway":
                queue_depth = random.randint(100, 200)  # High queue during anomaly
            
            metrics.extend([
                StandardMetric(
                    timestamp=timestamp,
                    source="appdynamics",
                    metric_type=MetricType.APPLICATION,
                    entity=EntityInfo(
                        id=entity_id,
                        name=entity.name,
                        type="service",
                        environment=entity.environment.value,
                        criticality=entity.criticality,
                        team=entity.team
                    ),
                    value=response_time,
                    unit="milliseconds",
                    business_info=self._generate_business_context(entity_id),
                    tags={"metric": "response_time"}
                ),
                StandardMetric(
                    timestamp=timestamp,
                    source="appdynamics",
                    metric_type=MetricType.APPLICATION,
                    entity=EntityInfo(
                        id=entity_id,
                        name=entity.name,
                        type="service",
                        environment=entity.environment.value,
                        criticality=entity.criticality,
                        team=entity.team
                    ),
                    value=throughput,
                    unit="requests_per_second",
                    business_info=self._generate_business_context(entity_id),
                    tags={"metric": "throughput"}
                ),
                StandardMetric(
                    timestamp=timestamp,
                    source="appdynamics",
                    metric_type=MetricType.APPLICATION,
                    entity=EntityInfo(
                        id=entity_id,
                        name=entity.name,
                        type="service",
                        environment=entity.environment.value,
                        criticality=entity.criticality,
                        team=entity.team
                    ),
                    value=error_rate,
                    unit="percent",
                    business_info=self._generate_business_context(entity_id),
                    tags={"metric": "error_rate"}
                ),
                StandardMetric(
                    timestamp=timestamp,
                    source="appdynamics",
                    metric_type=MetricType.APPLICATION,
                    entity=EntityInfo(
                        id=entity_id,
                        name=entity.name,
                        type="service",
                        environment=entity.environment.value,
                        criticality=entity.criticality,
                        team=entity.team
                    ),
                    value=success_rate,
                    unit="percent",
                    business_info=self._generate_business_context(entity_id),
                    tags={"metric": "success_rate"}
                ),
                StandardMetric(
                    timestamp=timestamp,
                    source="appdynamics",
                    metric_type=MetricType.APPLICATION,
                    entity=EntityInfo(
                        id=entity_id,
                        name=entity.name,
                        type="service",
                        environment=entity.environment.value,
                        criticality=entity.criticality,
                        team=entity.team
                    ),
                    value=queue_depth,
                    unit="requests",
                    business_info=self._generate_business_context(entity_id),
                    tags={"metric": "queue_depth"}
                )
            ])
        
        return metrics
    
    def _generate_business_metrics(self, timestamp: datetime) -> List[StandardMetric]:
        """Generate business metrics for the payment system."""
        
        metrics = []
        
        # Payment success rate
        base_success_rate = 99.2
        if self.anomaly_mode:
            base_success_rate = 94.5  # Lower success rate during anomaly
        
        payment_success_rate = max(0, min(100, base_success_rate + random.gauss(0, 0.3)))
        
        # Transaction volume (varies by hour)
        base_volume = 10000
        hour_factor = 1.0
        
        if 9 <= timestamp.hour <= 17:  # Business hours
            hour_factor = 2.5
        elif 18 <= timestamp.hour <= 22:  # Evening peak
            hour_factor = 1.8
        elif 23 <= timestamp.hour or timestamp.hour <= 6:  # Night
            hour_factor = 0.3
        
        transaction_volume = max(0, int(base_volume * hour_factor + random.gauss(0, 500)))
        
        # Revenue per minute
        revenue_per_minute = transaction_volume * 0.15  # Average $0.15 per transaction
        
        # Fraud detection rate
        fraud_detection_rate = random.gauss(2.5, 0.5)  # 2.5% average
        
        # Settlement delay
        base_settlement_delay = 180  # 3 minutes
        if self.anomaly_mode:
            base_settlement_delay = 600  # 10 minutes during anomaly
        
        settlement_delay = max(0, base_settlement_delay + random.gauss(0, 30))
        
        # Create business metrics for payment-gateway entity
        payment_gateway_entity = self.entities["payment-gateway"]
        
        metrics.extend([
            StandardMetric(
                timestamp=timestamp,
                source="business-api",
                metric_type=MetricType.BUSINESS,
                entity=EntityInfo(
                    id="payment-gateway",
                    name=payment_gateway_entity.name,
                    type="service",
                    environment=payment_gateway_entity.environment.value,
                    criticality=payment_gateway_entity.criticality,
                    team=payment_gateway_entity.team
                ),
                value=payment_success_rate,
                unit="percent",
                business_info=self._generate_business_context("payment-gateway"),
                tags={"metric": "payment_success_rate"}
            ),
            StandardMetric(
                timestamp=timestamp,
                source="business-api",
                metric_type=MetricType.BUSINESS,
                entity=EntityInfo(
                    id="payment-gateway",
                    name=payment_gateway_entity.name,
                    type="service",
                    environment=payment_gateway_entity.environment.value,
                    criticality=payment_gateway_entity.criticality,
                    team=payment_gateway_entity.team
                ),
                value=transaction_volume,
                unit="transactions_per_minute",
                business_info=self._generate_business_context("payment-gateway"),
                tags={"metric": "transaction_volume"}
            ),
            StandardMetric(
                timestamp=timestamp,
                source="business-api",
                metric_type=MetricType.BUSINESS,
                entity=EntityInfo(
                    id="payment-gateway",
                    name=payment_gateway_entity.name,
                    type="service",
                    environment=payment_gateway_entity.environment.value,
                    criticality=payment_gateway_entity.criticality,
                    team=payment_gateway_entity.team
                ),
                value=revenue_per_minute,
                unit="dollars_per_minute",
                business_info=self._generate_business_context("payment-gateway"),
                tags={"metric": "revenue_per_minute"}
            ),
            StandardMetric(
                timestamp=timestamp,
                source="business-api",
                metric_type=MetricType.BUSINESS,
                entity=EntityInfo(
                    id="fraud-detection",
                    name=self.entities["fraud-detection"].name,
                    type="service",
                    environment=self.entities["fraud-detection"].environment.value,
                    criticality=self.entities["fraud-detection"].criticality,
                    team=self.entities["fraud-detection"].team
                ),
                value=fraud_detection_rate,
                unit="percent",
                business_info=self._generate_business_context("fraud-detection"),
                tags={"metric": "fraud_detection_rate"}
            ),
            StandardMetric(
                timestamp=timestamp,
                source="business-api",
                metric_type=MetricType.BUSINESS,
                entity=EntityInfo(
                    id="settlement-service",
                    name=self.entities["settlement-service"].name,
                    type="service",
                    environment=self.entities["settlement-service"].environment.value,
                    criticality=self.entities["settlement-service"].criticality,
                    team=self.entities["settlement-service"].team
                ),
                value=settlement_delay,
                unit="seconds",
                business_info=self._generate_business_context("settlement-service"),
                tags={"metric": "settlement_delay"}
            )
        ])
        
        return metrics
    
    def _simulate_anomaly(self):
        """Simulate an anomaly in the payment system."""
        if not self.anomaly_mode and random.random() < 0.01:  # 1% chance per cycle
            self.anomaly_mode = True
            logger.info("🚨 ANOMALY DETECTED: Payment system performance degradation")
            
            # Schedule anomaly end
            asyncio.create_task(self._end_anomaly())
    
    async def _end_anomaly(self):
        """End the anomaly after a random duration."""
        duration = random.randint(300, 900)  # 5-15 minutes
        await asyncio.sleep(duration)
        self.anomaly_mode = False
        logger.info("✅ Anomaly resolved: Payment system back to normal")
    
    def _send_to_kafka(self, metrics: List[StandardMetric]):
        """Send metrics to Kafka topics."""
        try:
            # Send raw metrics
            raw_metrics = [metric.dict() for metric in metrics]
            self.kafka_producer.send('raw-metrics', raw_metrics)
            
            # Send business metrics separately
            business_metrics = [metric.dict() for metric in metrics if metric.metric_type == MetricType.BUSINESS]
            if business_metrics:
                self.kafka_producer.send('business-metrics', business_metrics)
            
            self.kafka_producer.flush()
            
        except Exception as e:
            logger.error(f"Failed to send metrics to Kafka: {e}")
    
    def _send_to_influxdb(self, metrics: List[StandardMetric]):
        """Send metrics to InfluxDB."""
        try:
            points = []
            
            for metric in metrics:
                point = Point("system_metrics") \
                    .tag("source", metric.source) \
                    .tag("entity_id", metric.entity.id) \
                    .tag("entity_type", metric.entity.type) \
                    .tag("environment", metric.entity.environment) \
                    .tag("metric_type", metric.metric_type.value) \
                    .tag("service_line", metric.business_info.service_line) \
                    .field("value", float(metric.value)) \
                    .field("business_impact_score", metric.business_info.revenue_impact) \
                    .field("criticality_level", metric.entity.criticality) \
                    .time(metric.timestamp)
                
                # Add metric-specific tags
                for key, value in metric.tags.items():
                    point = point.tag(key, str(value))
                
                points.append(point)
            
            self.write_api.write(bucket="metrics", record=points)
            
        except Exception as e:
            logger.error(f"Failed to send metrics to InfluxDB: {e}")
    
    async def generate_metrics(self):
        """Generate and send metrics for all entities."""
        timestamp = datetime.utcnow()
        all_metrics = []
        
        # Generate infrastructure metrics
        for entity_id in ["host-payment-01", "host-payment-02", "payment-db"]:
            metrics = self._generate_infrastructure_metrics(entity_id, timestamp)
            all_metrics.extend(metrics)
        
        # Generate application metrics
        for entity_id in ["payment-gateway", "fraud-detection", "settlement-service", "notification-service"]:
            metrics = self._generate_application_metrics(entity_id, timestamp)
            all_metrics.extend(metrics)
        
        # Generate business metrics
        business_metrics = self._generate_business_metrics(timestamp)
        all_metrics.extend(business_metrics)
        
        # Send metrics to destinations
        self._send_to_kafka(all_metrics)
        self._send_to_influxdb(all_metrics)
        
        logger.info(f"Generated {len(all_metrics)} metrics for {timestamp}")
        
        return all_metrics
    
    async def run_simulation(self):
        """Run the continuous data simulation."""
        logger.info("🚀 Starting Payment System Data Simulation")
        logger.info(f"Simulation interval: {self.simulation_interval} seconds")
        logger.info(f"Kafka brokers: {self.kafka_brokers}")
        logger.info(f"InfluxDB URL: {self.influxdb_url}")
        
        while True:
            try:
                # Simulate potential anomalies
                self._simulate_anomaly()
                
                # Generate and send metrics
                await self.generate_metrics()
                
                # Wait for next cycle
                await asyncio.sleep(self.simulation_interval)
                
            except KeyboardInterrupt:
                logger.info("🛑 Simulation stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in simulation: {e}")
                await asyncio.sleep(5)  # Wait before retrying
        
        # Cleanup
        self.kafka_producer.close()
        self.influxdb_client.close()


async def main():
    """Main entry point."""
    import os
    
    kafka_brokers = os.getenv("KAFKA_BROKERS", "localhost:9092")
    influxdb_url = os.getenv("INFLUXDB_URL", "http://localhost:8086")
    simulation_interval = int(os.getenv("SIMULATION_INTERVAL", "30"))
    
    simulator = PaymentSystemDataSimulator(
        kafka_brokers=kafka_brokers,
        influxdb_url=influxdb_url,
        simulation_interval=simulation_interval
    )
    
    await simulator.run_simulation()


if __name__ == "__main__":
    asyncio.run(main()) 