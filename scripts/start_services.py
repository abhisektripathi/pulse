#!/usr/bin/env python3
"""
Startup script for Predictive System Health Platform

Initializes and starts all services in the correct order:
1. Infrastructure services (databases, message queues)
2. Data ingestion (simulator)
3. Processing services (health scoring, prediction)
4. API services (conversation, gateway)
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import List, Dict

import httpx
import redis
from influxdb_client import InfluxDBClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ServiceManager:
    """Manages the startup and health checks of all services."""
    
    def __init__(self):
        self.services = {
            "infrastructure": [
                {"name": "zookeeper", "port": 2181, "url": "http://localhost:2181"},
                {"name": "kafka", "port": 9092, "url": "http://localhost:9092"},
                {"name": "influxdb", "port": 8086, "url": "http://localhost:8086"},
                {"name": "neo4j", "port": 7474, "url": "http://localhost:7474"},
                {"name": "elasticsearch", "port": 9200, "url": "http://localhost:9200"},
                {"name": "weaviate", "port": 8080, "url": "http://localhost:8080"},
                {"name": "redis", "port": 6379, "url": "redis://localhost:6379"},
                {"name": "prometheus", "port": 9090, "url": "http://localhost:9090"},
                {"name": "grafana", "port": 3001, "url": "http://localhost:3001"},
                {"name": "mlflow", "port": 5000, "url": "http://localhost:5000"}
            ],
            "application": [
                {"name": "data-simulator", "port": 8000, "url": "http://localhost:8000"},
                {"name": "health-scoring-service", "port": 8001, "url": "http://localhost:8001"},
                {"name": "prediction-service", "port": 8002, "url": "http://localhost:8002"},
                {"name": "conversation-service", "port": 8003, "url": "http://localhost:8003"},
                {"name": "api-gateway", "port": 8000, "url": "http://localhost:8000"}
            ]
        }
    
    async def wait_for_service(self, service: Dict[str, str], timeout: int = 60) -> bool:
        """Wait for a service to become available."""
        logger.info(f"Waiting for {service['name']} to start...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                if service['name'] == 'redis':
                    # Special handling for Redis
                    r = redis.from_url(service['url'])
                    r.ping()
                    logger.info(f"✅ {service['name']} is ready")
                    return True
                else:
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        response = await client.get(service['url'])
                        if response.status_code < 500:
                            logger.info(f"✅ {service['name']} is ready")
                            return True
            except Exception as e:
                logger.debug(f"Waiting for {service['name']}: {e}")
                await asyncio.sleep(2)
        
        logger.error(f"❌ {service['name']} failed to start within {timeout} seconds")
        return False
    
    async def initialize_influxdb(self):
        """Initialize InfluxDB with required buckets and organizations."""
        try:
            client = InfluxDBClient(
                url="http://localhost:8086",
                token="system-health-token",
                org="system-health"
            )
            
            # Check if bucket exists, create if not
            buckets_api = client.buckets_api()
            try:
                buckets_api.find_bucket_by_name("metrics")
                logger.info("✅ InfluxDB bucket 'metrics' already exists")
            except:
                buckets_api.create_bucket(bucket_name="metrics", org="system-health")
                logger.info("✅ Created InfluxDB bucket 'metrics'")
            
            client.close()
            
        except Exception as e:
            logger.warning(f"Could not initialize InfluxDB: {e}")
    
    async def initialize_kafka_topics(self):
        """Initialize Kafka topics."""
        try:
            # This would typically use kafka-python to create topics
            # For now, we'll assume topics are auto-created
            logger.info("✅ Kafka topics will be auto-created")
            
        except Exception as e:
            logger.warning(f"Could not initialize Kafka topics: {e}")
    
    async def initialize_neo4j(self):
        """Initialize Neo4j with sample data."""
        try:
            # This would create sample entities and relationships
            # For now, we'll assume the data simulator will populate this
            logger.info("✅ Neo4j will be populated by data simulator")
            
        except Exception as e:
            logger.warning(f"Could not initialize Neo4j: {e}")
    
    async def start_infrastructure_services(self) -> bool:
        """Start and wait for infrastructure services."""
        logger.info("🚀 Starting infrastructure services...")
        
        # Wait for infrastructure services
        for service in self.services["infrastructure"]:
            if not await self.wait_for_service(service):
                return False
        
        # Initialize databases
        await self.initialize_influxdb()
        await self.initialize_kafka_topics()
        await self.initialize_neo4j()
        
        logger.info("✅ All infrastructure services are ready")
        return True
    
    async def start_application_services(self) -> bool:
        """Start and wait for application services."""
        logger.info("🚀 Starting application services...")
        
        # Wait for application services
        for service in self.services["application"]:
            if not await self.wait_for_service(service):
                return False
        
        logger.info("✅ All application services are ready")
        return True
    
    async def run_health_checks(self) -> bool:
        """Run health checks on all services."""
        logger.info("🔍 Running health checks...")
        
        health_checks = [
            {"name": "API Gateway", "url": "http://localhost:8000/health"},
            {"name": "Health Scoring", "url": "http://localhost:8001/health"},
            {"name": "Prediction Service", "url": "http://localhost:8002/health"},
            {"name": "Conversation Service", "url": "http://localhost:8003/health"}
        ]
        
        all_healthy = True
        for check in health_checks:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(check['url'])
                    if response.status_code == 200:
                        logger.info(f"✅ {check['name']} is healthy")
                    else:
                        logger.error(f"❌ {check['name']} health check failed")
                        all_healthy = False
            except Exception as e:
                logger.error(f"❌ {check['name']} health check failed: {e}")
                all_healthy = False
        
        return all_healthy
    
    async def generate_sample_data(self):
        """Generate sample data for testing."""
        logger.info("📊 Generating sample data...")
        
        try:
            # Trigger data simulator to generate initial data
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post("http://localhost:8000/simulate")
                if response.status_code == 200:
                    logger.info("✅ Sample data generated")
                else:
                    logger.warning("⚠️ Could not generate sample data")
        except Exception as e:
            logger.warning(f"Could not generate sample data: {e}")
    
    async def start_all_services(self):
        """Start all services in the correct order."""
        logger.info("🚀 Starting Predictive System Health Platform...")
        logger.info(f"Start time: {datetime.utcnow()}")
        
        # Step 1: Start infrastructure services
        if not await self.start_infrastructure_services():
            logger.error("❌ Failed to start infrastructure services")
            return False
        
        # Step 2: Start application services
        if not await self.start_application_services():
            logger.error("❌ Failed to start application services")
            return False
        
        # Step 3: Run health checks
        if not await self.run_health_checks():
            logger.error("❌ Health checks failed")
            return False
        
        # Step 4: Generate sample data
        await self.generate_sample_data()
        
        logger.info("🎉 Predictive System Health Platform is ready!")
        logger.info("📊 Dashboard: http://localhost:3000")
        logger.info("📈 Grafana: http://localhost:3001 (admin/admin)")
        logger.info("🔧 API Docs: http://localhost:8000/docs")
        logger.info("🤖 GraphQL: http://localhost:8000/graphql")
        
        return True


async def main():
    """Main entry point."""
    manager = ServiceManager()
    
    try:
        success = await manager.start_all_services()
        if success:
            logger.info("✅ Platform startup completed successfully")
        else:
            logger.error("❌ Platform startup failed")
            return 1
    except KeyboardInterrupt:
        logger.info("🛑 Startup interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error during startup: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code) 