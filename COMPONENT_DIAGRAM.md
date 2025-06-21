# Predictive System Health Platform - Component Diagram

## Current Implementation Structure

```mermaid
graph TB
    subgraph "Frontend (React + TypeScript)"
        REACT_APP[React App<br/>Port 3000]
        NAV_COMP[Navigation Component]
        DASH_COMP[Dashboard Component]
        HEALTH_COMP[SystemHealth Component]
        PRED_COMP[Predictions Component]
        CONV_COMP[Conversation Component]
    end
    
    subgraph "Backend API (FastAPI + Python)"
        DEMO_SERVER[Demo Standalone Server<br/>Port 8000]
        API_ROUTES[API Routes]
        HEALTH_ENDPOINTS[Health Endpoints]
        PRED_ENDPOINTS[Prediction Endpoints]
        CHAT_ENDPOINTS[Chat Endpoints]
    end
    
    subgraph "Core Services"
        DATA_SIM_SVC[Data Simulator Service]
        HEALTH_SCORE_SVC[Health Scoring Service]
        PREDICTION_SVC[Prediction Service]
        CONVERSATION_SVC[Conversation Service]
    end
    
    subgraph "Data Models"
        METRICS_MODEL[Metrics Model]
        HEALTH_MODEL[Health Model]
        PREDICTION_MODEL[Prediction Model]
        CONVERSATION_MODEL[Conversation Model]
        ENTITIES_MODEL[Entities Model]
    end
    
    subgraph "ML Models"
        ISOLATION_FOREST[Isolation Forest<br/>Anomaly Detection]
        RANDOM_FOREST[Random Forest<br/>Failure Prediction]
        LSTM_MODEL[LSTM<br/>Time Series]
        TRANSFORMER_MODEL[Transformer<br/>NLP]
    end
    
    subgraph "Infrastructure (Docker)"
        DOCKER_COMPOSE[Docker Compose]
        KAFKA_CONTAINER[Kafka Container]
        INFLUX_CONTAINER[InfluxDB Container]
        NEO4J_CONTAINER[Neo4j Container]
        ES_CONTAINER[Elasticsearch Container]
        WEAVIATE_CONTAINER[Weaviate Container]
        REDIS_CONTAINER[Redis Container]
        PROMETHEUS_CONTAINER[Prometheus Container]
        GRAFANA_CONTAINER[Grafana Container]
        MLFLOW_CONTAINER[MLflow Container]
    end
    
    REACT_APP --> NAV_COMP
    REACT_APP --> DASH_COMP
    REACT_APP --> HEALTH_COMP
    REACT_APP --> PRED_COMP
    REACT_APP --> CONV_COMP
    
    REACT_APP --> DEMO_SERVER
    DEMO_SERVER --> API_ROUTES
    API_ROUTES --> HEALTH_ENDPOINTS
    API_ROUTES --> PRED_ENDPOINTS
    API_ROUTES --> CHAT_ENDPOINTS
    
    HEALTH_ENDPOINTS --> HEALTH_SCORE_SVC
    PRED_ENDPOINTS --> PREDICTION_SVC
    CHAT_ENDPOINTS --> CONVERSATION_SVC
    
    DATA_SIM_SVC --> METRICS_MODEL
    HEALTH_SCORE_SVC --> HEALTH_MODEL
    PREDICTION_SVC --> PREDICTION_MODEL
    CONVERSATION_SVC --> CONVERSATION_MODEL
    
    PREDICTION_SVC --> ISOLATION_FOREST
    PREDICTION_SVC --> RANDOM_FOREST
    PREDICTION_SVC --> LSTM_MODEL
    CONVERSATION_SVC --> TRANSFORMER_MODEL
    
    DOCKER_COMPOSE --> KAFKA_CONTAINER
    DOCKER_COMPOSE --> INFLUX_CONTAINER
    DOCKER_COMPOSE --> NEO4J_CONTAINER
    DOCKER_COMPOSE --> ES_CONTAINER
    DOCKER_COMPOSE --> WEAVIATE_CONTAINER
    DOCKER_COMPOSE --> REDIS_CONTAINER
    DOCKER_COMPOSE --> PROMETHEUS_CONTAINER
    DOCKER_COMPOSE --> GRAFANA_CONTAINER
    DOCKER_COMPOSE --> MLFLOW_CONTAINER
```

## File Structure Overview

```
pulse/
├── 📁 frontend/                    # React Frontend
│   ├── 📄 package.json            # Dependencies
│   ├── 📁 public/
│   │   └── 📄 index.html          # HTML Template
│   └── 📁 src/
│       ├── 📄 index.tsx           # App Entry Point
│       ├── 📄 App.tsx             # Main App Component
│       ├── 📄 index.css           # Global Styles
│       └── 📁 components/
│           ├── 📄 Navigation.tsx  # Sidebar Navigation
│           ├── 📄 Dashboard.tsx   # Main Dashboard
│           ├── 📄 SystemHealth.tsx # Health View
│           ├── 📄 Predictions.tsx # Predictions View
│           └── 📄 Conversation.tsx # Chat Interface
│
├── 📁 services/                   # Microservices
│   ├── 📁 api_gateway/           # API Gateway Service
│   ├── 📁 conversation/          # Conversational AI
│   ├── 📁 data_simulator/        # Data Generation
│   ├── 📁 health_scoring/        # Health Calculations
│   └── 📁 prediction/            # ML Predictions
│
├── 📁 src/                       # Shared Models
│   └── 📁 models/
│       ├── 📄 conversation.py    # Chat Models
│       ├── 📄 entities.py        # System Entities
│       ├── 📄 health.py          # Health Models
│       ├── 📄 metrics.py         # Metrics Models
│       └── 📄 predictions.py     # Prediction Models
│
├── 📄 demo_standalone.py         # Standalone Demo Server
├── 📄 docker-compose.yml         # Infrastructure
├── 📄 requirements.txt           # Python Dependencies
└── 📄 README.md                  # Documentation
```

## Service Communication Flow

```mermaid
sequenceDiagram
    participant UI as React Frontend
    participant API as API Gateway
    participant DS as Data Simulator
    participant HS as Health Scoring
    participant PRED as Prediction Engine
    participant CONV as Conversation AI
    participant DB as Data Stores
    
    UI->>API: GET /health/summary
    API->>HS: Request Health Data
    HS->>DB: Query Metrics
    DB-->>HS: Return Data
    HS-->>API: Health Summary
    API-->>UI: Display Health
    
    UI->>API: POST /predict
    API->>PRED: Request Prediction
    PRED->>DB: Get Historical Data
    DB-->>PRED: Return Data
    PRED->>PRED: Run ML Models
    PRED-->>API: Prediction Results
    API-->>UI: Show Predictions
    
    UI->>API: POST /chat
    API->>CONV: Send Message
    CONV->>DB: Query Knowledge Graph
    DB-->>CONV: Context Data
    CONV->>CONV: Process with NLP
    CONV-->>API: AI Response
    API-->>UI: Display Response
    
    Note over DS: Background Process
    DS->>DB: Generate Metrics
    DB-->>DS: Store Data
```

## Data Flow Architecture

```mermaid
flowchart TD
    subgraph "Data Generation"
        DS[Data Simulator]
        METRICS[System Metrics]
        EVENTS[Business Events]
    end
    
    subgraph "Processing"
        HS[Health Scoring]
        PRED[Prediction Engine]
        CONV[Conversation AI]
    end
    
    subgraph "Storage"
        INFLUX[(InfluxDB<br/>Time Series)]
        NEO4J[(Neo4j<br/>Knowledge Graph)]
        ES[(Elasticsearch<br/>Search)]
        WEAVIATE[(Weaviate<br/>Vectors)]
    end
    
    subgraph "Presentation"
        API[API Gateway]
        UI[React Dashboard]
    end
    
    DS --> METRICS
    DS --> EVENTS
    
    METRICS --> INFLUX
    EVENTS --> NEO4J
    
    INFLUX --> HS
    NEO4J --> HS
    INFLUX --> PRED
    NEO4J --> PRED
    ES --> CONV
    WEAVIATE --> CONV
    
    HS --> API
    PRED --> API
    CONV --> API
    
    API --> UI
```

## Technology Stack

### Frontend
- **React 18** - UI Framework
- **TypeScript** - Type Safety
- **Material-UI** - Component Library
- **React Router** - Navigation
- **Axios** - HTTP Client

### Backend
- **FastAPI** - API Framework
- **Python 3.13** - Runtime
- **Pydantic** - Data Validation
- **Uvicorn** - ASGI Server

### Machine Learning
- **Scikit-learn** - ML Algorithms
- **NumPy/Pandas** - Data Processing
- **MLflow** - Model Management
- **OpenAI** - NLP Models

### Data Storage
- **InfluxDB** - Time Series Data
- **Neo4j** - Knowledge Graph
- **Elasticsearch** - Search & Analytics
- **Weaviate** - Vector Database
- **Redis** - Caching

### Infrastructure
- **Docker** - Containerization
- **Docker Compose** - Orchestration
- **Kafka** - Event Streaming
- **Prometheus** - Metrics
- **Grafana** - Visualization

## Deployment Architecture

```mermaid
graph TB
    subgraph "Development"
        DEV_FRONTEND[React Dev Server<br/>Port 3000]
        DEV_BACKEND[FastAPI Dev Server<br/>Port 8000]
    end
    
    subgraph "Production"
        PROD_LB[Load Balancer]
        PROD_FRONTEND[React Build<br/>Static Files]
        PROD_BACKEND[FastAPI<br/>Multiple Instances]
    end
    
    subgraph "Infrastructure"
        K8S[Kubernetes Cluster]
        DOCKER[Docker Containers]
        CLOUD[Cloud Platform]
    end
    
    DEV_FRONTEND --> DEV_BACKEND
    PROD_LB --> PROD_FRONTEND
    PROD_LB --> PROD_BACKEND
    PROD_BACKEND --> K8S
    K8S --> DOCKER
    DOCKER --> CLOUD
```

## Security Architecture

```mermaid
graph TB
    subgraph "Client Security"
        HTTPS[HTTPS/TLS]
        CORS[CORS Policy]
        CSP[Content Security Policy]
    end
    
    subgraph "API Security"
        AUTH[Authentication]
        AUTHZ[Authorization]
        RATE_LIMIT[Rate Limiting]
        VALIDATION[Input Validation]
    end
    
    subgraph "Data Security"
        ENCRYPT[Encryption at Rest]
        TRANSPORT[TLS in Transit]
        AUDIT[Audit Logging]
        BACKUP[Backup & Recovery]
    end
    
    HTTPS --> AUTH
    CORS --> AUTHZ
    CSP --> RATE_LIMIT
    AUTH --> VALIDATION
    AUTHZ --> ENCRYPT
    RATE_LIMIT --> TRANSPORT
    VALIDATION --> AUDIT
    ENCRYPT --> BACKUP
```

This component diagram provides a comprehensive view of the current implementation, showing how all the pieces fit together in the Predictive System Health Platform. 