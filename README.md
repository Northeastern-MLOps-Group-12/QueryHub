# QueryHub - RAG-Based Text-to-SQL System

QueryHub is a Retrieval-Augmented Generation (RAG)-based text-to-SQL platform that enables users to securely connect cloud-hosted SQL datasets and interact with them via natural language queries. It automatically generates SQL, executes queries, and returns results as shareable datasets or interactive visualizations.

---

## 👥 Team Members

- Jay Vipin Jajoo
- Rohan Ojha
- Rahul Reddy Mandadi
- Abhinav Gadgil
- Ved Dipak Deore
- Ashwin Khairnar

---

## 🚀 Features

- **Natural Language Querying**: Convert plain English queries into accurate SQL/NoSQL commands
- **Real-Time Database Connectivity**: Securely connect to relational databases such as Google Cloud SQL, AWS RDS, and Azure SQL
- **Auto-Generated Visualizations**: Transform query results into dynamic Plotly-based charts
- **CSV Export**: Download query outputs as CSV files for offline analysis
- **Feedback Loop**: Users can refine charts and queries iteratively
- **Monitoring & Logging**: Track model performance, latency, visualization success, and system uptime

---

## 📋 Table of Contents

1. [Quick Start](#-quick-start)
2. [Environment Variables](#-environment-variables)
3. [Data Pipeline Setup](#-data-pipeline-setup)
4. [Backend Setup](#-backend-setup)
5. [Frontend Setup](#-frontend-setup)
6. [Model Training Pipeline](#-model-training-and-evaluation-pipeline)
7. [CI/CD & Deployment Scripts](#-cicd--deployment-scripts)
8. [Architecture](#-architecture)
9. [Repository Structure](#-repository-structure)

---

## 🏁 Quick Start

### Prerequisites

- **Operating System**: Linux, macOS, or Windows with WSL2
- **Python**: 3.10
- **Docker & Docker Compose**
- **Git**
- **DVC** (for data versioning)
- **RAM**: Minimum 8GB (16GB recommended for parallel processing)
- **CPU**: Multi-core processor (pipeline uses 75% of cores)
- **Disk Space**: ~10GB for dataset and generated files
- **Google Cloud Platform** account (with appropriate permissions)

```bash
# Verify installations
python --version
docker --version
git --version
dvc version
```

### 1. Clone the Repository

```bash
git clone https://github.com/Northeastern-MLOps-Group-12/QueryHub.git
cd QueryHub
```

---

### 2. Environment Variables

Create a `.env` file in the `backend/` directory with the following variables:

```bash
# Database Configuration
DATABASE_URL=postgresql+pg8000://user:password@host:port/database

# LLM Configuration
LLM_API_KEY=your_llm_api_key
MODEL=gemini
MODEL_NAME=gemini-2.5-flash
EMBD_MODEL_PROVIDER=google
EMBEDDING_MODEL=text-embedding-004

# Frontend Configuration
FRONTEND_ORIGIN=http://localhost:5173

# GCP Configuration
PROJECT_ID=your_gcp_project_id
GCS_BUCKET_NAME=your_gcs_bucket
GCS_VECTORSTORE_BUCKET_NAME=your_vectorstore_bucket
FIREBASE_DATABASE_ID=your_firebase_db_id

# Authentication
ACCESS_TOKEN_EXPIRE_MINUTES=30
SECRET_KEY=your_secret_key
ALGORITHM=HS256

# Application Mode
MODE=development

# LangSmith Tracing (Optional)
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=your_project_name

# OpenAI (if using GPT models)
OPENAI_API_KEY=your_openai_api_key
```

### 3. Authenticate Google Cloud
```bash
# Login to Google Cloud
gcloud auth login

# Set your project
gcloud config set project your_gcp_project_id

# Authenticate application default credentials (for local development)
gcloud auth application-default login
```

### 4. Start Backend
```bash
cd backend
docker compose up
```
Navigate to `http://localhost:8081` to access the backend Swagger.

### 5. Start Data Pipeline
```bash
cd data-pipeline
docker compose up
```
Navigate to `http://localhost:8080` to access the data pipelines.
Navigate to `http://localhost:3001` to access the Grafana monitoring dashboard.

### 6. Start Frontend
```bash
cd frontend
docker compose up
```
Navigate to `http://localhost:5173` to access the frontend interface.

---

## 🔄 Data Pipeline Setup

The data pipeline uses Apache Airflow to orchestrate ETL workflows, model training, and retraining DAGs.

### Data Pipeline Variables
```json
{
  "alert_email": "<mail_to_send_alerts>",
  "gcp_evaluation_output_csv": "gs://<bucket_name>/output_data/",
  "gcp_processed_data_path": "gs://<bucket_name>/processed_datasets",
  "gcp_project": "queryhub-473901",
  "gcp_region": "us-east1",
  "gcs_bias_and_syntax_validation_output": "gs://<bucket_name>/bias_and_syntax_validation/",
  "gcs_bucket_name": "gs://<bucket_name>",
  "gcs_registered_models": "gs://<bucket_name>/registered_models",
  "gcs_staging_bucket": "gs://<bucket_name>/staging_bucket",
  "num_train_epochs": "1",
  "service_account": "<your_service_account_email>",
  "serving_container_image_uri": "<your_image_on_artifiact_registry>",
  "train_samples": "1",
  "val_samples": "1",
  "vertex_ai_eval_gpu_type": "NVIDIA_L4",
  "vertex_ai_eval_machine_type": "g2-standard-4",
  "vertex_ai_train_gpu_type": "NVIDIA_L4",
  "vertex_ai_training_image_uri": "<your_image_for_training_n_eval_scripts>",
  "vertex_ai_train_machine_type": "g2-standard-4"
}
```

### DAGs Overview

| DAG | Description |
|-----|-------------|
| **Data Pipeline DAG** | Handles data ingestion, SQL validation, duplicate removal, bias detection/mitigation, and schema validation |
| **Retraining DAG** | Monitors data drift based on SQL complexity distribution and triggers model retraining when thresholds are exceeded |

### Airflow ETL Pipeline Components

| Component | Description |
|-----------|-------------|
| Data Ingestion | Ingests synthetic SQL datasets using GretelAI and custom scripts |
| SQL Validation | Validates SQL syntax with `sqlglot` |
| Duplicate Removal | Drops duplicate synthetic queries |
| **Bias Detection** | Detects underrepresentation in SQL types (JOIN, CTE, Aggregations) |
| **Bias Mitigation** | Generates synthetic SQL to rebalance dataset |
| Schema Validation | Ensures dataset follows strict structure |
| Notifications | Sends email alerts for bias or task failures |

### Setup Instructions

#### 1. Navigate to Data Pipeline Directory

```bash
cd data-pipeline
```

#### 2. Set Up Python Environment (Optional for local development)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### 3. Data Versioning with DVC

##### Initialize DVC
```bash
dvc init
```

##### Configure Remote Storage (GCS)
```bash
dvc remote add -d myremote gs://my-bucket/data-pipeline
dvc remote modify myremote credentialpath ~/.config/gcloud/credentials.json
```

##### Configure Remote Storage (AWS S3 - Alternative)
```bash
dvc remote add -d myremote s3://my-bucket/data-pipeline
dvc remote modify myremote access_key_id YOUR_ACCESS_KEY
dvc remote modify myremote secret_access_key YOUR_SECRET_KEY
```

##### Track Data Directory
```bash
dvc add data/
git add data.dvc .gitignore
git commit -m "Track data with DVC"
dvc push
```

#### 4. Configure Airflow

```bash
# Create necessary folders
mkdir ./logs ./plugins ./config

# Initialize Airflow
docker compose run airflow-cli airflow config list

# Initialize Airflow DB
docker compose up airflow-init

# Start Docker Services
docker compose up -d
```

#### 5. Configure SMTP for Email Alerts

Edit `docker-compose.yaml`:

```yaml
environment:
  AIRFLOW__SMTP__SMTP_HOST: smtp.gmail.com
  AIRFLOW__SMTP__SMTP_PORT: 587
  AIRFLOW__SMTP__SMTP_USER: your-email@gmail.com
  AIRFLOW__SMTP__SMTP_PASSWORD: your-app-password
  AIRFLOW__SMTP__SMTP_MAIL_FROM: your-email@gmail.com
```

**Note**: For Gmail, use an [App Password](https://support.google.com/accounts/answer/185833).

#### 6. Access Airflow UI

Navigate to `http://localhost:8080`

Login credentials:
- **Username**: airflow
- **Password**: airflow

#### 7. Configure Airflow Variables

Go to **Admin → Variables** and import `data-pipeline/variables.json`, or set manually:

| Variable | Description |
|----------|-------------|
| `alert_email` | Email address for pipeline failure alerts |
| `gcp_evaluation_output_csv` | GCS path for evaluation output CSVs |
| `gcp_processed_data_path` | GCS path for saving processed data |
| `gcp_project` | GCP Project ID |
| `gcp_region` | GCP Region (e.g., `us-central1`) |
| `gcs_bias_and_syntax_validation_output` | GCS path for bias/syntax validation outputs |
| `gcs_bucket_name` | GCS bucket name (use `gs://<bucket_name>` format) |
| `gcs_registered_models` | GCS path to trained model files |
| `gcs_staging_bucket` | Vertex AI staging bucket |
| `num_train_epochs` | Number of training epochs |
| `service_account` | GCP Service Account email |
| `serving_container_image_uri` | Docker image URI for model serving |
| `train_samples` | Number of training samples |
| `val_samples` | Number of validation samples |
| `vertex_ai_eval_gpu_type` | GPU type for evaluation (e.g., `NVIDIA_TESLA_T4`) |
| `vertex_ai_eval_machine_type` | Machine type for evaluation |
| `vertex_ai_train_gpu_type` | GPU type for training |
| `vertex_ai_training_image_uri` | Docker image URI for training/eval scripts |
| `vertex_ai_train_machine_type` | Machine type for training |

#### 8. Run the Pipeline

1. In the Airflow UI, find the DAG: `data_pipeline_with_synthetic_v1_schema_validation`
2. Toggle the DAG to **ON**
3. Click **Trigger DAG** to start execution
4. Monitor progress in the **Graph View** or **Gantt Chart**

#### 9. DVC Workflow (After Pipeline Execution)

```bash
# Track new data files
dvc add data/

# Commit DVC files
git add data.dvc
git commit -m "Update dataset after pipeline run"

# Push data to remote
dvc push
git push
```

#### 10. Reproduce on Another Machine

```bash
git clone https://github.com/Northeastern-MLOps-Group-12/QueryHub.git
cd QueryHub/data-pipeline
dvc pull
ls data/
```

---

## 🖥️ Backend Setup

The backend is built with FastAPI and provides REST APIs for database connections, query execution, and visualization generation.

### Setup Instructions

```bash
cd backend
docker compose up -d
```

### Access the API

Navigate to `http://localhost:8081` to open the Swagger UI with all available API endpoints.

### Key API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/connect/addConnection` | POST | Add a new database connection |
| `/connect/connections` | GET | List all connections |
| `/query/execute` | POST | Execute a natural language query |
| `/health` | GET | Health check endpoint |

### LangGraph Workflow

The application uses a **LangGraph workflow** with two primary nodes:

**1. `save_creds`**
- **Purpose**: Securely stores database credentials
- **Process**: Validates and encrypts connection parameters before storage

**2. `build_vector_store`**
- **Purpose**: Indexes database schema for intelligent querying
- **Process**:
  1. Retrieves the database schema using stored credentials
  2. Generates natural language descriptions of tables, columns, and relationships using LLM
  3. Chunks and embeds the schema information
  4. Stores embeddings in **ChromaDB** vector database for semantic search

---

## 🎨 Frontend Setup

The frontend is built with React/TypeScript and provides an intuitive interface for database querying and visualization.

### Setup Instructions

```bash
cd frontend
docker compose up -d
```

### Access the Application

Navigate to `http://localhost:5173` to access the QueryHub interface.

---

## ⚙️ Model Training and Evaluation Pipeline

### Model Overview

| Component | Details |
|-----------|----------|
| Base Model | `t5-large-lm-adapt` (Spider SQL) |
| Fine-Tuning | LoRA (Low-Rank Adaptation) |
| Dataset | Custom SQL dataset (GretelAI + synthetic queries) |
| Training Metadata | Includes query complexity & domain context |
| Versioning | Managed via **DVC** |
| Evaluation | Execution Accuracy (EX) + Logical Form Match (EM) |

### Required GCP Resources

1. **GCS Buckets**: Training/Testing/Validation data, Model Artifacts, Prediction results, Evaluation results, Validation outputs
2. **Vertex AI**: GPU Quota (ideally 2), Enabled API, Service account with permissions, Model Registry enabled

### Build & Push Vertex AI Training Image

From `model_fine_tuning/vertex_ai_image` directory:

```bash
# Build the Docker Image
docker build --platform linux/amd64 -t your-region-docker.pkg.dev/your-project/your-artifact/your-image-name:your-image-tag .

# Push to Artifact Registry
docker push your-region-docker.pkg.dev/your-project/your-artifact/your-image-name:your-image-tag
```

### Trigger Training Pipeline

**From Airflow UI:**
1. Navigate to DAGs
2. Locate `vertex_ai_model_training_pipeline`
3. Click **Trigger DAG**

**From CLI:**
```bash
airflow dags trigger vertex_ai_model_training_pipeline
```

---

## 🚢 CI/CD & Deployment Scripts

QueryHub uses GitHub Actions for continuous integration and deployment. The following workflows are configured in `.github/workflows/`:

| Workflow | File | Description |
|----------|------|-------------|
| **Airflow Deployment** | `airflow-cicd.yml` | Deploys Data and Model Training pipeline to VM |
| **Verte AI Scripts** | `build-vertex-ai-image.yml` | Pushes Model Training and Evaluation Script to Artifact Registry |
| **Deploy Backend** | `deploy-backend.yml` | Deploys the Cackend to Google Cloud Run |
| **Deploy Monitoring** | `deploy-monitoring.yml` | Pushes Monitoring code to VM and deploys Grafana/Prometheus dashboards |
| **Deploy Frontend** | `frontend-deploy.yml` | Deploys the Frontend to Google Cloud Run |
| **Run Tests** | `run_tests.yml` | Executes unit tests on every Push and PR |
| **Trigger Data Pipeline** | `trigger-data-pipeline.yml` | Triggers Data Pipeline DAG daily at 10 AM EST |

### Deployment Architecture

- **Backend**: Deployed to Cloud Run (containerized FastAPI service)
- **Frontend**: Deployed to Cloud Run (containerized React app)
- **Data Pipeline**: Runs on a GCP VM with Docker (Airflow + DAGs)
- **Model Training**: Runs on Vertex AI
- **Monitoring**: Grafana and Prometheus deployed on a dedicated VM

### Verifying Deployment

To verify the Airflow VM is running correctly:

```bash
# SSH into your VM
gcloud compute ssh <vm-name> --zone <zone>

# Check running containers
docker ps
```

You should see Airflow webserver, scheduler, and worker containers running.

---

## 🏗️ Architecture

### Backend Flowchart

![Backend Architecture](https://lucid.app/publicSegments/view/967cb8f0-2b53-499e-94b2-ee26074eb6f5/image.png)

### Frontend Flowchart

![Frontend Flow](https://lucid.app/publicSegments/view/91d4e32f-6dbd-4131-9993-55b6a51896e3/image.png)

### Deployment Architecture

![Overall Architecture](https://lucid.app/publicSegments/view/3bb3a15f-5945-44b9-8498-473e13a5fc95/image.png)

---

## 📂 Repository Structure

```
QueryHub/
├── .github/workflows/                              # GitHub Actions Workflows
│   ├── airflow-cicd.yml                            # Deploy airflow
│   ├── build-vertex-ai-image.yml                   # Push training image to artifact registry
│   ├── deploy-backend.yml                          # Deploy backend
│   ├── deploy-monitoring.yml                       # Deploy monitoring
│   ├── frontend-deploy.yml                         # Deploy frontend
│   ├── run_tests.yml                               # CI/CD workflow to run tests
│   └── trigger-data-pipeline.yml                   # Triggers data pipeline daily
│
├── .vscode/
│   └── settings.json                               # Internal Settings
│
├── agents/                                         # LLM agent logic
│   ├── load_data_to_vector/
│   │   ├── graph.py                                # Defines workflow graph for agents
│   │   ├── load_creds_to_vectordb.py               # Saves DB creds & builds vector store
│   │   └── state.py                                # Pydantic models for agent state
│   ├── nl_to_data_viz/
│   │   ├── database_selector.py
│   │   ├── generate_sql_query.py
│   │   ├── graph.py
│   │   ├── guardrails.py
│   │   ├── query_result_saver.py
│   │   ├── sql_complexity_analyzer.py
│   │   ├── sql_runner.py
│   │   ├── state.py
│   │   └── test_guardrails.py
│   ├── update_data_in_vector/
│   │   ├── graph.py
│   │   ├── state.py
│   │   └── update_creds_in_vectordb.py
│   ├── __init__.py
│   └── base_agent.py                               # Base wrapper for chat models
│
├── backend/                                        # FastAPI backend
│   ├── models/
│   │   ├── chat_model.py 
│   │   ├── chat_request.py 
│   │   ├── connector_request.py                    # Pydantic models for API requests
│   │   ├── signin_request.py 
│   │   ├── signin_response.py 
│   │   ├── signup_request.py 
│   │   ├── tokendata.py
│   │   └── user_response.py
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── monitoring.json
│   ├── utils/
│   │   ├── __init__.py 
│   │   ├── agent_utils.py 
│   │   ├── chat_utils.py
│   │   ├── connectors_api_utils.py 
│   │   ├── user_api_utils.py 
│   │   ├── user_security.py 
│   │   └── vectorstore_gcs.py
│   ├── __init__
│   ├── chat_api.py
│   ├── connectors_api.py     # FastAPI routes for database connectors
│   ├── docker-compose.yml
│   ├── Dockerfile
│   ├── main.py    
│   ├── prometheus.yml
│   ├── requirements.txt
│   └── user_api.py
│
├── connectors/                                     # Database connector service
│   ├── engines/
│   │   ├── mysql/
│   │   ├── __init__.py
│   │   │   └── mysql_connector.py
│   │   └── postgres/
│   │       ├── __init__.py
│   │       └── postgres_connector.py
│   ├── __init__.py                           
│   ├── base_connector.py                           # Abstract base class for connectors
│   ├── connector.py                                # Factory to instantiate connectors
│   └─ README.md                               
│
├── data-pipeline/                                  # Data pipeline and ETL DAGs
│   ├── .dvc/
│   │   ├── config                          
│   ├── dags/
│   │   ├── model_scripts/                          # Model Training and Evaluation
│   │   │   ├── bias_detection.py
│   │   │   ├── dag_experiment_utils.py
│   │   │   ├── model_deployment.py
│   │   │   ├── model_eval_job_launcher.py
│   │   │   ├── README.md
│   │   │   ├── retrain_model.py
│   │   │   ├── syntax_validation.py
│   │   │   └── train_utils.py
│   │   ├── utils/
│   │   │   ├── DataGenData/
│   │   │   │   ├── DomainData/                     # Domain-specific generators
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── Ecommerce.py
│   │   │   │   │   ├── Education.py
│   │   │   │   │   ├── Finance.py
│   │   │   │   │   ├── Gaming.py
│   │   │   │   │   ├── Healthcare.py
│   │   │   │   │   ├── Hospitality.py
│   │   │   │   │   ├── Logistics.py
│   │   │   │   │   ├── Manufacturing.py
│   │   │   │   │   ├── RealEstate.py
│   │   │   │   │   ├── Retail.py
│   │   │   │   │   ├── SocialMedia.py
│   │   │   │   ├── Templates/                      # SQL query templates
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── CTETemplates.py
│   │   │   │   │   ├── MultipleJoinsTemplates.py
│   │   │   │   │   ├── SETTemplates.py
│   │   │   │   │   ├── SubQueryTemplates.py
│   │   │   │   │   ├── WindowFunctionTemplates.py
│   │   │   │   └── __init__.py
│   │   │   ├── DataGenerator.py
│   │   │   ├── EmailContentGenerator.py
│   │   │   ├── SQLValidator.py
│   │   │   └── test_utils.py
│   │   ├── data_pipeline_dag.py                    # Main Airflow DAG
│   │   └── train_model_and_save.py                 # Model Training DAG
│   ├── scripts/
│   │   └── docker-entrypoint.sh               
│   ├── tests/
│   │   ├── model_training_tests.py             
│   │   └── test.py    
│   ├── .dvcignore
│   ├── data.dvc
│   ├── docker-compose.yaml
│   ├── Dockerfile
│   └── requirements.txt
│
├── databases/                                      # Database access layer
│   ├── cloudsql/
│   │   ├── models         
│   │   │   ├── credentials.py
│   │   │   ├── user.py
│   │   ├── __init__.py
│   │   ├── crud.py
│   │   └── database.py
│   └── __init__.py
│
├── frontend/                                       # React frontend application
│   ├── public/
│   │   └── logo.png                                # Application logo
│   ├── src/
│   │   ├── account/                                # Authentication pages
│   │   │   ├── index.tsx
│   │   │   ├── SignIn.tsx
│   │   │   └── SignUp.tsx
│   │   ├── assets/
│   │   │   └── default-avatar.png                  # Default user avatar
│   │   ├── chat-interface/                         # Chat UI components
│   │   │   ├── index.tsx
│   │   │   ├── NewChatModal.css
│   │   │   └── NewChatModal.tsx
│   │   ├── components/
│   │   │   └── ProtectedRoute.tsx                  # Route authentication wrapper
│   │   ├── data/                                   # Static data and content
│   │   │   ├── dbOptions.tsx
│   │   │   └── homeContent.tsx
│   │   ├── database/                               # Database connection components
│   │   │   ├── ConnectedDatabases.tsx
│   │   │   ├── DatabaseConnection.tsx
│   │   │   ├── DatabaseDescription.tsx
│   │   │   ├── DatabaseEditor.tsx
│   │   │   └── index.tsx
│   │   ├── home/
│   │   │   └── index.tsx                           # Landing/home page
│   │   ├── hooks/                                  # Custom React hooks
│   │   │   ├── AuthProvider.tsx
│   │   │   └── useAuth.tsx
│   │   ├── services/                               # API service layer
│   │   │   ├── api.ts                              # Base API configuration
│   │   │   ├── authService.tsx                     # Authentication API calls
│   │   │   ├── chatService.tsx                     # Chat-related API calls
│   │   │   └── databaseService.tsx                 # Database management API calls
│   │   ├── App.css                                 # Global app styles
│   │   ├── App.tsx                                 # Main app component
│   │   ├── Footer.tsx                              # Footer component
│   │   ├── index.css                               # Root styles
│   │   ├── main.tsx                                # Application entry point
│   │   └── Navbar.tsx                              # Navigation bar component
│   ├── .dockerignore                               # Docker ignore rules
│   ├── .gitignore                                  # Git ignore rules
│   ├── docker-compose.yml                          # Docker configuration for frontend
│   ├── Dockerfile                                  # Docker image definition
│   ├── eslint.config.js                            # ESLint configuration
│   ├── index.html                                  # HTML entry point
│   ├── nginx.conf                                  # Nginx server configuration
│   ├── package-lock.json                           # Locked dependencies
│   ├── package.json                                # NPM dependencies and scripts
│   ├── README.md                                   # Frontend documentation
│   ├── tsconfig.app.json                           # TypeScript config for app
│   ├── tsconfig.json                               # Base TypeScript configuration
│   ├── tsconfig.node.json                          # TypeScript config for Node
│   └── vite.config.ts                              # Vite build configuration
│
├── model_fine_tuning/                              # Fine-tuning experiments and research
│   ├── FT_NoteBooks/                               # Jupyter notebooks for fine-tuning
│   │   ├── QH_FT_Sensitivity.ipynb                 # Sensitivity analysis experiments
│   │   ├── QH_FT_T1.ipynb                          # Fine-tuning trial 1
│   │   ├── QH_FT_T2.ipynb                          # Fine-tuning trial 2
│   │   └── QH_FT_T3.ipynb                          # Fine-tuning trial 3
│   └── vertex_ai_image/                            # Vertex AI custom training
│       ├── Dockerfile                              # Custom training container
│       ├── experiment_utils.py                     # Experiment utilities and helpers
│       ├── model_eval.py                           # Model evaluation scripts
│       ├── README.md                               # Vertex AI training documentation
│       ├── requirements.txt                        # Python dependencies
│       └── train.py                                # Training script
│
├── tests/                                          # Test suite
│   ├── agents/                                     # Agent component tests
│   │   ├── test_base_agent.py                      # Base agent functionality tests
│   │   ├── test_graph.py                           # Graph agent tests
│   │   ├── test_load_creds_to_vectordb.py          # Credential loading tests
│   │   └── test_state.py                           # State management tests
│   ├── backend/                                    # Backend API tests
│   │   └── test_connectors_api.py                  # Connector API endpoint tests
│   ├── connectors/    
│   │   └── test_connectors.py                      # Database connector tests
│   ├── conftest.py                                 # Pytest configuration and fixtures
│   └── requirements.txt                            # Test dependencies
├── vectorstore/                                    # ChromaDB vector store integration
│   ├── __init__.py
│   └── chroma_vector_store.py
├── .gitignore
└── README.md
```

---

## 🎥 Video Demo
- [QueryHub Demo](https://northeastern-my.sharepoint.com/:f:/g/personal/deore_v_northeastern_edu/IgASnWrcp33oRJ9C-89fnrx3AcSS7gJxhJezJU_nV01W4UA?e=puKjrH)

---

## 📄 Documentation

- [Scoping Document](https://docs.google.com/document/d/1Iblflv-p4wUgzQoSpWiBj2JXwwROFsZgEVZwYK-Z9Hs/edit?usp=sharing)
- [Model Development Document](https://docs.google.com/document/d/1D5nyl2Pb45JF5NJGTn9cwV6xBmpFTchtbwbQRXK_C5E/edit?usp=sharing)
- [Data Pipeline, Errors + Graceful Failures, User Needs + Defining Success, Data Drift, Model Retraining, Monitoring](https://northeastern-my.sharepoint.com/:f:/g/personal/deore_v_northeastern_edu/IgASnWrcp33oRJ9C-89fnrx3AcSS7gJxhJezJU_nV01W4UA?e=puKjrH)

---

## 📝 License

This project is developed as part of the MLOps curriculum at Northeastern University.