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

- **Natural Language Querying**: Convert plain English queries into accurate SQL/NoSQL commands.
- **Real-Time Database Connectivity**: Securely connect to relational databases such as Google Cloud SQL, AWS RDS, and Azure SQL.
- **Auto-Generated Visualizations**: Transform query results into dynamic Plotly-based charts.
- **CSV Export**: Download query outputs as CSV files for offline analysis.
- **Feedback Loop**: Users can refine charts and queries iteratively.
- **Monitoring & Logging**: Track model performance, latency, visualization success, and system uptime.

---

## 🏗️ Architecture

### Backend Flowchart:

![Backend Architecture](https://lucid.app/publicSegments/view/967cb8f0-2b53-499e-94b2-ee26074eb6f5/image.png)

### Frontend Flowchart:

![Frontend Flow](https://lucid.app/publicSegments/view/91d4e32f-6dbd-4131-9993-55b6a51896e3/image.png)

---

## 🛠️ Data Pipeline

The data pipeline prepares structured data for fine-tuning the NL to SQL model while ensuring data quality and fairness.

### Airflow ETL Pipeline

The **Apache Airflow** pipeline orchestrates:

| Component | Description |
|-----------|-------------|
| Data Ingestion | Ingests synthetic SQL datasets using GretelAI and custom scripts |
| SQL Validation | Validates SQL syntax with `sqlglot` |
| Duplicate Removal | Drops duplicate synthetic queries |
| **Bias Detection** | Detects underrepresentation in SQL types (JOIN, CTE, Aggregations) |
| **Bias Mitigation** | Generates synthetic SQL to rebalance dataset |
| Schema Validation | Ensures dataset follows strict structure |
| Notifications | Sends email alerts for bias or task failures |

---

### LangGraph Workflow

The application uses a **LangGraph workflow** with two primary nodes to manage database connections and schema indexing.

#### Workflow Nodes:

**1. `save_creds`**  
- **Purpose**: Securely stores database credentials  
- **Input**: User-provided database connection details  
- **Output**: Credentials persisted in the credentials database  
- **Process**: Validates and encrypts connection parameters before storage  

**2. `build_vector_store`**  
- **Purpose**: Indexes database schema for intelligent querying  
- **Process**:  
  1. Retrieves the database schema using stored credentials  
  2. Generates natural language descriptions of tables, columns, and relationships using LLM  
  3. Chunks and embeds the schema information  
  4. Stores embeddings in **ChromaDB** vector database for semantic search


---

## ⚙️ Model Training & Fine-Tuning

| Component | Details |
|-----------|----------|
| Base Model | `t5-large-lm-adapt` (Spider SQL) |
| Fine-Tuning | LoRA (Low-Rank Adaptation) |
| Dataset | Custom SQL dataset (GretelAI + synthetic queries) |
| Training Metadata | Includes query complexity & domain context |
| Versioning | Managed via **DVC** |
| Evaluation | Execution Accuracy (EX) + Logical Form Match (EM) |

---

## 📦 Prerequisites

### System Requirements

- **Operating System**: Linux, macOS, or Windows with WSL2
- **Python**: 3.10
- **RAM**: Minimum 8GB (16GB recommended for parallel processing)
- **CPU**: Multi-core processor (pipeline uses 75% of cores)
- **Disk Space**: ~5GB for dataset and generated files

```bash
# Python 3.10
python --version

# Docker (optional, for containerized Airflow)
docker --version

# Git (for version control)
git --version

# DVC (for data versioning)
dvc version
```

---

## 🚀 Quick Start

## Data Pipeline

### 1. Clone the Repository

```bash
git clone https://github.com/Northeastern-MLOps-Group-12/QueryHub.git
cd queryhub-pipeline/data-pipeline
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3.  Data Versioning with DVC

### Setup DVC

#### 3.1. Initialize DVC
```bash
cd data-pipeline
dvc init
```

#### 3.2. Configure Remote Storage

- **Google Cloud Storage (GCS)**:
```bash
dvc remote add -d myremote gs://my-bucket/data-pipeline
dvc remote modify myremote credentialpath ~/.config/gcloud/credentials.json
```

- **AWS S3**:
```bash
dvc remote add -d myremote s3://my-bucket/data-pipeline
dvc remote modify myremote access_key_id YOUR_ACCESS_KEY
dvc remote modify myremote secret_access_key YOUR_SECRET_KEY
```

#### 3.3. Track Data Directory
```bash
dvc add data/
git add data.dvc .gitignore
git commit -m "Track data with DVC"
```

#### 3.4. Push Data to Remote
```bash
dvc push
```

### 4. Configure Airflow

```bash
# ADD necessary folders
mkdir ./logs , ./plugins , ./config

# Initialize Airflow
docker compose run airflow-cli airflow config list

# Initialize Airflow DB
docker compose up airflow-init

# Start Docker Services
docker compose up -d
```

### 5. Configure SMTP for Email Alerts

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

### 6. Configure GCP Cloud SQL & Airflow Variables

1. **Create a Service Account**  
   - Navigate to [GCP Console → IAM & Admin → Service Accounts](https://console.cloud.google.com/iam-admin/serviceaccounts).  
   - Create a new service account with **Storage Admin** and **Cloud SQL Client** roles.  
   - Download the JSON key file.

2. **Prepare GCP Bucket**  
   - Create a bucket named:  
     ```
     text-to-sql-dataset
     ```

3. **Set up Airflow Connection**  
   - Open Airflow UI → **Admin → Connections**.  
   - Locate or create the connection: `google_cloud_default`.  
   - Set the **Key Path** to:  
     ```
     /opt/airflow/keys/gcp_sa.json
     ```  
   - Set the **Project ID** to your GCP project ID.

4. **Add Airflow Environment Variables**  
   - Go to Airflow → **Admin → Variables** and add:  
     ```python
      GCS_BUCKET_NAME = YOUR-BUCKET-NAME
      GCP_PROJECT_ID = YOUR-PROJECT-ID
     ```

> This ensures that your Airflow DAGs can securely access Cloud SQL and GCS for dataset processing.


### 7. Start Airflow

#### Option A: Docker Compose (Recommended)

```bash
# Start Airflow services
docker-compose up -d

# Check status
docker-compose ps
```

### 7. Access Airflow UI

Open your browser and navigate to:
```
http://localhost:8080
```

Login with:
- **Username**: airflow
- **Password**: airflow

### 8. Run the Pipeline

1. In the Airflow UI, find the DAG: `data_pipeline_with_synthetic_v1_schema_validation`
2. Toggle the DAG to **ON**
3. Click **Trigger DAG** to start execution
4. Monitor progress in the **Graph View** or **Gantt Chart**

### 9. DVC Workflow (After Pipeline Execution)

#### After Pipeline Execution
```bash
# Track new data files
dvc add data/

# Commit DVC files
git add data.dvc
git commit -m "Update dataset after pipeline run"

# Push data to remote
dvc push

# Push metadata to Git
git push
```

### 10. To Reproduce on Another Machine
```bash
# Clone repository
git clone https://github.com/Northeastern-MLOps-Group-12/QueryHub.git
cd queryhub-pipeline/data-pipeline

# Pull data from DVC remote
dvc pull

# Data is now available in data/
ls data/
```

---

## Database Connector

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| ⁠ DATABASE_URL ⁠ | PostgreSQL connection string (This should be the one given in the example) | ⁠ postgresql+pg8000://user:pass@host:port/db ⁠ |
| ⁠ LLM_API_KEY ⁠ | API key for LLM service | Your API key |
| ⁠ MODEL ⁠ | LLM provider type | ⁠ gemini ⁠ or ⁠ gpt ⁠ |
| ⁠ MODEL_NAME ⁠ | Specific model to use | ⁠ gemini-2.5-flash ⁠ |
| ⁠ EMBEDDING_MODEL ⁠ | Embedding model name | ⁠ text-embedding-004 ⁠ |
| ⁠ FRONTEND_ORIGIN ⁠ | Frontend URL for CORS | ⁠ http://localhost:5173 ⁠ |
| ⁠ LANGSMITH_API_KEY ⁠ | LangSmith API key for tracing | Your LangSmith key |
| ⁠ LANGSMITH_ENDPOINT ⁠ | LangSmith API endpoint | ⁠ https://api.smith.langchain.com ⁠ |
| ⁠ LANGSMITH_TRACING ⁠ | Enable LangSmith tracing | ⁠ true ⁠ or ⁠ false ⁠ |
| ⁠ LANGSMITH_PROJECT ⁠ | LangSmith project name | Your project name |

### 1. Clone the Repository

```bash
git clone https://github.com/Northeastern-MLOps-Group-12/QueryHub.git
cd queryhub
```

### 2. Build Docker Image

```bash
docker build -f ./backend/Dockerfile -t backend:latest .
```

### 3. Configure Network Access (GCP Cloud SQL)

To allow your application to connect to Cloud SQL, whitelist your IP address:

1. **Navigate to Cloud SQL in the GCP Console**
2. **Select your instance** and go to **Connections → Networking → Authorized Networks**
3. **Click Add Network** and provide:
   - **Network Name**: A descriptive name (e.g., "Local Development")
   - **IP Address**: Your machine's public IP address
   - **Note**: For testing purposes, you can use `0.0.0.0/0` _(not recommended for production)_
4. **Click Save** to apply the changes

### 4. Run Docker Container

```bash
docker run -p 8080:8080 -it \
  -e DATABASE_URL="$DATABASE_URL" \
  -e LLM_API_KEY="$LLM_API_KEY" \
  -e MODEL="$MODEL" \
  -e MODEL_NAME="$MODEL_NAME" \
  -e EMBEDDING_MODEL="$EMBEDDING_MODEL" \
  -e FRONTEND_ORIGIN="$FRONTEND_ORIGIN" \
  -e LANGSMITH_API_KEY="$LANGSMITH_API_KEY" \
  -e LANGSMITH_ENDPOINT="$LANGSMITH_ENDPOINT" \
  -e LANGSMITH_TRACING="$LANGSMITH_TRACING" \
  -e LANGSMITH_PROJECT="$LANGSMITH_PROJECT" \
  backend:latest
```

### 5. Test the Connection

Once the container is running, test the API by adding a database connection:

1. ⁠Open your browser and navigate to ⁠**http://localhost:8080/docs**
2. ⁠Locate the ⁠ **/connect/addConnection** ⁠ endpoint
3. Use the following request payload:

```json
{
  "engine": "postgres",
  "provider": "gcp",
  "config": {
    "user_id": "⁠user_id",
    "connection_name": "connection_name",
    "host": "db_host",
    "db_type": "postgres",
    "username": "db_user",
    "password": "db_password",
    "database": "db_name"
  }
}
```

**Configuration Parameters:**

| Parameter | Description |
|-----------|-------------|
| ⁠ user_id ⁠ | Unique identifier for the user (can be any string for testing) |
| ⁠ connection_name ⁠ | A unique name for this database connection |
| ⁠ db_host ⁠ | Public IP address of your Cloud SQL instance |
| ⁠ db_user ⁠ | Database username |
| ⁠ db_password ⁠ | Database password |
| ⁠ db_name ⁠ | Name of the database to connect to |

---

## 📂 Repository Structure

```
QueryHub/
├── .github/                                        # GitHub Actions workflows
│   └── workflows/
│       └── deploy-backend.yml                      # CI/CD workflow to deploy backend
│       └── run_tests.yml                           # CI/CD workflow to run tests
│
├── agents/                                         # LLM agent logic
│   ├── load_data_to_vector/
│   │   ├── graph.py                                # Defines workflow graph for agents
│   │   ├── load_creds_to_vectordb.py               # Saves DB creds & builds vector store
│   │   └── state.py                                # Pydantic models for agent state
│   ├── __init__.py                                 # Makes `agents` a package
│   └── base_agent.py                               # Base wrapper for chat models (Google/OpenAI)
│
├── backend/                                        # FastAPI backend
│   ├── models/
│   │   └── connector_request.py                    # Pydantic models for API requests
│   ├── Dockerfile                                  # Docker image for backend
│   ├── connectors_api.py                           # FastAPI routes for database connectors
│   ├── main.py                                     # Entry point for FastAPI server
│   └── requirements.txt                            # Backend Python dependencies
│
├── connectors/                                     # Database connector service
│   ├── engines/
│   │   ├── mysql/
│   │   │   ├── __init__.py                         # MySQL connector package
│   │   │   └── mysql_connector.py                  # MySQL-specific connector implementation
│   │   └── postgres/
│   │       ├── __init__.py                         # PostgreSQL connector package
│   │       └── postgres_connector.py               # PostgreSQL-specific connector implementation
│   ├── README.md                                   # Connectors module documentation
│   ├── __init__.py                                 # Makes `connectors` a package
│   ├── base_connector.py                           # Abstract base class for all connectors
│   └── connector.py                                # Factory to instantiate appropriate connector
│
├── data-pipeline/                                  # Data pipeline and ETL DAGs
│   ├── .dvc/
│   │   └── config                                  # DVC configuration
│   ├── dags/
│   │   ├── utils/
│   │   │   ├── DataGenData/
│   │   │   │   ├── DomainData/                     # Domain-specific synthetic data generators
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
│   │   │   │   │   └── SocialMedia.py
│   │   │   │   ├── Templates/                      # SQL query template definitions
│   │   │   │   │   ├── CTETemplates.py
│   │   │   │   │   ├── MultipleJoinsTemplates.py
│   │   │   │   │   ├── SETTemplates.py
│   │   │   │   │   ├── SubQueryTemplates.py
│   │   │   │   │   ├── WindowFunctionTemplates.py
│   │   │   │   │   └── __init__.py
│   │   │   │   └── __init__.py
│   │   │   ├── DataGenerator.py                    # Main synthetic data generator
│   │   │   ├── EmailContentGenerator.py            # Email notification utilities
│   │   │   ├── SQLValidator.py                     # Validates SQL using sqlglot
│   │   │   └── __init__.py
│   │   └── data_pipeline_dag.py                    # Main Airflow DAG
│   ├── tests/
│   │   └── test.py                                 # Pytest suite (45 tests)
│   ├── .dvcignore                                  # DVC ignore patterns
│   ├── Dockerfile                                  # Docker image for data pipeline
│   ├── README.md                                   # Data pipeline documentation
│   ├── data.dvc                                    # DVC tracking for data/
│   ├── docker-compose.yaml                         # Compose config for pipeline
│   └── requirements.txt                            # Python dependencies
│
├── databases/                                      # Database access layer
│   ├── cloudsql/
│   │   ├── models/
│   │   │   └── credentials.py                      # DB credential model
│   │   ├── __init__.py
│   │   ├── crud.py                                 # CRUD operations for cloud SQL
│   │   └── database.py                             # DB connection/session management
│   └── __init__.py
│
├── vectorstore/                                    # Vector database integration
│   ├── __init__.md
│   └── chroma_vector_store.py                      # ChromaDB embedding storage
│
├── .gitignore                                      # Git ignore rules
└── README.md                                       # Project overview and instructions
```

---

## ✅ Success Criteria

- 85%+ query-to-SQL accuracy
- 80%+ visualization coverage for chart-suitable queries
- <15s average response time per query
- GDPR/CCPA compliant data storage and user control
- High user satisfaction scores from feedback surveys