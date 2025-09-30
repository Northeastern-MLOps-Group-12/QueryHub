# QueryHub - RAG-Based Text-to-SQL System

QueryHub is a Retrieval-Augmented Generation (RAG)-based text-to-SQL platform that enables users to securely connect cloud-hosted SQL datasets and interact with them via natural language queries. It automatically generates SQL, executes queries, and returns results as shareable datasets or interactive visualizations.

---

## 🚀 Features

- **Natural Language Querying**: Convert plain English queries into accurate SQL/NoSQL commands.
- **Real-Time Database Connectivity**: Securely connect to relational databases such as Google Cloud SQL, AWS RDS, and Azure SQL.
- **Auto-Generated Visualizations**: Transform query results into dynamic Plotly-based charts.
- **CSV Export**: Download query outputs as CSV files for offline analysis.
- **Feedback Loop**: Users can refine charts and queries iteratively.
- **Monitoring & Logging**: Track model performance, latency, visualization success, and system uptime.

---

## 👥 Team Members

- Jay Vipin Jajoo
- Rohan Ojha
- Rahul Reddy Mandadi
- Abhinav Gadgil
- Ved Dipak Deore
- Ashwin Khairnar

---

## 📂 Repository Structure

```
QueryHub/
├── .github/ # CI/CD workflows
│ └── workflows/
│ ├── backend-ci.yml
│ └── frontend-ci.yml
│
├── backend/ # Backend (FastAPI + LLM integration)
│ ├── app-server/
│ │ ├── api/ # API routes and controllers
│ │ ├── services/ # Business logic
│ │ ├── connectors/ # DB connection modules
│ │ ├── llm_clients/ # LLM API integrations
│ │ ├── llm-evaluator/ # LLM evaluation logic
│ │ ├── airflow/ # Airflow pipelines
│ │ └── utils/ # Shared utilities
│ ├── tests/ # Backend tests
│ ├── Dockerfile
│ ├── requirements.txt
│ └── README.md
│
├── frontend/ # Frontend (React + TypeScript)
│ ├── public/
│ ├── src/
│ │ ├── assets/
│ │ ├── components/
│ │ ├── pages/
│ │ ├── services/
│ │ ├── utils/
│ │ └── App.tsx
│ ├── Dockerfile
│ ├── package.json
│ ├── tsconfig.json
│ └── README.md
│
├── docker-compose.yml # Run frontend + backend together
├── .gitignore
└── README.md # Main project overview
```

---

## 🏗️ Architecture

### Backend Flowchart:

![Backend Architecture](https://lucid.app/publicSegments/view/967cb8f0-2b53-499e-94b2-ee26074eb6f5/image.png)

### Frontend Flowchart:

![Frontend Flow](https://lucid.app/publicSegments/view/91d4e32f-6dbd-4131-9993-55b6a51896e3/image.png)

## 🛠️ Installation

### 1. Clone the Repository

```
git clone https://github.com/Northeastern-MLOps-Group-12/QueryHub.git
cd QueryHub
```

### 2. Backend Setup (FastAPI)

```
cd backend
python3 -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app-server.api.main:app --reload
```

### 3. Frontend Setup (React + TypeScript)

```
cd frontend
npm install
npm run dev # Starts frontend on localhost:3000
```

### 4. Run with Docker Compose

```
docker-compose up --build
```

---

## ⚙️ Usage

QueryHub is available as a web application with a simple chat-style interface. Users can interact with their connected databases through natural language queries.

The flow of the application works as follows:

#### 1. User Access

- Existing users can log in, while new users can sign up and connect their database.

#### 2. Chat Interface

- Users enter natural language queries into the chat interface.
- The system attempts to generate a valid SQL query.

#### 3. Query Handling

- If the query can be generated successfully, results are returned as tables, CSV downloads, and interactive Plotly charts.
- If the query cannot be resolved, the system provides error messages or suggestions for refinement.

#### 4. Continuous Access

- Users can return later and still see their previous queries and results within the chat interface.

---

## 📊 Monitoring & Metrics

- **Model Performance:** LLM error rates, RAG Retriever Score, SQL semantic accuracy
- **Query Efficiency:** End-to-end latency, bottlenecks, error rates
- **System Health:** Uptime, resource utilization, pipeline failures
- **Visualization Success:** Chart generation success rate, CSV export effectiveness
- **Tools Used:** GCP Cloud Monitoring, Grafana, MLflow, Prometheus, Sentry

---

## ✅ Success Criteria

- 85%+ query-to-SQL accuracy
- 80%+ visualization coverage for chart-suitable queries
- <15s average response time per query
- GDPR/CCPA compliant data storage and user control
- High user satisfaction scores from feedback surveys

---

📅 Timeline

- **Week 1–2:** Repo setup, frontend + backend scaffolding
- **Week 3–5:** Data pipeline development and testing
- **Week 6–7:** NLP engine + RAG-based SQL mapping
- **Week 8:** Visualization + frontend enhancements
- **Week 9:** Security, monitoring, alpha testing
- **Week 10:** Deployment on GCP & final documentation
