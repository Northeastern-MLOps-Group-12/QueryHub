# QueryHub Text-to-SQL Data Pipeline

> **MLOps Data Pipeline Project - Comprehensive Production-Grade Implementation**

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Pipeline Stages](#pipeline-stages)
- [Bias Detection & Mitigation](#bias-detection--mitigation)
- [Data Versioning with DVC](#data-versioning-with-dvc)
- [Schema Validation & Statistics](#schema-validation--statistics)
- [Testing](#testing)
- [Monitoring & Alerts](#monitoring--alerts)
- [Pipeline Optimization](#pipeline-optimization)
- [Reproducibility](#reproducibility)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This project implements a **production-grade MLOps data pipeline** for the QueryHub Text-to-SQL dataset. The pipeline is orchestrated using **Apache Airflow** and includes comprehensive data processing, quality validation, bias detection and mitigation, synthetic data generation, and automated monitoring.

### What This Pipeline Does

1. **Downloads and validates** synthetic text-to-SQL training data from Hugging Face
2. **Validates SQL queries** in parallel using sqlglot parser (filters out 100+ invalid queries)
3. **Detects class imbalance bias** in SQL complexity distribution
4. **Generates synthetic data** using template-based approach across 11 domains
5. **Removes data leakage** through comprehensive duplicate detection (intra-split + cross-split)
6. **Validates data schemas** at both raw and engineered feature stages
7. **Creates train/val/test splits** with stratified sampling
8. **Sends email alerts** for pipeline failures and success with detailed statistics
9. **Tracks data versions** using DVC for reproducibility
---

## 🚀 Key Features

### ✨ Core Pipeline Features

| Feature | Description | Implementation |
|---------|-------------|----------------|
| **Data Acquisition** | Downloads text-to-SQL dataset from Hugging Face | `load_data()` |
| **SQL Validation** | Parallel validation of 100K+ SQL queries | `validate_sql()` with ProcessPoolExecutor |
| **Preprocessing** | Data cleaning and transformation | `preprocess()` |
| **Schema Validation** | Dual-stage validation (raw + engineered) | `validate_raw_schema()`, `validate_engineered_schema()` |
| **Bias Detection** | Class imbalance detection with severity levels | `detect_bias()` |
| **Synthetic Generation** | Template-based data augmentation | `analyze_and_generate_synthetic()` |
| **Data Leakage Prevention** | 2-step duplicate removal | `remove_data_leakage()` |
| **Feature Engineering** | Creates input_text from context + prompt | `merge_and_split()` |

### 🎨 Advanced Features

- **Email Notifications**: HTML emails with comprehensive statistics
- **Anomaly Detection**: Identifies invalid SQL, missing values, empty strings
- **Statistical Profiling**: Min/max/median/percentiles for all text columns
- **Diversity Analysis**: Calculates expected duplicate rates using birthday paradox
- **Multi-Domain Synthesis**: 11 domains (retail, healthcare, finance, etc.)
- **Template Variety**: 75+ SQL templates across 5 complexity types

---

## 🏗️ Architecture

<img width="330" height="781" alt="image" src="https://github.com/user-attachments/assets/a96fbda5-5d7c-4588-990e-a9ba1683d5da" />

### Pipeline Flow

1. **Testing Phase**: Runs pytest suite before pipeline execution
2. **Acquisition Phase**: Downloads dataset from Hugging Face
3. **Validation Phase**: Validates SQL syntax using parallel processing
4. **Quality Phase**: Schema validation + statistical profiling
5. **Bias Phase**: Detects class imbalance and calculates severity
6. **Augmentation Phase**: Generates synthetic data for minority classes
7. **Integration Phase**: Merges data, creates train/val/test splits
8. **Deduplication Phase**: Removes intra-split and cross-split duplicates
9. **Final Validation**: Validates engineered features and checks for leakage
10. **Notification Phase**: Sends comprehensive success email

---

## 📦 Prerequisites

### System Requirements

- **Operating System**: Linux, macOS, or Windows with WSL2
- **Python**: 3.10
- **RAM**: Minimum 8GB (16GB recommended for parallel processing)
- **CPU**: Multi-core processor (pipeline uses 75% of cores)
- **Disk Space**: ~5GB for dataset and generated files

### Required Software

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

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/queryhub-pipeline.git
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

**Google Cloud Storage (GCS)**:
```bash
dvc remote add -d myremote gs://my-bucket/data-pipeline
dvc remote modify myremote credentialpath ~/.config/gcloud/credentials.json
```

**AWS S3**:
```bash
dvc remote add -d myremote s3://my-bucket/data-pipeline
dvc remote modify myremote access_key_id YOUR_ACCESS_KEY
dvc remote modify myremote secret_access_key YOUR_SECRET_KEY
```

#### 3.4. Track Data Directory
```bash
dvc add data/
git add data.dvc .gitignore
git commit -m "Track data with DVC"
```

#### 3.5. Push Data to Remote
```bash
dvc push
```
---

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

### 6. Start Airflow

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

#### 9.1. After Pipeline Execution
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
git clone https://github.com/yourusername/queryhub-pipeline.git
cd queryhub-pipeline/data-pipeline

# Pull data from DVC remote
dvc pull

# Data is now available in data/
ls data/
```
---

## 📁 Project Structure

```
data-pipeline/
│
├── dags/
│   ├── data_pipeline_dag.py              # Main Airflow DAG
│   │
│   └── utils/
│       ├── EmailContentGenerator.py      # Email notification utilities
│       ├── SQLValidator.py               # SQL validation with sqlglot
│       ├── DataGenerator.py              # Synthetic data generation engine
│       │
│       ├── DataGenData/
│       │   ├── Templates/
│       │   │   ├── CTETemplates.py       # 15 CTE query templates
│       │   │   ├── SETTemplates.py       # 15 set operation templates
│       │   │   ├── MultipleJoinsTemplates.py  # 15 join templates
│       │   │   ├── SubQueryTemplates.py  # 15 subquery templates
│       │   │   └── WindowFunctionTemplates.py # 15 window function templates
│       │   │
│       │   └── DomainData/
│       │       ├── Ecommerce.py          # E-commerce domain data
│       │       ├── Healthcare.py         # Healthcare domain data
│       │       ├── Finance.py            # Finance domain data
│       │       ├── Education.py          # Education domain data
│       │       ├── Logistics.py          # Logistics domain data
│       │       ├── Manufacturing.py      # Manufacturing domain data
│       │       ├── RealEstate.py         # Real estate domain data
│       │       ├── Hospitality.py        # Hospitality domain data
│       │       ├── SocialMedia.py        # Social media domain data
│       │       ├── Gaming.py             # Gaming domain data
│       │       └── Retail.py             # Retail domain data
│       │
│       └── __init__.py
│
├── data/
│   ├── train.csv                         # Final training data
│   ├── val.csv                           # Final validation data
│   ├── test.csv                          # Final test data
│   ├── synthetic_data.csv                # Generated synthetic samples
│   ├── sql_validation_anomalies.csv      # Invalid SQL queries report
│   ├── raw_schema_and_stats.json         # Raw data statistics
│   └── engineered_schema_and_stats.json  # Final data statistics
│
├── tests/
│   └── test.py                           # Comprehensive pytest suite (45 tests)
│
├── logs/
│   └── (Airflow logs - auto-generated)
│
├── plugins/
│   └── (Airflow plugins - if any)
│
├── .dvc/
│   └── config                            # DVC configuration
│
├── .dvcignore                            # DVC ignore patterns
├── data.dvc                              # DVC tracking file for data/
├── .gitignore                            # Git ignore patterns
├── .env                                  # Environment variables (not tracked)
├── requirements.txt                      # Python dependencies
├── docker-compose.yaml                   # Docker Compose configuration
├── Dockerfile                            # Docker image definition
└── README.md                             # This file
```

---

## 🔄 Pipeline Stages

### Stage 1: Pre-Pipeline Testing (`run_pytest_tests`)

**Purpose**: Validate code quality before pipeline execution

**Implementation**:
```python
def run_pytest_tests(**context):
    """Run pytest tests before starting the pipeline"""
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', test_file, '-v'],
        capture_output=True, text=True, timeout=60
    )
```

**Output**: Test results logged to Airflow

---

### Stage 2: Data Acquisition (`load_data`)

**Purpose**: Download text-to-SQL dataset from Hugging Face

**Dataset**: `gretelai/synthetic_text_to_sql`

**Output**:
- `/tmp/train_raw.pkl`: Raw training data (~100,000 samples)
- `/tmp/test_raw.pkl`: Raw test data (~10,000 samples)

**DVC Tracked**: Yes

---

### Stage 3: SQL Validation (`validate_sql`)

**Purpose**: Validate SQL syntax and identify anomalies

**Key Features**:
- **Parallel Processing**: Uses `ProcessPoolExecutor` with 75% of CPU cores
- **SQL Parser**: Uses `sqlglot` for syntax validation
- **Anomaly Tracking**: Logs all invalid queries with error types

**Performance**: ~1000 queries/second on 8-core CPU

**Output**:
- `/tmp/train_valid.pkl`: Validated training data
- `/tmp/test_valid.pkl`: Validated test data
- `/opt/airflow/data/sql_validation_anomalies.csv`: Anomaly report

**DVC Tracked**: Yes

---

### Stage 4: Preprocessing (`preprocess`)

**Purpose**: Prepare data for schema validation

**Output**:
- `/tmp/train_preprocessed.pkl`
- `/tmp/test_preprocessed.pkl`

---

### Stage 5: Raw Schema Validation (`validate_raw_schema`)

**Purpose**: Validate raw data schema and generate baseline statistics

**Schema Validation**:
```python
expected_schema = {
    'sql_prompt': 'object',      # string
    'sql_context': 'object',     # string
    'sql': 'object',             # string
    'sql_complexity': 'object'   # string
}
```

**Checks Performed**:
1. ✅ All required columns exist
2. ✅ Data types match expectations
3. ✅ No null values in critical columns
4. ✅ No empty strings

**Statistical Profiling**: For each text column, calculates:
- Count, null count, null rate
- Unique count, duplicate count
- Length statistics: min, max, mean, median, std
- Percentiles: P25, P50, P75, P90, P95, P99

**Output**: `/opt/airflow/data/raw_schema_and_stats.json`

**DVC Tracked**: Yes

---

### Stage 6: Bias Detection (`detect_bias`)

**Purpose**: Detect class imbalance in SQL complexity distribution

**Bias Severity Levels**:

| Level | Imbalance Ratio | Action Required |
|-------|----------------|-----------------|
| **SEVERE** | ≥ 10x | 🚨 Urgent - Generate synthetic data |
| **MODERATE** | 5-10x | ⚠️ Recommended - Synthetic data generation |
| **MILD** | 2-5x | ℹ️ Optional - Monitor and balance |
| **None** | < 2x | ✅ No action needed |

**Minority Class Detection**: Classes with count < 50% of majority class are flagged

**Email Alert**: Sends HTML email with detailed bias report

---

### Stage 7: Synthetic Data Generation (`analyze_and_generate_synthetic`)

**Purpose**: Generate synthetic data to balance minority classes

**Approach**: Template-based generation using domain-specific data

**Key Components**:

#### A. Template Library (75 total templates)
- **CTEs**: 15 templates (Common Table Expressions)
- **Set Operations**: 15 templates (UNION, INTERSECT, EXCEPT)
- **Multiple Joins**: 15 templates (2-3 table joins)
- **Subqueries**: 15 templates (nested SELECT)
- **Window Functions**: 15 templates (ROW_NUMBER, RANK, LAG, LEAD)

#### B. Domain Data (11 domains)
- Retail, Healthcare, Finance, Education
- Logistics, E-commerce, Manufacturing
- Real Estate, Hospitality, Social Media, Gaming

#### C. Diversity Maximization

**Variation Sources**:
```
Total Combinations ≈ 10^15+
- 75 templates
- 11 domains
- 5 prompt variations
- ~12 average verbs
- 3,360 date combinations (10 years × 12 months × 28 days)
- 100,000 threshold range
- 50 top_n range
- 125 SQL style variations (5 agg × 5 operators × 5 WHERE styles)
```

**Expected Duplicate Rate**: < 0.0001% for 40,000 samples

**Output**: `/opt/airflow/data/synthetic_data.csv` (40,000-80,000 samples)

**Performance**: ~6000 samples/second, ~40 seconds for 240K samples

**DVC Tracked**: Yes

---

### Stage 8: Data Merging & Splitting (`merge_and_split`)

**Purpose**: Combine original + synthetic data, create train/val/test splits

**Operations**:
1. Merge original and synthetic data
2. Create `input_text` feature from context + prompt
3. Stratified 90/10 train/val split (random_state=42)

**Output Files**:
- `/opt/airflow/data/train.csv`: Training set (~285K samples)
- `/opt/airflow/data/val.csv`: Validation set (~32K samples)
- `/opt/airflow/data/test.csv`: Test set (~6K samples, unchanged)

**DVC Tracked**: Yes

---

### Stage 9: Data Leakage Removal (`remove_data_leakage`)

**Purpose**: Eliminate duplicates within and across train/val/test splits

**2-Step Process**:

#### Step 1: Intra-Split Deduplication
Removes duplicates within each split

#### Step 2: Cross-Split Leakage Detection
Identifies and removes samples that appear in multiple splits
- Removes from train if overlap with val
- Removes from train if overlap with test
- Raises error if val-test overlap (critical error)

**Example Output**:
```
DUPLICATE REMOVAL SUMMARY
========================================
Intra-split duplicates removed: 237
Cross-split duplicates removed: 42
Total samples removed: 279
========================================
```

**DVC Tracked**: Yes

---

### Stage 10: Engineered Schema Validation (`validate_engineered_schema`)

**Purpose**: Validate final dataset schema and quality

**Schema Validation**:
```python
expected_schema = {
    'input_text': 'object',
    'sql': 'object',
    'sql_complexity': 'object'
}
```

**Validation Checks**:
1. ✅ Column validation
2. ✅ Data type validation
3. ✅ Format validation (input_text contains proper format)
4. ✅ Quality checks (no nulls, no empty strings, no leakage)

**Output**: `/opt/airflow/data/engineered_schema_and_stats.json`

**DVC Tracked**: Yes

---

### Stage 11: Success Notification (`send_pipeline_success_notification`)

**Purpose**: Send comprehensive success email with pipeline statistics

**Email Content Includes**:
1. Pipeline summary
2. Final dataset sizes
3. SQL complexity distribution
4. Before/after comparison
5. Data quality metrics
6. Generated files list

---

## 🎯 Bias Detection & Mitigation

### Problem Statement

SQL complexity classes in the original dataset are severely imbalanced:

| Complexity Class | Original Count | Percentage | Status |
|-----------------|---------------|------------|--------|
| **basic** | 55,000 | 55% | Majority |
| **aggregation** | 30,000 | 30% | Adequate |
| **multiple_joins** | 12,000 | 12% | Adequate |
| **CTEs** | 2,500 | 2.5% | **Minority** |
| **window functions** | 500 | 0.5% | **Minority** |

**Imbalance Ratio**: 55,000 / 500 = **110x** (SEVERE)

### Mitigation Strategy

**Template-Based Synthetic Data Generation**

**Advantages**:
1. ✅ Semantic Correctness: Templates ensure valid SQL syntax
2. ✅ Domain Diversity: 11 domains prevent overfitting
3. ✅ High Variance: 10^15+ unique combinations minimize duplicates
4. ✅ Scalability: Can generate millions of samples

### Post-Mitigation Results

| Complexity Class | After Balancing | Percentage | Status |
|-----------------|----------------|------------|--------|
| **basic** | 55,000 | 25.8% | Balanced |
| **aggregation** | 52,000 | 24.4% | Balanced |
| **multiple_joins** | 51,000 | 23.9% | Balanced |
| **CTEs** | 54,500 | 25.5% | **Balanced** ✅ |
| **window functions** | 54,500 | 25.5% | **Balanced** ✅ |

**New Imbalance Ratio**: 1.08x (NONE) ✅

---

## 📊 Schema Validation & Statistics

### Dual-Stage Validation

#### Stage 1: Raw Data Schema Validation
- Validates schema before transformations
- Checks columns, data types, null values, empty strings
- Output: `raw_schema_and_stats.json`

#### Stage 2: Engineered Features Schema Validation
- Validates final dataset ready for ML training
- Checks format, data leakage, duplicates
- Output: `engineered_schema_and_stats.json`

### Statistical Profiling

#### Text Column Statistics
- Character length: min, max, mean, median, std
- Percentiles: P25, P50, P75, P90, P95, P99
- Unique count, duplicate count, null count

#### Categorical Column Statistics
- Distribution (counts per class)
- Unique values, null count

---

## 🧪 Testing

### Test Coverage

**Total Tests**: 45 comprehensive unit tests

**Categories**:
1. Data Loading: 5 tests
2. SQL Validation: 7 tests
3. Preprocessing: 3 tests
4. Schema Validation: 8 tests
5. Bias Detection: 6 tests
6. Synthetic Generation: 5 tests
7. Data Splitting: 6 tests
8. Leakage Removal: 7 tests
9. Success Notification: 3 tests

### Running Tests (Runs as part of DAG)

**Important**: Local test execution will show 4 failures due to hardcoded Airflow paths in the pipeline code. The complete test suite (45/45 tests) passes successfully when executed within the Airflow container/DAG runtime environment.
```

```bash
# Run all tests (Will Fail 4 test cases if run locally due to airflow paths set in code but passes all test cases when run inside airflow DAG)
pytest tests/test.py -v 

# With coverage
pytest tests/test.py --cov=dags --cov-report=html

# Run specific test
pytest tests/test.py::test_load_data_success -v

# Run test category
pytest tests/test.py -k "bias" -v
```

---

## 📡 Monitoring & Alerts

### Email Notification System

#### Configuration

**SMTP Settings** (in `docker-compose.yaml`):
```yaml
AIRFLOW__SMTP__SMTP_HOST: smtp.gmail.com
AIRFLOW__SMTP__SMTP_PORT: 587
AIRFLOW__SMTP__SMTP_USER: your-email@gmail.com
AIRFLOW__SMTP__SMTP_PASSWORD: your-app-password
```

### Alert Types

1. **Task Failure Alert**: Triggered by `on_failure_callback`
2. **Bias Detection Alert**: Sent when imbalance detected
3. **SQL Validation Anomalies Alert**: When invalid queries found
4. **Pipeline Success Alert**: Comprehensive statistics after completion

### Logging

**Location**: `/opt/airflow/logs/` or 'logs' (in local machine)

**Custom Logging**:
```python
import logging
logging.info("📊 Analyzing SQL complexity distribution")
logging.warning("⚠️ Found 237 duplicate samples")
logging.error("❌ Schema validation failed")
```

---

## ⚡ Pipeline Optimization

### Current Performance

**Total Pipeline Duration**: 3 minutes

**Breakdown**:
1. Pre-pipeline tests: 11s
2. Data loading: 12s
3. SQL validation: 40s (runs parallely)
4. Preprocessing: 1.5s
5. Raw schema validation: 1.1s
6. Bias detection: 4s
7. Synthetic generation: 40s
8. Merge and split: 25s
9. Leakage removal: 29s
10. Engineered schema validation: 10s
11. Success notification: 2s



### Optimization Techniques

1. **Parallel SQL Validation**: 70% faster using ProcessPoolExecutor
2. **Efficient Statistical Calculations**: Memory-optimized with garbage collection
3. **Batch CSV Writes**: Only necessary columns

---

## 🔄 Reproducibility

### Reproducibility Checklist

- ✅ **Environment**: `requirements.txt` with pinned versions
- ✅ **Data**: DVC tracking for all datasets
- ✅ **Code**: Git version control
- ✅ **Configuration**: `.env` file for environment variables
- ✅ **Random Seeds**: `random_state=42` in all splits
- ✅ **Docker**: `Dockerfile` and `docker-compose.yaml`

### Step-by-Step Reproduction

1. Clone repository
2. Set up Python environment
3. Configure environment variables
4. Start Airflow
5. Trigger DAG
6. Verify results
7. Push Generated Data and Logs to DVC

---

## 🛠️ Troubleshooting

### Common Issues

#### Issue 1: SMTP Connection Error
**Solution**: Check SMTP settings, use Gmail App Password

#### Issue 2: DVC Pull Fails
**Solution**: Verify remote configuration, check credentials

#### Issue 3: Out of Memory
**Solution**: Reduce parallel workers, process in chunks, increase Docker memory

#### Issue 4: Task Stuck in "Running"
**Solution**: Check logs, restart Airflow, clear task state

#### Issue 5: Synthetic Generation Fails
**Solution**: Check domain compatibility, increase max_attempts

#### Issue 6: Test Failures
**Solution**: Check mock configuration, run with verbose output

---
