# Database Connector Service

A **flexible and extensible database connector service** that provides a unified interface for connecting to multiple database engines.

---

## Project Structure

```
connectors/
├── engines/
│   ├── mysql/
│   │   ├── __init__.py
│   │   └── mysql_connector.py
│   └── postgres/
│       ├── __init__.py
│       └── postgres_connector.py
├── README.md
├── __init__.py
├── base_connector.py
└── connector.py
```

---

## Files Overview

- **`base_connector.py`**: Defines the abstract base class that serves as a blueprint for all database connectors (PostgreSQL, MySQL, etc.)  
- **`connectors.py`**: Contains the connector factory class that identifies and instantiates the appropriate connector based on user requirements  
- **`engines/postgres/`**: Houses the PostgreSQL-specific connector implementation  
- **`engines/mysql/`**: Houses the MySQL-specific connector implementation  

### Implementation Files in Agents

- **`agents/load_data_to_vector/graph.py`**: Contains the graph construction logic and node definitions  
- **`agents/load_data_to_vector/state.py`**: Defines the agent state schema and state management  

---

## Key Features

- Unified interface for multiple database engines  
- Extensible architecture for adding new connectors easily  
- Secure credential storage  
- Automatic schema indexing and embedding into ChromaDB  
- Ready integration with LLM-based natural language queries
