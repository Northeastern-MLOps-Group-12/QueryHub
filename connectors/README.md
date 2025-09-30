connectors/
│
├── __init__.py
│
├── base_connector.py      # The single abstract blueprint for ALL connectors
│
├── factory.py             # The factory to get any specific connector
│
└── engines/
    │
    ├── postgres/
    │   ├── __init__.py
    │   ├── base.py          # The generic BasePostgresConnector
    │   └── providers/
    │       ├── __init__.py
    │       ├── aws_rds.py   # AwsRdsPostgresConnector
    │       └── gcp_cloudsql.py # GcpCloudSqlPostgresConnector
    │
    └── mysql/
        ├── __init__.py
        ├── base.py          # The generic BaseMySQLConnector
        └── providers/
            ├── __init__.py
            ├── aws_rds.py   # AwsRdsMySQLConnector
            └── gcp_cloudsql.py # GcpCloudSqlMySQLConnector