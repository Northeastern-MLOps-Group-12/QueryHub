from typing import Dict

def _get_finance_domain() -> Dict:
    return {
        'description': 'Banking and financial services data including accounts, transactions, loans, and investments.',
        'tables': {
            'accounts': {
                'columns': {
                    'account_id': ('INT', list(range(1, 16))),
                    'customer_id': ('INT', list(range(1, 16))),
                    'account_type': ('VARCHAR(50)', ['Checking', 'Savings', 'Credit', 'Checking', 'Savings', 'Credit', 'Checking', 'Savings', 'Investment', 'Credit', 'Checking', 'Money Market', 'Savings', 'Investment', 'Credit']),
                    'balance': ('DECIMAL(10,2)', [5000.00, 12000.00, 1500.00, 8000.00, 25000.00, 3000.00, 6500.00, 18000.00, 50000.00, 2000.00, 7500.00, 30000.00, 15000.00, 45000.00, 2500.00]),
                    'status': ('VARCHAR(50)', ['Active', 'Active', 'Active', 'Active', 'Active', 'Frozen', 'Active', 'Active', 'Active', 'Closed', 'Active', 'Active', 'Active', 'Active', 'Frozen']),
                    'open_date': ('DATE', ['2020-01-15', '2020-05-20', '2021-03-10', '2021-07-22', '2021-11-05', '2022-02-14', '2022-06-18', '2022-09-25', '2023-01-30', '2023-05-12', '2020-08-15', '2021-04-20', '2022-10-10', '2023-07-05', '2024-01-15'])
                },
                'primary_key': 'account_id',
                'grouping_cols': ['account_type', 'status'],
                'metric_cols': ['balance'],
                'date_cols': ['open_date']
            },
            'transactions': {
                'columns': {
                    'transaction_id': ('INT', list(range(1, 16))),
                    'account_id': ('INT', [1, 2, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
                    'amount': ('DECIMAL(10,2)', [250.00, 1500.00, 75.50, 500.00, 2000.00, 125.75, 800.00, 350.25, 1200.00, 450.50, 600.00, 2500.00, 175.00, 1800.00, 325.50]),
                    'transaction_type': ('VARCHAR(50)', ['Deposit', 'Withdrawal', 'Transfer', 'Deposit', 'Withdrawal', 'Transfer', 'Deposit', 'Withdrawal', 'Transfer', 'Deposit', 'Withdrawal', 'Transfer', 'Deposit', 'Withdrawal', 'Transfer']),
                    'transaction_date': ('DATE', ['2023-01-10', '2023-02-15', '2023-03-20', '2023-04-05', '2023-05-12', '2023-06-18', '2023-07-22', '2023-08-14', '2023-09-09', '2023-10-30', '2023-11-15', '2023-12-05', '2024-01-12', '2024-02-08', '2024-03-14'])
                },
                'primary_key': 'transaction_id',
                'grouping_cols': ['transaction_type'],
                'metric_cols': ['amount'],
                'date_cols': ['transaction_date'],
                'foreign_keys': {'account_id': 'accounts'}
            },
            'loans': {
                'columns': {
                    'loan_id': ('INT', list(range(1, 16))),
                    'account_id': ('INT', list(range(1, 16))),
                    'loan_amount': ('DECIMAL(10,2)', [25000.00, 50000.00, 15000.00, 100000.00, 30000.00, 45000.00, 20000.00, 75000.00, 35000.00, 60000.00, 40000.00, 85000.00, 28000.00, 55000.00, 32000.00]),
                    'interest_rate': ('DECIMAL(5,2)', [3.5, 4.2, 5.0, 3.8, 4.5, 3.9, 5.2, 4.0, 4.8, 3.6, 4.3, 3.7, 4.9, 4.1, 4.6]),
                    'loan_type': ('VARCHAR(50)', ['Personal', 'Mortgage', 'Auto', 'Mortgage', 'Personal', 'Auto', 'Personal', 'Mortgage', 'Auto', 'Personal', 'Business', 'Mortgage', 'Auto', 'Personal', 'Business']),
                    'issue_date': ('DATE', ['2022-01-15', '2022-03-20', '2022-06-10', '2022-09-05', '2023-01-12', '2023-03-18', '2023-05-22', '2023-07-14', '2023-09-08', '2023-11-20', '2022-04-15', '2022-10-12', '2023-02-25', '2023-08-18', '2024-01-10'])
                },
                'primary_key': 'loan_id',
                'grouping_cols': ['loan_type'],
                'metric_cols': ['loan_amount', 'interest_rate'],
                'date_cols': ['issue_date'],
                'foreign_keys': {'account_id': 'accounts'}
            }
        }
    }
