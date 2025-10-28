from typing import Dict

def _get_manufacturing_domain() -> Dict:
    return {
        'description': 'Manufacturing operations data including production lines, quality control, inventory, and maintenance.',
        'tables': {
            'production_lines': {
                'columns': {
                    'line_id': ('INT', list(range(1, 16))),
                    'line_name': ('VARCHAR(100)', ['Assembly A', 'Assembly B', 'Packaging 1', 'Packaging 2', 'Quality Check 1', 'Quality Check 2', 'Assembly C', 'Packaging 3', 'Testing 1', 'Assembly D', 'Packaging 4', 'Quality Check 3', 'Testing 2', 'Assembly E', 'Packaging 5']),
                    'capacity_per_hour': ('INT', [100, 120, 200, 180, 150, 160, 110, 190, 80, 105, 195, 155, 85, 115, 185]),
                    'status': ('VARCHAR(50)', ['Operating', 'Operating', 'Maintenance', 'Operating', 'Operating', 'Maintenance', 'Operating', 'Operating', 'Operating', 'Maintenance', 'Operating', 'Operating', 'Operating', 'Operating', 'Maintenance']),
                    'department': ('VARCHAR(100)', ['Assembly', 'Assembly', 'Packaging', 'Packaging', 'Quality', 'Quality', 'Assembly', 'Packaging', 'Testing', 'Assembly', 'Packaging', 'Quality', 'Testing', 'Assembly', 'Packaging']),
                    'install_date': ('DATE', ['2020-01-15', '2020-03-20', '2020-06-10', '2020-09-15', '2021-01-20', '2021-04-12', '2021-07-18', '2021-10-25', '2022-01-30', '2022-05-15', '2022-08-20', '2022-11-10', '2023-02-18', '2023-06-22', '2023-10-15'])
                },
                'primary_key': 'line_id',
                'grouping_cols': ['department', 'status'],
                'metric_cols': ['capacity_per_hour'],
                'date_cols': ['install_date']
            },
            'products': {
                'columns': {
                    'product_id': ('INT', list(range(1, 16))),
                    'product_name': ('VARCHAR(255)', ['Widget A', 'Widget B', 'Component X', 'Component Y', 'Assembly Unit 1', 'Assembly Unit 2', 'Module P', 'Module Q', 'Part Z', 'Subassembly 1', 'Subassembly 2', 'Final Product A', 'Final Product B', 'Component K', 'Module R']),
                    'production_cost': ('DECIMAL(10,2)', [15.50, 22.75, 8.25, 12.00, 45.50, 52.25, 18.75, 25.00, 6.50, 35.00, 42.50, 125.00, 150.00, 10.75, 28.50]),
                    'defect_rate': ('DECIMAL(3,2)', [0.02, 0.01, 0.03, 0.02, 0.01, 0.02, 0.01, 0.02, 0.04, 0.01, 0.02, 0.01, 0.02, 0.03, 0.01]),
                    'category': ('VARCHAR(100)', ['Electronics', 'Electronics', 'Mechanical', 'Mechanical', 'Electronics', 'Electronics', 'Electrical', 'Electrical', 'Mechanical', 'Electronics', 'Electronics', 'Final', 'Final', 'Mechanical', 'Electrical']),
                    'launch_date': ('DATE', ['2021-01-10', '2021-03-15', '2021-06-20', '2021-09-05', '2022-01-12', '2022-03-18', '2022-06-22', '2022-09-14', '2022-12-08', '2023-03-15', '2023-06-20', '2023-09-10', '2023-12-05', '2022-04-15', '2023-01-22'])
                },
                'primary_key': 'product_id',
                'grouping_cols': ['category'],
                'metric_cols': ['production_cost', 'defect_rate'],
                'date_cols': ['launch_date']
            },
            'production_runs': {
                'columns': {
                    'run_id': ('INT', list(range(1, 16))),
                    'line_id': ('INT', [1, 2, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
                    'product_id': ('INT', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
                    'units_produced': ('INT', [950, 1150, 1950, 980, 1480, 1560, 1080, 1870, 780, 1030, 1920, 1520, 830, 1130, 1810]),
                    'run_date': ('DATE', ['2023-01-15', '2023-02-20', '2023-03-25', '2023-04-10', '2023-05-18', '2023-06-22', '2023-07-15', '2023-08-20', '2023-09-25', '2023-10-30', '2023-11-15', '2023-12-05', '2024-01-12', '2024-02-08', '2024-03-14'])
                },
                'primary_key': 'run_id',
                'grouping_cols': [],
                'metric_cols': ['units_produced'],
                'date_cols': ['run_date'],
                'foreign_keys': {'line_id': 'production_lines', 'product_id': 'products'}
            }
        }
    }
