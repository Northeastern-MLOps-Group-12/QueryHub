from typing import Dict

def _get_logistics_domain() -> Dict:
    return {
        'description': 'Supply chain and logistics data including warehouses, shipments, routes, and delivery tracking.',
        'tables': {
            'warehouses': {
                'columns': {
                    'warehouse_id': ('INT', list(range(1, 16))),
                    'location': ('VARCHAR(100)', ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'Austin', 'Seattle', 'Denver', 'Boston', 'Atlanta', 'Miami']),
                    'capacity': ('INT', [10000, 15000, 12000, 8000, 9000, 11000, 7000, 13000, 14000, 10500, 16000, 9500, 11500, 8500, 12500]),
                    'manager': ('VARCHAR(255)', ['John Doe', 'Jane Smith', 'Bob Wilson', 'Alice Brown', 'Charlie Davis', 'Diana Miller', 'Eve Taylor', 'Frank Moore', 'Grace Lee', 'Henry Clark', 'Iris Robinson', 'Jack Lewis', 'Kate Walker', 'Leo Hall', 'Maya Allen']),
                    'warehouse_type': ('VARCHAR(50)', ['Distribution', 'Storage', 'Distribution', 'Storage', 'Distribution', 'Storage', 'Distribution', 'Fulfillment', 'Distribution', 'Storage', 'Fulfillment', 'Distribution', 'Storage', 'Fulfillment', 'Distribution']),
                    'open_date': ('DATE', ['2018-01-15', '2018-05-20', '2019-03-10', '2019-07-22', '2020-02-14', '2020-06-18', '2021-01-25', '2021-09-12', '2022-03-30', '2022-11-08', '2019-08-15', '2020-10-20', '2021-05-12', '2022-07-18', '2023-02-25'])
                },
                'primary_key': 'warehouse_id',
                'grouping_cols': ['location', 'warehouse_type'],
                'metric_cols': ['capacity'],
                'date_cols': ['open_date']
            },
            'shipments': {
                'columns': {
                    'shipment_id': ('INT', list(range(1, 16))),
                    'warehouse_id': ('INT', [1, 2, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
                    'weight': ('DECIMAL(10,2)', [150.50, 200.75, 125.00, 180.25, 220.50, 95.75, 300.00, 175.50, 250.25, 140.00, 190.75, 210.50, 160.25, 230.00, 145.50]),
                    'status': ('VARCHAR(50)', ['Delivered', 'In Transit', 'Processing', 'Delivered', 'In Transit', 'Delivered', 'Processing', 'Delivered', 'In Transit', 'Delivered', 'Processing', 'In Transit', 'Delivered', 'Processing', 'Delivered']),
                    'priority': ('VARCHAR(50)', ['Standard', 'Express', 'Standard', 'Priority', 'Express', 'Standard', 'Priority', 'Standard', 'Express', 'Standard', 'Priority', 'Express', 'Standard', 'Priority', 'Express']),
                    'ship_date': ('DATE', ['2023-01-10', '2023-02-15', '2023-03-20', '2023-04-05', '2023-05-12', '2023-06-18', '2023-07-22', '2023-08-14', '2023-09-09', '2023-10-30', '2023-11-15', '2023-12-05', '2024-01-10', '2024-02-08', '2024-03-12'])
                },
                'primary_key': 'shipment_id',
                'grouping_cols': ['status', 'priority'],
                'metric_cols': ['weight'],
                'date_cols': ['ship_date'],
                'foreign_keys': {'warehouse_id': 'warehouses'}
            },
            'routes': {
                'columns': {
                    'route_id': ('INT', list(range(1, 16))),
                    'origin': ('VARCHAR(100)', ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'Austin', 'Seattle', 'Denver', 'Boston', 'Atlanta', 'Miami']),
                    'destination': ('VARCHAR(100)', ['Boston', 'San Francisco', 'Detroit', 'Miami', 'Seattle', 'Atlanta', 'Denver', 'Portland', 'Nashville', 'Las Vegas', 'Los Angeles', 'Chicago', 'New York', 'Houston', 'Dallas']),
                    'distance': ('INT', [215, 380, 280, 1200, 1420, 400, 920, 1100, 240, 1100, 1100, 920, 215, 800, 1050]),
                    'route_type': ('VARCHAR(50)', ['Ground', 'Air', 'Ground', 'Air', 'Air', 'Ground', 'Ground', 'Air', 'Ground', 'Air', 'Air', 'Ground', 'Ground', 'Air', 'Ground']),
                    'active_date': ('DATE', ['2022-01-01', '2022-02-01', '2022-03-01', '2022-04-01', '2022-05-01', '2022-06-01', '2022-07-01', '2022-08-01', '2022-09-01', '2022-10-01', '2022-11-01', '2022-12-01', '2023-01-01', '2023-02-01', '2023-03-01'])
                },
                'primary_key': 'route_id',
                'grouping_cols': ['origin', 'destination', 'route_type'],
                'metric_cols': ['distance'],
                'date_cols': ['active_date']
            }
        }
    }
