from typing import Dict

def _get_retail_domain() -> Dict:
    return {
        'description': 'Comprehensive retail data covering sales, inventory, customers, and product analytics.',
        'tables': {
            'products': {
                'columns': {
                    'product_id': ('INT', list(range(1, 16))),
                    'name': ('VARCHAR(255)', ['Laptop', 'Phone', 'Tablet', 'Monitor', 'Keyboard', 'Mouse', 'Headphones', 'Webcam', 'Speaker', 'Router', 'Hard Drive', 'Memory Card', 'USB Hub', 'Printer', 'Scanner']),
                    'price': ('DECIMAL(10,2)', [999.99, 699.50, 299.00, 450.25, 89.99, 45.50, 129.99, 79.99, 199.00, 150.75, 89.50, 39.99, 55.00, 299.99, 349.50]),
                    'category': ('VARCHAR(100)', ['Electronics', 'Electronics', 'Electronics', 'Electronics', 'Accessories', 'Accessories', 'Audio', 'Video', 'Audio', 'Network', 'Storage', 'Storage', 'Accessories', 'Office', 'Office']),
                    'stock': ('INT', [50, 120, 80, 35, 200, 150, 90, 65, 110, 45, 75, 300, 125, 40, 55]),
                    'supplier': ('VARCHAR(100)', ['SupplierA', 'SupplierB', 'SupplierA', 'SupplierC', 'SupplierB', 'SupplierA', 'SupplierC', 'SupplierB', 'SupplierA', 'SupplierC', 'SupplierB', 'SupplierA', 'SupplierC', 'SupplierB', 'SupplierA']),
                    'created_date': ('DATE', ['2022-01-10', '2022-03-15', '2022-06-20', '2022-09-05', '2023-01-12', '2023-02-18', '2023-04-22', '2023-07-30', '2023-09-15', '2023-11-08', '2022-05-14', '2022-08-22', '2023-03-10', '2023-06-25', '2023-10-15'])
                },
                'primary_key': 'product_id',
                'grouping_cols': ['category', 'supplier'],
                'metric_cols': ['price', 'stock'],
                'date_cols': ['created_date']
            },
            'customers': {
                'columns': {
                    'customer_id': ('INT', list(range(1, 16))),
                    'name': ('VARCHAR(255)', ['John Smith', 'Jane Doe', 'Bob Johnson', 'Alice Williams', 'Charlie Brown', 'Diana Prince', 'Eve Davis', 'Frank Miller', 'Grace Lee', 'Henry Wilson', 'Iris Chen', 'Jack Taylor', 'Kate Anderson', 'Leo Martinez', 'Mia Robinson']),
                    'city': ('VARCHAR(100)', ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'Austin', 'Seattle', 'Denver', 'Boston', 'Portland', 'Miami']),
                    'total_purchases': ('INT', [5, 12, 3, 8, 15, 7, 20, 4, 11, 6, 9, 14, 2, 18, 10]),
                    'member_status': ('VARCHAR(50)', ['Gold', 'Silver', 'Bronze', 'Gold', 'Platinum', 'Silver', 'Platinum', 'Bronze', 'Gold', 'Silver', 'Gold', 'Platinum', 'Bronze', 'Platinum', 'Silver']),
                    'signup_date': ('DATE', ['2020-01-15', '2020-03-22', '2020-06-10', '2021-02-18', '2021-08-05', '2021-11-12', '2022-01-30', '2022-05-15', '2022-09-20', '2023-03-10', '2020-07-25', '2021-04-18', '2022-06-22', '2023-01-14', '2023-08-30'])
                },
                'primary_key': 'customer_id',
                'grouping_cols': ['city', 'member_status'],
                'metric_cols': ['total_purchases'],
                'date_cols': ['signup_date']
            },
            'orders': {
                'columns': {
                    'order_id': ('INT', list(range(1, 16))),
                    'customer_id': ('INT', [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
                    'product_id': ('INT', [1, 2, 1, 3, 2, 4, 5, 3, 1, 2, 6, 7, 8, 9, 10]),
                    'quantity': ('INT', [1, 2, 1, 3, 1, 2, 4, 1, 2, 3, 1, 2, 1, 3, 2]),
                    'status': ('VARCHAR(50)', ['Delivered', 'Delivered', 'Processing', 'Shipped', 'Delivered', 'Processing', 'Shipped', 'Delivered', 'Cancelled', 'Processing', 'Delivered', 'Shipped', 'Processing', 'Delivered', 'Shipped']),
                    'order_date': ('DATE', ['2023-01-15', '2023-02-20', '2023-03-10', '2023-03-25', '2023-04-12', '2023-05-18', '2023-06-22', '2023-07-14', '2023-08-09', '2023-09-30', '2023-10-15', '2023-11-05', '2023-12-01', '2024-01-10', '2024-02-14'])
                },
                'primary_key': 'order_id',
                'grouping_cols': ['status'],
                'metric_cols': ['quantity'],
                'date_cols': ['order_date'],
                'foreign_keys': {'customer_id': 'customers', 'product_id': 'products'}
            },
            'sales': {
                'columns': {
                    'sale_id': ('INT', list(range(1, 16))),
                    'product_id': ('INT', [1, 2, 3, 1, 2, 4, 5, 3, 1, 6, 7, 8, 9, 10, 11]),
                    'amount': ('DECIMAL(10,2)', [999.99, 1399.00, 299.00, 999.99, 699.50, 900.50, 179.96, 897.00, 1999.98, 89.99, 259.97, 159.96, 55.00, 299.99, 349.50]),
                    'region': ('VARCHAR(100)', ['North', 'South', 'East', 'West', 'North', 'South', 'East', 'West', 'North', 'South', 'East', 'West', 'North', 'South', 'East']),
                    'sale_date': ('DATE', ['2023-01-10', '2023-01-15', '2023-02-20', '2023-03-05', '2023-03-15', '2023-04-10', '2023-05-22', '2023-06-18', '2023-07-25', '2023-08-30', '2023-09-12', '2023-10-20', '2023-11-08', '2023-12-15', '2024-01-22'])
                },
                'primary_key': 'sale_id',
                'grouping_cols': ['region'],
                'metric_cols': ['amount'],
                'date_cols': ['sale_date'],
                'foreign_keys': {'product_id': 'products'}
            }
        }
    }
