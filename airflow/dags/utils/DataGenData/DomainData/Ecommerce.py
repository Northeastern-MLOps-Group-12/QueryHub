from typing import Dict

def _get_ecommerce_domain() -> Dict:
    return {
        'description': 'Online marketplace data including products, orders, reviews, and customer interactions.',
        'tables': {
            'products': {
                'columns': {
                    'product_id': ('INT', list(range(1, 16))),
                    'name': ('VARCHAR(255)', ['Wireless Earbuds', 'Smart Watch', 'Fitness Tracker', 'Bluetooth Speaker', 'Phone Case', 'Charger', 'Screen Protector', 'Power Bank', 'USB Cable', 'Car Mount', 'Tablet Stand', 'Laptop Bag', 'Wireless Mouse', 'Keyboard Cover', 'Stylus Pen']),
                    'price': ('DECIMAL(10,2)', [79.99, 299.99, 149.99, 59.99, 19.99, 29.99, 9.99, 49.99, 14.99, 24.99, 34.99, 59.99, 39.99, 29.99, 44.99]),
                    'category': ('VARCHAR(100)', ['Audio', 'Electronics', 'Electronics', 'Audio', 'Accessories', 'Accessories', 'Accessories', 'Electronics', 'Accessories', 'Accessories', 'Accessories', 'Accessories', 'Electronics', 'Accessories', 'Electronics']),
                    'rating': ('DECIMAL(2,1)', [4.5, 4.7, 4.3, 4.6, 4.2, 4.4, 4.1, 4.8, 4.0, 4.3, 4.5, 4.6, 4.4, 4.2, 4.7]),
                    'seller': ('VARCHAR(100)', ['TechStore', 'GadgetHub', 'TechStore', 'AudioPro', 'CaseMaster', 'TechStore', 'ScreenGuard', 'PowerPlus', 'CableCo', 'AutoTech', 'TechStore', 'BagWorld', 'GadgetHub', 'CaseMaster', 'PenTech']),
                    'list_date': ('DATE', ['2022-06-15', '2022-07-20', '2022-08-10', '2022-09-05', '2022-10-12', '2023-01-18', '2023-02-22', '2023-03-14', '2023-04-08', '2023-05-20', '2022-11-15', '2023-06-10', '2023-07-25', '2023-08-18', '2023-09-30'])
                },
                'primary_key': 'product_id',
                'grouping_cols': ['category', 'seller'],
                'metric_cols': ['price', 'rating'],
                'date_cols': ['list_date']
            },
            'orders': {
                'columns': {
                    'order_id': ('INT', list(range(1, 16))),
                    'product_id': ('INT', [1, 2, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
                    'customer_id': ('INT', list(range(1, 16))),
                    'quantity': ('INT', [2, 1, 1, 3, 2, 1, 4, 1, 2, 1, 2, 1, 3, 2, 1]),
                    'total_amount': ('DECIMAL(10,2)', [159.98, 299.99, 149.99, 239.97, 39.98, 29.99, 39.96, 49.99, 29.98, 24.99, 69.98, 59.99, 119.97, 59.98, 44.99]),
                    'order_date': ('DATE', ['2023-06-01', '2023-06-05', '2023-06-10', '2023-06-15', '2023-06-20', '2023-06-25', '2023-07-01', '2023-07-05', '2023-07-10', '2023-07-15', '2023-07-20', '2023-07-25', '2023-08-01', '2023-08-05', '2023-08-10'])
                },
                'primary_key': 'order_id',
                'grouping_cols': [],
                'metric_cols': ['quantity', 'total_amount'],
                'date_cols': ['order_date'],
                'foreign_keys': {'product_id': 'products'}
            },
            'reviews': {
                'columns': {
                    'review_id': ('INT', list(range(1, 16))),
                    'product_id': ('INT', list(range(1, 16))),
                    'rating': ('INT', [5, 4, 5, 4, 3, 5, 4, 5, 4, 4, 5, 4, 5, 3, 4]),
                    'verified_purchase': ('VARCHAR(10)', ['Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'No']),
                    'review_date': ('DATE', ['2023-07-01', '2023-07-05', '2023-07-10', '2023-07-15', '2023-07-20', '2023-07-25', '2023-08-01', '2023-08-05', '2023-08-10', '2023-08-15', '2023-08-20', '2023-08-25', '2023-09-01', '2023-09-05', '2023-09-10'])
                },
                'primary_key': 'review_id',
                'grouping_cols': ['verified_purchase'],
                'metric_cols': ['rating'],
                'date_cols': ['review_date'],
                'foreign_keys': {'product_id': 'products'}
            }
        }
    }
