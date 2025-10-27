from typing import Dict

def _get_real_estate_domain() -> Dict:
    return {
        'description': 'Real estate market data including properties, transactions, agents, and market analytics.',
        'tables': {
            'properties': {
                'columns': {
                    'property_id': ('INT', list(range(1, 16))),
                    'address': ('VARCHAR(255)', ['123 Main St', '456 Oak Ave', '789 Pine Rd', '321 Elm St', '654 Maple Dr', '987 Cedar Ln', '147 Birch Way', '258 Spruce Ct', '369 Willow Pl', '741 Ash Blvd', '852 Cherry St', '963 Poplar Ave', '159 Hickory Rd', '357 Walnut Dr', '486 Beech Ln']),
                    'city': ('VARCHAR(100)', ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'Austin', 'Seattle', 'Denver', 'Boston', 'Portland', 'Miami']),
                    'price': ('DECIMAL(12,2)', [450000.00, 850000.00, 325000.00, 275000.00, 395000.00, 525000.00, 285000.00, 725000.00, 350000.00, 425000.00, 675000.00, 485000.00, 625000.00, 575000.00, 495000.00]),
                    'bedrooms': ('INT', [3, 4, 2, 2, 3, 4, 2, 4, 3, 3, 4, 3, 4, 3, 3]),
                    'property_type': ('VARCHAR(50)', ['House', 'House', 'Condo', 'Apartment', 'House', 'House', 'Condo', 'House', 'Townhouse', 'House', 'House', 'Townhouse', 'House', 'Condo', 'House']),
                    'list_date': ('DATE', ['2023-01-15', '2023-02-20', '2023-03-10', '2023-04-05', '2023-05-12', '2023-06-18', '2023-07-22', '2023-08-14', '2023-09-09', '2023-10-30', '2023-11-15', '2023-12-05', '2024-01-10', '2024-02-08', '2024-03-12'])
                },
                'primary_key': 'property_id',
                'grouping_cols': ['city', 'property_type'],
                'metric_cols': ['price', 'bedrooms'],
                'date_cols': ['list_date']
            },
            'agents': {
                'columns': {
                    'agent_id': ('INT', list(range(1, 16))),
                    'name': ('VARCHAR(255)', ['Sarah Mitchell', 'John Peterson', 'Emily Roberts', 'Michael Chang', 'Lisa Thompson', 'David Kim', 'Jennifer Lopez', 'Robert Wilson', 'Maria Garcia', 'James Anderson', 'Patricia Taylor', 'Thomas Brown', 'Linda Davis', 'Charles Martinez', 'Barbara White']),
                    'agency': ('VARCHAR(100)', ['Prime Realty', 'Urban Homes', 'Prime Realty', 'Skyline Properties', 'Urban Homes', 'Prime Realty', 'Coastal Real Estate', 'Skyline Properties', 'Urban Homes', 'Prime Realty', 'Coastal Real Estate', 'Urban Homes', 'Skyline Properties', 'Prime Realty', 'Urban Homes']),
                    'commission_rate': ('DECIMAL(3,2)', [0.03, 0.025, 0.03, 0.028, 0.025, 0.03, 0.027, 0.028, 0.025, 0.03, 0.027, 0.025, 0.028, 0.03, 0.025]),
                    'years_experience': ('INT', [12, 8, 15, 6, 10, 18, 7, 14, 9, 20, 11, 13, 5, 16, 8]),
                    'hire_date': ('DATE', ['2015-03-15', '2018-06-20', '2012-09-10', '2020-01-25', '2017-11-30', '2010-05-12', '2019-02-18', '2014-07-22', '2018-10-05', '2008-08-14', '2016-04-18', '2015-11-22', '2021-01-15', '2013-09-08', '2018-03-30'])
                },
                'primary_key': 'agent_id',
                'grouping_cols': ['agency'],
                'metric_cols': ['commission_rate', 'years_experience'],
                'date_cols': ['hire_date']
            },
            'transactions': {
                'columns': {
                    'transaction_id': ('INT', list(range(1, 16))),
                    'property_id': ('INT', list(range(1, 16))),
                    'agent_id': ('INT', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
                    'sale_price': ('DECIMAL(12,2)', [445000.00, 840000.00, 320000.00, 270000.00, 390000.00, 520000.00, 280000.00, 720000.00, 345000.00, 420000.00, 670000.00, 480000.00, 620000.00, 570000.00, 490000.00]),
                    'sale_date': ('DATE', ['2023-03-15', '2023-04-20', '2023-05-10', '2023-06-05', '2023-07-12', '2023-08-18', '2023-09-22', '2023-10-14', '2023-11-09', '2024-01-05', '2024-02-15', '2024-03-05', '2024-04-10', '2024-05-08', '2024-06-12'])
                },
                'primary_key': 'transaction_id',
                'grouping_cols': [],
                'metric_cols': ['sale_price'],
                'date_cols': ['sale_date'],
                'foreign_keys': {'property_id': 'properties', 'agent_id': 'agents'}
            }
        }
    }
