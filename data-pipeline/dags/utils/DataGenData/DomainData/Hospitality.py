from typing import Dict

def _get_hospitality_domain() -> Dict:
    return {
        'description': 'Hotel and hospitality industry data including reservations, rooms, guests, and service ratings.',
        'tables': {
            'hotels': {
                'columns': {
                    'hotel_id': ('INT', list(range(1, 16))),
                    'name': ('VARCHAR(255)', ['Grand Plaza', 'Ocean View Resort', 'Mountain Lodge', 'City Center Inn', 'Sunset Hotel', 'Riverside Suite', 'Garden Retreat', 'Downtown Marriott', 'Beach Paradise', 'Hilltop Hotel', 'Lakeside Resort', 'Urban Stay', 'Country Manor', 'Skyline Tower', 'Harbor Hotel']),
                    'city': ('VARCHAR(100)', ['New York', 'Los Angeles', 'Denver', 'Chicago', 'San Diego', 'Austin', 'Portland', 'Boston', 'Miami', 'Seattle', 'Orlando', 'Houston', 'Nashville', 'San Francisco', 'Baltimore']),
                    'star_rating': ('INT', [5, 5, 4, 3, 4, 4, 3, 5, 5, 4, 4, 3, 4, 5, 4]),
                    'room_count': ('INT', [250, 350, 120, 80, 200, 150, 100, 300, 280, 140, 220, 90, 110, 320, 180]),
                    'open_date': ('DATE', ['2010-05-15', '2012-08-20', '2015-03-10', '2018-06-25', '2013-11-12', '2016-09-18', '2019-01-22', '2011-07-14', '2014-04-08', '2017-10-30', '2015-12-15', '2020-02-05', '2018-08-10', '2012-11-20', '2016-05-25'])
                },
                'primary_key': 'hotel_id',
                'grouping_cols': ['city', 'star_rating'],
                'metric_cols': ['room_count'],
                'date_cols': ['open_date']
            },
            'rooms': {
                'columns': {
                    'room_id': ('INT', list(range(1, 16))),
                    'hotel_id': ('INT', [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8]),
                    'room_number': ('VARCHAR(10)', ['101', '102', '201', '202', '301', '302', '101', '102', '201', '202', '301', '302', '101', '102', '201']),
                    'room_type': ('VARCHAR(50)', ['Deluxe', 'Suite', 'Standard', 'Deluxe', 'Standard', 'Suite', 'Standard', 'Deluxe', 'Suite', 'Standard', 'Deluxe', 'Suite', 'Standard', 'Deluxe', 'Suite']),
                    'price_per_night': ('DECIMAL(10,2)', [250.00, 450.00, 150.00, 200.00, 120.00, 300.00, 100.00, 180.00, 380.00, 130.00, 220.00, 400.00, 110.00, 190.00, 420.00]),
                    'last_renovation': ('DATE', ['2020-01-15', '2020-01-15', '2019-06-20', '2019-06-20', '2021-03-10', '2021-03-10', '2022-08-05', '2022-08-05', '2020-11-12', '2020-11-12', '2021-09-18', '2021-09-18', '2023-01-22', '2023-01-22', '2020-07-14'])
                },
                'primary_key': 'room_id',
                'grouping_cols': ['room_type'],
                'metric_cols': ['price_per_night'],
                'date_cols': ['last_renovation'],
                'foreign_keys': {'hotel_id': 'hotels'}
            },
            'reservations': {
                'columns': {
                    'reservation_id': ('INT', list(range(1, 16))),
                    'room_id': ('INT', list(range(1, 16))),
                    'guest_name': ('VARCHAR(255)', ['Alice Johnson', 'Bob Smith', 'Carol White', 'David Brown', 'Emma Davis', 'Frank Wilson', 'Grace Miller', 'Henry Moore', 'Iris Taylor', 'Jack Anderson', 'Kelly Thomas', 'Leo Jackson', 'Maya Harris', 'Noah Martin', 'Olivia Thompson']),
                    'nights': ('INT', [3, 5, 2, 4, 1, 7, 3, 2, 6, 4, 3, 5, 2, 4, 3]),
                    'total_cost': ('DECIMAL(10,2)', [750.00, 2250.00, 300.00, 800.00, 120.00, 2100.00, 300.00, 360.00, 2280.00, 520.00, 660.00, 2000.00, 220.00, 760.00, 1260.00]),
                    'check_in_date': ('DATE', ['2023-06-01', '2023-06-05', '2023-06-10', '2023-06-15', '2023-06-20', '2023-06-25', '2023-07-01', '2023-07-05', '2023-07-10', '2023-07-15', '2023-07-20', '2023-07-25', '2023-08-01', '2023-08-05', '2023-08-10'])
                },
                'primary_key': 'reservation_id',
                'grouping_cols': [],
                'metric_cols': ['nights', 'total_cost'],
                'date_cols': ['check_in_date'],
                'foreign_keys': {'room_id': 'rooms'}
            }
        }
    }
