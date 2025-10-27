from typing import Dict

def _get_social_media_domain() -> Dict:
    return {
        'description': 'Social media platform data including users, posts, engagement metrics, and content analytics.',
        'tables': {
            'users': {
                'columns': {
                    'user_id': ('INT', list(range(1, 16))),
                    'username': ('VARCHAR(100)', ['techguru', 'travelbug', 'foodie_life', 'fitness_pro', 'art_lover', 'music_fan', 'book_worm', 'gamer_101', 'photo_wizard', 'cook_master', 'sports_enthusiast', 'fashion_icon', 'pet_parent', 'nature_explorer', 'film_critic']),
                    'followers': ('INT', [15000, 8500, 22000, 12000, 6500, 18000, 4200, 25000, 11000, 7800, 9500, 16000, 5500, 13000, 10500]),
                    'following': ('INT', [450, 320, 580, 410, 290, 520, 230, 650, 380, 310, 420, 490, 270, 460, 390]),
                    'account_type': ('VARCHAR(50)', ['Creator', 'Business', 'Creator', 'Business', 'Personal', 'Creator', 'Personal', 'Creator', 'Business', 'Creator', 'Personal', 'Business', 'Personal', 'Creator', 'Personal']),
                    'join_date': ('DATE', ['2019-03-15', '2020-06-20', '2018-09-10', '2020-01-25', '2021-11-30', '2019-05-12', '2022-02-18', '2018-07-22', '2020-10-05', '2021-08-14', '2022-04-18', '2019-11-22', '2023-01-15', '2020-09-08', '2021-03-30'])
                },
                'primary_key': 'user_id',
                'grouping_cols': ['account_type'],
                'metric_cols': ['followers', 'following'],
                'date_cols': ['join_date']
            },
            'posts': {
                'columns': {
                    'post_id': ('INT', list(range(1, 16))),
                    'user_id': ('INT', list(range(1, 16))),
                    'likes': ('INT', [1500, 850, 2200, 1200, 650, 1800, 420, 2500, 1100, 780, 950, 1600, 550, 1300, 1050]),
                    'comments': ('INT', [45, 28, 68, 42, 21, 55, 15, 72, 38, 26, 32, 51, 19, 44, 35]),
                    'shares': ('INT', [120, 75, 180, 95, 40, 140, 30, 200, 85, 60, 70, 130, 45, 110, 90]),
                    'content_type': ('VARCHAR(50)', ['Image', 'Video', 'Image', 'Video', 'Image', 'Video', 'Image', 'Video', 'Image', 'Video', 'Image', 'Video', 'Image', 'Video', 'Image']),
                    'post_date': ('DATE', ['2023-06-01', '2023-06-05', '2023-06-10', '2023-06-15', '2023-06-20', '2023-06-25', '2023-07-01', '2023-07-05', '2023-07-10', '2023-07-15', '2023-07-20', '2023-07-25', '2023-08-01', '2023-08-05', '2023-08-10'])
                },
                'primary_key': 'post_id',
                'grouping_cols': ['content_type'],
                'metric_cols': ['likes', 'comments', 'shares'],
                'date_cols': ['post_date'],
                'foreign_keys': {'user_id': 'users'}
            },
            'campaigns': {
                'columns': {
                    'campaign_id': ('INT', list(range(1, 16))),
                    'user_id': ('INT', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
                    'campaign_name': ('VARCHAR(255)', ['Summer Sale', 'New Product Launch', 'Holiday Special', 'Flash Deal', 'Brand Awareness', 'Influencer Collab', 'Seasonal Promo', 'Product Review', 'Contest Giveaway', 'Tutorial Series', 'Customer Stories', 'Behind Scenes', 'FAQ Session', 'Live Event', 'Product Demo']),
                    'budget': ('DECIMAL(10,2)', [5000.00, 8000.00, 3500.00, 2000.00, 10000.00, 6500.00, 4000.00, 1500.00, 7500.00, 3000.00, 5500.00, 4500.00, 2500.00, 9000.00, 6000.00]),
                    'impressions': ('INT', [150000, 220000, 95000, 65000, 280000, 175000, 120000, 55000, 210000, 85000, 145000, 125000, 75000, 250000, 165000]),
                    'start_date': ('DATE', ['2023-06-01', '2023-06-15', '2023-07-01', '2023-07-15', '2023-08-01', '2023-08-15', '2023-09-01', '2023-09-15', '2023-10-01', '2023-10-15', '2023-11-01', '2023-11-15', '2023-12-01', '2023-12-15', '2024-01-01'])
                },
                'primary_key': 'campaign_id',
                'grouping_cols': [],
                'metric_cols': ['budget', 'impressions'],
                'date_cols': ['start_date'],
                'foreign_keys': {'user_id': 'users'}
            }
        }
    }
