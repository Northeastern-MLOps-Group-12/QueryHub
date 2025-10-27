from typing import Dict

def _get_gaming_domain() -> Dict:
    return {
        'description': 'Gaming platform data including players, matches, achievements, and performance statistics.',
        'tables': {
            'players': {
                'columns': {
                    'player_id': ('INT', list(range(1, 16))),
                    'username': ('VARCHAR(100)', ['DragonSlayer', 'ShadowNinja', 'IceQueen', 'FireStorm', 'NightHawk', 'StormBreaker', 'MysticMage', 'IronWarrior', 'SwiftArcher', 'DarkKnight', 'LightBringer', 'ThunderBolt', 'CrimsonBlade', 'FrostWolf', 'SilverFox']),
                    'level': ('INT', [45, 62, 38, 71, 52, 48, 67, 55, 43, 69, 51, 58, 40, 65, 47]),
                    'experience_points': ('INT', [45000, 78000, 32000, 95000, 58000, 51000, 82000, 61000, 40000, 88000, 56000, 70000, 38000, 80000, 49000]),
                    'rank': ('VARCHAR(50)', ['Gold', 'Platinum', 'Silver', 'Diamond', 'Gold', 'Gold', 'Platinum', 'Gold', 'Silver', 'Diamond', 'Gold', 'Platinum', 'Silver', 'Platinum', 'Gold']),
                    'region': ('VARCHAR(50)', ['NA', 'EU', 'NA', 'ASIA', 'EU', 'NA', 'EU', 'ASIA', 'NA', 'EU', 'ASIA', 'NA', 'EU', 'ASIA', 'NA']),
                    'join_date': ('DATE', ['2021-03-15', '2020-06-20', '2022-09-10', '2020-01-25', '2021-11-30', '2022-05-12', '2020-08-18', '2021-07-22', '2022-10-05', '2020-03-14', '2021-04-18', '2020-11-22', '2022-06-15', '2020-09-08', '2021-12-30'])
                },
                'primary_key': 'player_id',
                'grouping_cols': ['rank', 'region'],
                'metric_cols': ['level', 'experience_points'],
                'date_cols': ['join_date']
            },
            'matches': {
                'columns': {
                    'match_id': ('INT', list(range(1, 16))),
                    'player_id': ('INT', list(range(1, 16))),
                    'game_mode': ('VARCHAR(50)', ['Ranked', 'Casual', 'Ranked', 'Tournament', 'Casual', 'Ranked', 'Tournament', 'Casual', 'Ranked', 'Tournament', 'Casual', 'Ranked', 'Tournament', 'Casual', 'Ranked']),
                    'score': ('INT', [2500, 1800, 3200, 4100, 1500, 2800, 3900, 1700, 2900, 4300, 1600, 3100, 4500, 1900, 2700]),
                    'duration_minutes': ('INT', [35, 28, 42, 55, 25, 38, 52, 30, 40, 58, 27, 44, 60, 32, 37]),
                    'result': ('VARCHAR(20)', ['Win', 'Loss', 'Win', 'Win', 'Loss', 'Win', 'Loss', 'Loss', 'Win', 'Win', 'Loss', 'Win', 'Win', 'Loss', 'Win']),
                    'match_date': ('DATE', ['2023-08-01', '2023-08-05', '2023-08-10', '2023-08-15', '2023-08-20', '2023-08-25', '2023-09-01', '2023-09-05', '2023-09-10', '2023-09-15', '2023-09-20', '2023-09-25', '2023-10-01', '2023-10-05', '2023-10-10'])
                },
                'primary_key': 'match_id',
                'grouping_cols': ['game_mode', 'result'],
                'metric_cols': ['score', 'duration_minutes'],
                'date_cols': ['match_date'],
                'foreign_keys': {'player_id': 'players'}
            },
            'achievements': {
                'columns': {
                    'achievement_id': ('INT', list(range(1, 16))),
                    'player_id': ('INT', list(range(1, 16))),
                    'achievement_name': ('VARCHAR(255)', ['First Blood', 'Triple Kill', 'Legendary', 'Untouchable', 'Dominating', 'Rampage', 'Godlike', 'Master Strategist', 'Perfect Game', 'Comeback King', 'Speed Demon', 'Sharpshooter', 'Tank Buster', 'Support Hero', 'Solo Carry']),
                    'rarity': ('VARCHAR(50)', ['Common', 'Rare', 'Epic', 'Legendary', 'Rare', 'Epic', 'Legendary', 'Epic', 'Legendary', 'Rare', 'Common', 'Rare', 'Epic', 'Common', 'Legendary']),
                    'points': ('INT', [100, 250, 500, 1000, 250, 500, 1000, 500, 1000, 250, 100, 250, 500, 100, 1000]),
                    'unlock_date': ('DATE', ['2023-07-15', '2023-08-05', '2023-08-20', '2023-09-10', '2023-09-15', '2023-09-25', '2023-10-05', '2023-10-15', '2023-10-25', '2023-11-01', '2023-11-10', '2023-11-20', '2023-12-01', '2023-12-10', '2023-12-20'])
                },
                'primary_key': 'achievement_id',
                'grouping_cols': ['rarity'],
                'metric_cols': ['points'],
                'date_cols': ['unlock_date'],
                'foreign_keys': {'player_id': 'players'}
            }
        }
    }
