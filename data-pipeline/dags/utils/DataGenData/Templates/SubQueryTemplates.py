from typing import List,Dict

def _get_subquery_templates() -> List[Dict]:
    """15 diverse subquery templates"""
    return [
        # ============ TEMPLATE 1 ============
        {
            'sql': """SELECT {group_col}, {metric_col}
FROM {table1}
WHERE {metric_col} > (SELECT AVG({metric_col}) FROM {table1})
AND {date_col} >= '{year}-01-01'
ORDER BY {metric_col} DESC;""",
            'prompt_templates': [
                "{verb} {group_col} where {metric_col} exceeds the overall average since {year}",
                "{verb} above-average {group_col} by {metric_col} for {year}",
                "{verb} {group_col} with {metric_col} higher than mean value",
                "{verb} {group_col} surpassing average {metric_col} in {year}",
                "{verb} {group_col} outperforming mean {metric_col} since {year}"
            ],
            'explanation': "Uses subquery to calculate overall average {metric_col}, filters main query to show only {group_col} with above-average values from {year} onwards",
            'tables_needed': ['{table1}'],
            'requires': ['grouping_col', 'metric_col', 'date_col'],
            'verb_style': 'find'
        },
        
        # ============ TEMPLATE 2 ============
        {
            'sql': """SELECT t1.{group_col}, t1.{metric_col},
    (SELECT COUNT(*) FROM {table2} t2 WHERE t2.{fk1} = t1.{pk1}) as related_count,
    (SELECT AVG({metric_col2}) FROM {table2} t2 WHERE t2.{fk1} = t1.{pk1}) as avg_related_{metric_col2}
FROM {table1} t1
WHERE t1.{date_col} >= '{year}-01-01'
AND (SELECT COUNT(*) FROM {table2} t2 WHERE t2.{fk1} = t1.{pk1}) > 0;""",
            'prompt_templates': [
                "{verb} {group_col} with counts and averages from related {table2} records since {year}",
                "{verb} {group_col} including relationship statistics from {table2}",
                "{verb} {table1} {group_col} with aggregated metrics from associated {table2} entries",
                "{verb} {group_col} showing {table2} relationships and average values",
                "{verb} {group_col} with correlated {table2} counts and means"
            ],
            'explanation': "Uses correlated subqueries to count related records in {table2} and calculate their average {metric_col2} for each record in {table1}, filters for {year} and non-zero relationships",
            'tables_needed': ['{table1}', '{table2}'],
            'requires': ['foreign_key_relationship', 'grouping_col', 'metric_col', 'date_col'],
            'verb_style': 'show'
        },
        
        # ============ TEMPLATE 3 ============
        {
            'sql': """SELECT {group_col}, {metric_col}, {date_col}
FROM {table1} t1
WHERE {metric_col} >= (
    SELECT {metric_col}
    FROM {table1} t2
    WHERE t2.{group_col} = t1.{group_col}
    ORDER BY {date_col} DESC
    LIMIT 1
)
AND {date_col} >= '{year}-01-01'
ORDER BY {metric_col} DESC;""",
            'prompt_templates': [
                "{verb} {group_col} where current {metric_col} matches or exceeds most recent value",
                "{verb} {group_col} with {metric_col} at or above their latest recorded level",
                "{verb} {group_col} maintaining or improving {metric_col} versus most recent",
                "{verb} {group_col} with {metric_col} meeting or surpassing latest observation",
                "{verb} {group_col} where {metric_col} equals or tops most recent entry"
            ],
            'explanation': "Correlated subquery finds most recent {metric_col} value for each {group_col}, main query filters to show only records where {metric_col} meets or exceeds this recent value, for records since {year}",
            'tables_needed': ['{table1}'],
            'requires': ['grouping_col', 'metric_col', 'date_col'],
            'verb_style': 'identify'
        },
        
        # ============ TEMPLATE 4 ============
        {
            'sql': """SELECT {group_col}, COUNT(*) as count, AVG({metric_col}) as avg_{metric_col}
FROM {table1}
WHERE {group_col} IN (
    SELECT {group_col}
    FROM {table1}
    WHERE {date_col} >= '{year}-01-01'
    GROUP BY {group_col}
    HAVING SUM({metric_col}) > {threshold}
)
AND {date_col} >= '{year}-01-01'
GROUP BY {group_col}
ORDER BY avg_{metric_col} DESC;""",
            'prompt_templates': [
                "{verb} {group_col} statistics where total {metric_col} exceeds {threshold} for {year}",
                "{verb} counts and averages for {group_col} with cumulative {metric_col} above {threshold}",
                "{verb} aggregated {group_col} data filtered by total {metric_col} threshold",
                "{verb} {group_col} metrics for groups surpassing {threshold} in {metric_col}",
                "{verb} statistical breakdown of high-performing {group_col} by {metric_col}"
            ],
            'explanation': "Subquery identifies {group_col} values where sum of {metric_col} exceeds {threshold} in {year}, main query calculates count and average for those groups",
            'tables_needed': ['{table1}'],
            'requires': ['grouping_col', 'metric_col', 'date_col'],
            'verb_style': 'analyze'
        },
        
        # ============ TEMPLATE 5 ============
        {
            'sql': """SELECT {group_col}, {metric_col}, {date_col}
FROM {table1} t1
WHERE {metric_col} = (
    SELECT MAX({metric_col})
    FROM {table1} t2
    WHERE t2.{group_col} = t1.{group_col}
    AND t2.{date_col} >= '{year}-01-01'
)
AND {date_col} >= '{year}-01-01'
ORDER BY {metric_col} DESC;""",
            'prompt_templates': [
                "{verb} records with maximum {metric_col} for each {group_col} in {year}",
                "{verb} highest {metric_col} entry per {group_col} during {year}",
                "{verb} peak {metric_col} values across all {group_col} for {year}",
                "{verb} {group_col} showing their maximum {metric_col} achievement in {year}",
                "{verb} top {metric_col} records by {group_col} for {year}"
            ],
            'explanation': "Correlated subquery finds maximum {metric_col} for each {group_col} in {year}, main query returns only records matching these maximum values",
            'tables_needed': ['{table1}'],
            'requires': ['grouping_col', 'metric_col', 'date_col'],
            'verb_style': 'find'
        },
        
        # ============ TEMPLATE 6 ============
        {
            'sql': """SELECT {group_col}, {metric_col}
FROM {table1} t1
WHERE {metric_col} > ALL (
    SELECT {metric_col}
    FROM {table1} t2
    WHERE t2.{group_col} = t1.{group_col}
    AND t2.{date_col} < t1.{date_col}
    AND t2.{date_col} >= DATE_SUB(t1.{date_col}, INTERVAL 90 DAY)
)
AND {date_col} >= '{year}-01-01'
ORDER BY {date_col}, {metric_col} DESC;""",
            'prompt_templates': [
                "{verb} {group_col} where {metric_col} exceeds all values in previous 90 days",
                "{verb} {group_col} reaching new 90-day highs in {metric_col}",
                "{verb} {group_col} with {metric_col} surpassing recent 3-month peak",
                "{verb} {group_col} breaking 90-day {metric_col} records",
                "{verb} {group_col} achieving quarterly {metric_col} peaks"
            ],
            'explanation': "Uses ALL comparison in subquery to find records where {metric_col} exceeds all values from the previous 90 days for that {group_col}, identifying new quarterly peaks, filtered for {year}",
            'tables_needed': ['{table1}'],
            'requires': ['grouping_col', 'metric_col', 'date_col'],
            'verb_style': 'detect'
        },
        
        # ============ TEMPLATE 7 ============
        {
            'sql': """SELECT {group_col}, {metric_col},
    (SELECT COUNT(*) FROM {table1} t2 WHERE t2.{metric_col} > t1.{metric_col}) + 1 as overall_rank
FROM {table1} t1
WHERE {date_col} >= '{year}-01-01'
ORDER BY overall_rank
LIMIT {top_n};""",
            'prompt_templates': [
                "{verb} top {top_n} {group_col} by {metric_col} with calculated ranks for {year}",
                "{verb} highest-ranked {group_col} based on {metric_col} in {year}",
                "{verb} leading {top_n} {group_col} ordered by {metric_col} rank",
                "{verb} best {top_n} {group_col} with position rankings by {metric_col}",
                "{verb} {top_n} highest {group_col} showing their {metric_col} ranks"
            ],
            'explanation': "Subquery counts how many records have higher {metric_col} to calculate rank for each record, main query returns top {top_n} by this calculated rank for records since {year}",
            'tables_needed': ['{table1}'],
            'requires': ['grouping_col', 'metric_col', 'date_col'],
            'verb_style': 'rank'
        },
        
        # ============ TEMPLATE 8 ============
        {
            'sql': """SELECT {group_col}, {metric_col}, {date_col}
FROM {table1} t1
WHERE {metric_col} > (SELECT PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {metric_col})
    FROM {table1}
    WHERE {date_col} >= '{year}-01-01')
AND {date_col} >= '{year}-01-01'
ORDER BY {metric_col} DESC;""",
            'prompt_templates': [
                "{verb} {group_col} in top quartile of {metric_col} for {year}",
                "{verb} {group_col} above 75th percentile in {metric_col}",
                "{verb} top 25% of {group_col} by {metric_col} since {year}",
                "{verb} {group_col} exceeding third quartile threshold",
                "{verb} {group_col} in highest quartile for {metric_col}"
            ],
            'explanation': "Subquery calculates 75th percentile of {metric_col}, main query returns records above this threshold, showing top quartile performers from {year}",
            'tables_needed': ['{table1}'],
            'requires': ['grouping_col', 'metric_col', 'date_col'],
            'verb_style': 'find'
        },
        
        # ============ TEMPLATE 9 ============
        {
            'sql': """SELECT {group_col}, 
    COUNT(*) as total_records,
    AVG({metric_col}) as avg_metric,
    (SELECT AVG({metric_col}) FROM {table1}) as overall_avg,
    AVG({metric_col}) - (SELECT AVG({metric_col}) FROM {table1}) as avg_difference
FROM {table1}
WHERE {date_col} >= '{year}-01-01'
GROUP BY {group_col}
HAVING AVG({metric_col}) > (SELECT AVG({metric_col}) FROM {table1})
ORDER BY avg_difference DESC;""",
            'prompt_templates': [
                "{verb} {group_col} with above-average {metric_col} and deviation",
                "{verb} {group_col} exceeding overall {metric_col} mean",
                "{verb} high-performing {group_col} with average comparisons",
                "{verb} {group_col} surpassing baseline {metric_col} average",
                "{verb} {group_col} showing positive deviation from mean"
            ],
            'explanation': "Calculates group averages and overall average via subquery, shows only groups above overall average with magnitude of difference",
            'tables_needed': ['{table1}'],
            'requires': ['grouping_col', 'metric_col', 'date_col'],
            'verb_style': 'compare'
        },
        
        # ============ TEMPLATE 10 ============
        {
            'sql': """SELECT {group_col}, {metric_col}, {date_col}
FROM {table1} t1
WHERE EXISTS (
    SELECT 1 FROM {table2} t2
    WHERE t2.{fk1} = t1.{pk1}
    AND t2.{metric_col2} > {threshold}
)
AND t1.{date_col} >= '{year}-01-01'
ORDER BY t1.{metric_col} DESC;""",
            'prompt_templates': [
                "{verb} {group_col} from {table1} with high-value {table2} relationships",
                "{verb} {group_col} connected to {table2} records above {threshold}",
                "{verb} {group_col} having related {table2} entries exceeding threshold",
                "{verb} {group_col} linked to significant {table2} records",
                "{verb} {group_col} with qualifying {table2} associations"
            ],
            'explanation': "Uses EXISTS subquery to filter {table1} records that have related {table2} entries with {metric_col2} > {threshold}, efficient for large datasets",
            'tables_needed': ['{table1}', '{table2}'],
            'requires': ['foreign_key_relationship', 'metric_col', 'date_col'],
            'verb_style': 'filter'
        },
        
        # ============ TEMPLATE 11 ============
        {
            'sql': """SELECT {group_col}, {metric_col},
    (SELECT COUNT(DISTINCT {group_col}) FROM {table1} WHERE {metric_col} > t1.{metric_col}) as better_count,
    (SELECT COUNT(DISTINCT {group_col}) FROM {table1}) as total_count,
    ROUND((SELECT COUNT(DISTINCT {group_col}) FROM {table1} WHERE {metric_col} > t1.{metric_col}) * 100.0 / 
            (SELECT COUNT(DISTINCT {group_col}) FROM {table1}), 2) as percentile
FROM {table1} t1
WHERE {date_col} >= '{year}-01-01'
ORDER BY percentile
LIMIT {top_n};""",
            'prompt_templates': [
                "{verb} bottom {top_n} {group_col} with percentile rankings",
                "{verb} lowest {top_n} {group_col} by {metric_col} with context",
                "{verb} {group_col} at lower percentiles of {metric_col}",
                "{verb} under-performing {top_n} {group_col} with rankings",
                "{verb} {top_n} weakest {group_col} showing relative position"
            ],
            'explanation': "Multiple subqueries calculate how many groups perform better and total groups, derives percentile, returns lowest {top_n} performers from {year}",
            'tables_needed': ['{table1}'],
            'requires': ['grouping_col', 'metric_col', 'date_col'],
            'verb_style': 'identify'
        },
        
        # ============ TEMPLATE 12 ============
        {
            'sql': """SELECT {group_col}, 
    SUM({metric_col}) as total_metric,
    (SELECT SUM({metric_col}) FROM {table1} WHERE {date_col} >= '{year}-01-01') as grand_total,
    ROUND(SUM({metric_col}) * 100.0 / 
            (SELECT SUM({metric_col}) FROM {table1} WHERE {date_col} >= '{year}-01-01'), 2) as pct_of_total
FROM {table1}
WHERE {date_col} >= '{year}-01-01'
GROUP BY {group_col}
HAVING SUM({metric_col}) * 100.0 / (SELECT SUM({metric_col}) FROM {table1} WHERE {date_col} >= '{year}-01-01') >= 5
ORDER BY pct_of_total DESC;""",
            'prompt_templates': [
                "{verb} {group_col} contributing 5%+ of total {metric_col}",
                "{verb} significant {group_col} by {metric_col} share",
                "{verb} major {group_col} contributors to {metric_col} total",
                "{verb} {group_col} with meaningful {metric_col} percentage",
                "{verb} substantial {group_col} in {metric_col} distribution"
            ],
            'explanation': "Subquery calculates grand total {metric_col}, main query shows groups contributing >= 5% of total, with percentage breakdown from {year}",
            'tables_needed': ['{table1}'],
            'requires': ['grouping_col', 'metric_col', 'date_col'],
            'verb_style': 'highlight'
        },
        
        # ============ TEMPLATE 13 ============
        {
            'sql': """SELECT {group_col}, {metric_col}, {date_col}
FROM {table1} t1
WHERE {metric_col} IN (
    SELECT {metric_col}
    FROM {table1}
    WHERE {date_col} >= '{year}-01-01'
    GROUP BY {metric_col}
    HAVING COUNT(*) >= 2
)
AND {date_col} >= '{year}-01-01'
ORDER BY {metric_col} DESC;""",
            'prompt_templates': [
                "{verb} {group_col} with duplicate {metric_col} values",
                "{verb} {group_col} sharing common {metric_col} levels",
                "{verb} {group_col} having non-unique {metric_col}",
                "{verb} {group_col} with repeated {metric_col} occurrences",
                "{verb} {group_col} where {metric_col} appears multiple times"
            ],
            'explanation': "Subquery identifies {metric_col} values that appear 2+ times, main query returns all records with these duplicate values from {year}",
            'tables_needed': ['{table1}'],
            'requires': ['grouping_col', 'metric_col', 'date_col'],
            'verb_style': 'find'
        },
        
        # ============ TEMPLATE 14 ============
        {
            'sql': """SELECT t1.{group_col}, t1.{metric_col},
    (SELECT COUNT(*) FROM {table2} t2 WHERE t2.{fk1} = t1.{pk1} AND t2.{date_col} >= '{year}-01-01') as recent_count,
    (SELECT SUM({metric_col2}) FROM {table2} t2 WHERE t2.{fk1} = t1.{pk1}) as lifetime_total
FROM {table1} t1
WHERE {date_col} >= '{year}-01-01'
AND (SELECT COUNT(*) FROM {table2} t2 WHERE t2.{fk1} = t1.{pk1} AND t2.{date_col} >= '{year}-01-01') >= 3
ORDER BY recent_count DESC;""",
            'prompt_templates': [
                "{verb} {group_col} with 3+ recent {table2} relationships",
                "{verb} active {group_col} by {table2} engagement since {year}",
                "{verb} {group_col} showing frequent {table2} connections",
                "{verb} highly connected {group_col} in {table2}",
                "{verb} {group_col} with substantial recent {table2} activity"
            ],
            'explanation': "Multiple correlated subqueries count recent and lifetime {table2} relationships, filters for groups with 3+ recent connections from {year}",
            'tables_needed': ['{table1}', '{table2}'],
            'requires': ['foreign_key_relationship', 'grouping_col', 'metric_col', 'date_col'],
            'verb_style': 'show'
        },
        
        # ============ TEMPLATE 15 ============
        {
            'sql': """SELECT {group_col}, 
    AVG({metric_col}) as avg_metric,
    (SELECT STDDEV({metric_col}) FROM {table1} WHERE {date_col} >= '{year}-01-01') as overall_stddev
FROM {table1}
WHERE {date_col} >= '{year}-01-01'
GROUP BY {group_col}
HAVING ABS(AVG({metric_col}) - (SELECT AVG({metric_col}) FROM {table1} WHERE {date_col} >= '{year}-01-01')) > 
    (SELECT STDDEV({metric_col}) FROM {table1} WHERE {date_col} >= '{year}-01-01')
ORDER BY ABS(AVG({metric_col}) - (SELECT AVG({metric_col}) FROM {table1} WHERE {date_col} >= '{year}-01-01')) DESC;""",
            'prompt_templates': [
                "{verb} statistically significant {group_col} deviations in {metric_col}",
                "{verb} {group_col} with extreme average {metric_col} values",
                "{verb} {group_col} beyond one standard deviation from mean",
                "{verb} outlier {group_col} in {metric_col} distribution",
                "{verb} {group_col} showing notable {metric_col} variance"
            ],
            'explanation': "Multiple subqueries calculate overall mean and standard deviation, identifies groups where average differs by more than 1 standard deviation from {year}",
            'tables_needed': ['{table1}'],
            'requires': ['grouping_col', 'metric_col', 'date_col'],
            'verb_style': 'detect'
        }
    ]




#     def _get_subquery_templates(self) -> List[Dict]:
#         """7 diverse subquery templates"""
#         return [
#             {
#                 'sql': """SELECT {group_col}, {metric_col}
# FROM {table1}
# WHERE {metric_col} > (SELECT AVG({metric_col}) FROM {table1})
# AND {date_col} >= '{year}-01-01'
# ORDER BY {metric_col} DESC;""",
#                 'prompt_templates': [
#                     "{verb} {group_col} where {metric_col} exceeds the overall average since {year}",
#                     "{verb} above-average {group_col} by {metric_col} for {year}",
#                     "{verb} {group_col} with {metric_col} higher than mean value",
#                     "{verb} {group_col} surpassing average {metric_col} in {year}",
#                     "{verb} {group_col} outperforming mean {metric_col} since {year}"
#                 ],
#                 'explanation': "Uses subquery to calculate overall average {metric_col}, filters main query to show only {group_col} with above-average values from {year} onwards",
#                 'tables_needed': ['{table1}'],
#                 'requires': ['grouping_col', 'metric_col', 'date_col'],
#                 'verb_style': 'find'
#             },
#             {
#                 'sql': """SELECT t1.{group_col}, t1.{metric_col},
#     (SELECT COUNT(*) FROM {table2} t2 WHERE t2.{fk1} = t1.{pk1}) as related_count,
#     (SELECT AVG({metric_col2}) FROM {table2} t2 WHERE t2.{fk1} = t1.{pk1}) as avg_related_{metric_col2}
# FROM {table1} t1
# WHERE t1.{date_col} >= '{year}-01-01'
# AND (SELECT COUNT(*) FROM {table2} t2 WHERE t2.{fk1} = t1.{pk1}) > 0;""",
#                 'prompt_templates': [
#                     "{verb} {group_col} with counts and averages from related {table2} records since {year}",
#                     "{verb} {group_col} including relationship statistics from {table2}",
#                     "{verb} {table1} {group_col} with aggregated metrics from associated {table2} entries",
#                     "{verb} {group_col} showing {table2} relationships and average values",
#                     "{verb} {group_col} with correlated {table2} counts and means"
#                 ],
#                 'explanation': "Uses correlated subqueries to count related records in {table2} and calculate their average {metric_col2} for each record in {table1}, filters for {year} and non-zero relationships",
#                 'tables_needed': ['{table1}', '{table2}'],
#                 'requires': ['foreign_key_relationship', 'grouping_col', 'metric_col', 'date_col'],
#                 'verb_style': 'show'
#             },
#             {
#                 'sql': """SELECT {group_col}, {metric_col}, {date_col}
# FROM {table1} t1
# WHERE {metric_col} >= (
#     SELECT {metric_col}
#     FROM {table1} t2
#     WHERE t2.{group_col} = t1.{group_col}
#     ORDER BY {date_col} DESC
#     LIMIT 1
# )
# AND {date_col} >= '{year}-01-01'
# ORDER BY {metric_col} DESC;""",
#                 'prompt_templates': [
#                     "{verb} {group_col} where current {metric_col} matches or exceeds most recent value",
#                     "{verb} {group_col} with {metric_col} at or above their latest recorded level",
#                     "{verb} {group_col} maintaining or improving {metric_col} versus most recent",
#                     "{verb} {group_col} with {metric_col} meeting or surpassing latest observation",
#                     "{verb} {group_col} where {metric_col} equals or tops most recent entry"
#                 ],
#                 'explanation': "Correlated subquery finds most recent {metric_col} value for each {group_col}, main query filters to show only records where {metric_col} meets or exceeds this recent value, for records since {year}",
#                 'tables_needed': ['{table1}'],
#                 'requires': ['grouping_col', 'metric_col', 'date_col'],
#                 'verb_style': 'identify'
#             },
#             {
#                 'sql': """SELECT {group_col}, COUNT(*) as count, AVG({metric_col}) as avg_{metric_col}
# FROM {table1}
# WHERE {group_col} IN (
#     SELECT {group_col}
#     FROM {table1}
#     WHERE {date_col} >= '{year}-01-01'
#     GROUP BY {group_col}
#     HAVING SUM({metric_col}) > {threshold}
# )
# AND {date_col} >= '{year}-01-01'
# GROUP BY {group_col}
# ORDER BY avg_{metric_col} DESC;""",
#                 'prompt_templates': [
#                     "{verb} {group_col} statistics where total {metric_col} exceeds {threshold} for {year}",
#                     "{verb} counts and averages for {group_col} with cumulative {metric_col} above {threshold}",
#                     "{verb} aggregated {group_col} data filtered by total {metric_col} threshold",
#                     "{verb} {group_col} metrics for groups surpassing {threshold} in {metric_col}",
#                     "{verb} statistical breakdown of high-performing {group_col} by {metric_col}"
#                 ],
#                 'explanation': "Subquery identifies {group_col} values where sum of {metric_col} exceeds {threshold} in {year}, main query calculates count and average for those groups",
#                 'tables_needed': ['{table1}'],
#                 'requires': ['grouping_col', 'metric_col', 'date_col'],
#                 'verb_style': 'analyze'
#             },
#             {
#                 'sql': """SELECT {group_col}, {metric_col}, {date_col}
# FROM {table1} t1
# WHERE {metric_col} = (
#     SELECT MAX({metric_col})
#     FROM {table1} t2
#     WHERE t2.{group_col} = t1.{group_col}
#     AND t2.{date_col} >= '{year}-01-01'
# )
# AND {date_col} >= '{year}-01-01'
# ORDER BY {metric_col} DESC;""",
#                 'prompt_templates': [
#                     "{verb} records with maximum {metric_col} for each {group_col} in {year}",
#                     "{verb} highest {metric_col} entry per {group_col} during {year}",
#                     "{verb} peak {metric_col} values across all {group_col} for {year}",
#                     "{verb} {group_col} showing their maximum {metric_col} achievement in {year}",
#                     "{verb} top {metric_col} records by {group_col} for {year}"
#                 ],
#                 'explanation': "Correlated subquery finds maximum {metric_col} for each {group_col} in {year}, main query returns only records matching these maximum values",
#                 'tables_needed': ['{table1}'],
#                 'requires': ['grouping_col', 'metric_col', 'date_col'],
#                 'verb_style': 'find'
#             },
#             {
#                 'sql': """SELECT {group_col}, {metric_col}
# FROM {table1} t1
# WHERE {metric_col} > ALL (
#     SELECT {metric_col}
#     FROM {table1} t2
#     WHERE t2.{group_col} = t1.{group_col}
#     AND t2.{date_col} < t1.{date_col}
#     AND t2.{date_col} >= DATE_SUB(t1.{date_col}, INTERVAL 90 DAY)
# )
# AND {date_col} >= '{year}-01-01'
# ORDER BY {date_col}, {metric_col} DESC;""",
#                 'prompt_templates': [
#                     "{verb} {group_col} where {metric_col} exceeds all values in previous 90 days",
#                     "{verb} {group_col} reaching new 90-day highs in {metric_col}",
#                     "{verb} {group_col} with {metric_col} surpassing recent 3-month peak",
#                     "{verb} {group_col} breaking 90-day {metric_col} records",
#                     "{verb} {group_col} achieving quarterly {metric_col} peaks"
#                 ],
#                 'explanation': "Uses ALL comparison in subquery to find records where {metric_col} exceeds all values from the previous 90 days for that {group_col}, identifying new quarterly peaks, filtered for {year}",
#                 'tables_needed': ['{table1}'],
#                 'requires': ['grouping_col', 'metric_col', 'date_col'],
#                 'verb_style': 'detect'
#             },
#             {
#                 'sql': """SELECT {group_col}, {metric_col},
#     (SELECT COUNT(*) FROM {table1} t2 WHERE t2.{metric_col} > t1.{metric_col}) + 1 as overall_rank
# FROM {table1} t1
# WHERE {date_col} >= '{year}-01-01'
# ORDER BY overall_rank
# LIMIT {top_n};""",
#                 'prompt_templates': [
#                     "{verb} top {top_n} {group_col} by {metric_col} with calculated ranks for {year}",
#                     "{verb} highest-ranked {group_col} based on {metric_col} in {year}",
#                     "{verb} leading {top_n} {group_col} ordered by {metric_col} rank",
#                     "{verb} best {top_n} {group_col} with position rankings by {metric_col}",
#                     "{verb} {top_n} highest {group_col} showing their {metric_col} ranks"
#                 ],
#                 'explanation': "Subquery counts how many records have higher {metric_col} to calculate rank for each record, main query returns top {top_n} by this calculated rank for records since {year}",
#                 'tables_needed': ['{table1}'],
#                 'requires': ['grouping_col', 'metric_col', 'date_col'],
#                 'verb_style': 'rank'
#             }
#         ]

