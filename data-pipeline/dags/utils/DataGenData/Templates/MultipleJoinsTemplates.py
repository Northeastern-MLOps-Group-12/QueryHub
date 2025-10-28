from typing import List, Dict

def _get_multiple_join_templates() -> List[Dict]:
    """15 diverse multiple join templates"""
    return [
        # ============ TEMPLATE 1 ============
        {
            'sql': """SELECT t1.{group_col1}, t2.{group_col2}, SUM(t1.{metric_col}) as total_{metric_col}
FROM {table1} t1
INNER JOIN {table2} t2 ON t1.{fk1} = t2.{pk2}
WHERE t1.{date_col} >= '{year}-01-01'
GROUP BY t1.{group_col1}, t2.{group_col2}
ORDER BY total_{metric_col} DESC
LIMIT {top_n};""",
            'prompt_templates': [
                "{verb} top {top_n} combinations of {group_col1} and {group_col2} by total {metric_col} since {year}",
                "{verb} highest {top_n} {group_col1}-{group_col2} pairs by {metric_col} for {year}",
                "{verb} leading {top_n} {group_col1} and {group_col2} combinations based on {metric_col}",
                "{verb} best performing {top_n} {group_col1}-{group_col2} groups by {metric_col}",
                "{verb} {top_n} strongest {group_col1} and {group_col2} pairings in {metric_col}"
            ],
            'explanation': "Joins {table1} and {table2}, aggregates {metric_col} by combinations of {group_col1} and {group_col2}, filters for {year} onwards, and returns top {top_n} results",
            'tables_needed': ['{table1}', '{table2}'],
            'requires': ['foreign_key_relationship', 'metric_col', 'grouping_col', 'date_col'],
            'verb_style': 'show'
        },
        
        # ============ TEMPLATE 2 ============
        {
            'sql': """SELECT t1.{group_col1}, t2.{group_col2}, AVG(t1.{metric_col}) as avg_{metric_col}, COUNT(*) as record_count
FROM {table1} t1
LEFT JOIN {table2} t2 ON t1.{fk1} = t2.{pk2}
WHERE t2.{pk2} IS NOT NULL
GROUP BY t1.{group_col1}, t2.{group_col2}
HAVING COUNT(*) >= 3
ORDER BY avg_{metric_col} DESC;""",
            'prompt_templates': [
                "{verb} {group_col1} and {group_col2} pairs with average {metric_col} having at least 3 records",
                "{verb} combinations of {group_col1}-{group_col2} where {metric_col} average is calculated from 3+ entries",
                "{verb} grouped {group_col1} and {group_col2} with mean {metric_col} from sufficient data",
                "{verb} {group_col1}-{group_col2} statistics filtered by minimum record count",
                "{verb} {group_col1} and {group_col2} averages with meaningful sample sizes"
            ],
            'explanation': "Left joins {table1} with {table2}, calculates average {metric_col} for each {group_col1}-{group_col2} combination, filters to show only groups with 3 or more records",
            'tables_needed': ['{table1}', '{table2}'],
            'requires': ['foreign_key_relationship', 'metric_col', 'grouping_col'],
            'verb_style': 'analyze'
        },
        
        # ============ TEMPLATE 3 ============
        {
            'sql': """SELECT t1.{group_col1}, COUNT(DISTINCT t2.{pk2}) as related_count, MAX(t1.{metric_col}) as max_{metric_col}
FROM {table1} t1
INNER JOIN {table2} t2 ON t1.{fk1} = t2.{pk2}
WHERE t1.{date_col} >= '{year}-01-01'
GROUP BY t1.{group_col1}
HAVING MAX(t1.{metric_col}) > {threshold};""",
            'prompt_templates': [
                "{verb} {group_col1} with maximum {metric_col} above {threshold} and their relationship counts",
                "{verb} {group_col1} where max {metric_col} exceeds {threshold}, including distinct {table2} associations",
                "{verb} high-performing {group_col1} based on {metric_col} threshold with relationship metrics",
                "{verb} {group_col1} surpassing {threshold} in {metric_col} with related {table2} counts",
                "{verb} top {group_col1} by {metric_col} threshold showing {table2} connection frequency"
            ],
            'explanation': "Joins tables to count distinct related records from {table2} for each {group_col1}, finds maximum {metric_col}, filters groups where max exceeds {threshold} for records since {year}",
            'tables_needed': ['{table1}', '{table2}'],
            'requires': ['foreign_key_relationship', 'metric_col', 'grouping_col', 'date_col'],
            'verb_style': 'find'
        },
        
        # ============ TEMPLATE 4 ============
        {
            'sql': """SELECT t1.{group_col1}, t2.{group_col2}, 
    SUM(t1.{metric_col}) as total, 
    ROUND(SUM(t1.{metric_col}) * 100.0 / SUM(SUM(t1.{metric_col})) OVER (), 2) as percentage
FROM {table1} t1
INNER JOIN {table2} t2 ON t1.{fk1} = t2.{pk2}
WHERE t1.{date_col} >= '{year}-01-01'
GROUP BY t1.{group_col1}, t2.{group_col2}
HAVING SUM(t1.{metric_col}) > {threshold}
ORDER BY percentage DESC;""",
            'prompt_templates': [
                "{verb} {group_col1}-{group_col2} distribution by {metric_col} with percentages for {year}",
                "{verb} relative contribution of each {group_col1}-{group_col2} pair to total {metric_col}",
                "{verb} {group_col1} and {group_col2} breakdown with percentage shares exceeding threshold",
                "{verb} proportional {metric_col} analysis by {group_col1} and {group_col2}",
                "{verb} {group_col1}-{group_col2} composition showing {metric_col} percentages"
            ],
            'explanation': "Joins {table1} and {table2}, calculates both absolute totals and percentage contribution of each {group_col1}-{group_col2} combination to overall {metric_col}, filters for {year} and values above {threshold}",
            'tables_needed': ['{table1}', '{table2}'],
            'requires': ['foreign_key_relationship', 'metric_col', 'grouping_col', 'date_col'],
            'verb_style': 'calculate'
        },
        
        # ============ TEMPLATE 5 ============
        {
            'sql': """SELECT t1.{group_col1}, t2.{group_col2},
    AVG(t1.{metric_col}) as avg_{metric_col},
    MIN(t1.{metric_col}) as min_{metric_col},
    MAX(t1.{metric_col}) as max_{metric_col},
    STDDEV(t1.{metric_col}) as stddev_{metric_col}
FROM {table1} t1
RIGHT JOIN {table2} t2 ON t1.{fk1} = t2.{pk2}
WHERE t1.{date_col} IS NOT NULL
GROUP BY t1.{group_col1}, t2.{group_col2}
ORDER BY avg_{metric_col} DESC;""",
            'prompt_templates': [
                "{verb} statistical summary of {metric_col} by {group_col1} and {group_col2}",
                "{verb} comprehensive {metric_col} metrics across {group_col1}-{group_col2} combinations",
                "{verb} complete {metric_col} statistics grouped by {group_col1} and {group_col2}",
                "{verb} {group_col1} and {group_col2} with full {metric_col} distribution metrics",
                "{verb} detailed {metric_col} analysis by {group_col1}-{group_col2} pairs"
            ],
            'explanation': "Right joins {table1} and {table2} to calculate comprehensive statistics (mean, minimum, maximum, standard deviation) of {metric_col} for each {group_col1}-{group_col2} combination, including all {table2} records",
            'tables_needed': ['{table1}', '{table2}'],
            'requires': ['foreign_key_relationship', 'metric_col', 'grouping_col', 'date_col'],
            'verb_style': 'summarize'
        },
        
        # ============ TEMPLATE 6 ============
        {
            'sql': """SELECT t1.{group_col1}, t2.{group_col2},
    COUNT(DISTINCT t1.{pk1}) as unique_records,
    SUM(t1.{metric_col}) as total_metric
FROM {table1} t1
FULL OUTER JOIN {table2} t2 ON t1.{fk1} = t2.{pk2}
WHERE t1.{date_col} >= '{year}-01-01' OR t2.{date_col} >= '{year}-01-01'
GROUP BY t1.{group_col1}, t2.{group_col2}
ORDER BY total_metric DESC NULLS LAST;""",
            'prompt_templates': [
                "{verb} complete {group_col1}-{group_col2} relationships with metrics",
                "{verb} all {group_col1} and {group_col2} combinations including unmatched",
                "{verb} comprehensive {group_col1}-{group_col2} analysis with nulls",
                "{verb} {group_col1} and {group_col2} including orphaned records",
                "{verb} full {group_col1}-{group_col2} mapping with {metric_col} totals"
            ],
            'explanation': "Full outer join preserves all records from both tables, aggregates {metric_col} by both grouping columns, includes unmatched records from either side",
            'tables_needed': ['{table1}', '{table2}'],
            'requires': ['foreign_key_relationship', 'metric_col', 'grouping_col', 'date_col'],
            'verb_style': 'show'
        },
        
        # ============ TEMPLATE 7 ============
        {
            'sql': """SELECT t1.{group_col1}, t2.{group_col2},
    t1.{metric_col} as table1_metric,
    AVG(t1.{metric_col}) OVER (PARTITION BY t2.{group_col2}) as avg_by_table2
FROM {table1} t1
INNER JOIN {table2} t2 ON t1.{fk1} = t2.{pk2}
WHERE t1.{date_col} >= '{year}-01-01'
AND t1.{metric_col} > (SELECT AVG({metric_col}) FROM {table1})
ORDER BY t2.{group_col2}, t1.{metric_col} DESC;""",
            'prompt_templates': [
                "{verb} above-average {group_col1} with {group_col2} context",
                "{verb} high-performing {group_col1} grouped by {group_col2}",
                "{verb} {group_col1} exceeding mean within {group_col2} groups",
                "{verb} top {group_col1} by {metric_col} categorized by {group_col2}",
                "{verb} {group_col1}-{group_col2} pairs where {metric_col} beats average"
            ],
            'explanation': "Joins tables to show {group_col1} with above-average {metric_col}, includes window function for {group_col2}-level averages, filtered for {year}",
            'tables_needed': ['{table1}', '{table2}'],
            'requires': ['foreign_key_relationship', 'metric_col', 'grouping_col', 'date_col'],
            'verb_style': 'find'
        },
        
        # ============ TEMPLATE 8 ============
        {
            'sql': """SELECT t2.{group_col2}, 
    COUNT(DISTINCT t1.{pk1}) as related_count,
    SUM(CASE WHEN t1.{metric_col} > {threshold} THEN 1 ELSE 0 END) as high_value_count,
    ROUND(SUM(CASE WHEN t1.{metric_col} > {threshold} THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as high_value_pct
FROM {table2} t2
LEFT JOIN {table1} t1 ON t2.{pk2} = t1.{fk1}
WHERE t1.{date_col} >= '{year}-01-01' OR t1.{date_col} IS NULL
GROUP BY t2.{group_col2}
HAVING COUNT(DISTINCT t1.{pk1}) > 0;""",
            'prompt_templates': [
                "{verb} {group_col2} with high-value record distribution",
                "{verb} proportion of records above {threshold} per {group_col2}",
                "{verb} {group_col2} showing threshold achievement rates",
                "{verb} {group_col2} with percentage exceeding {threshold}",
                "{verb} high-value composition by {group_col2}"
            ],
            'explanation': "Left joins to count total and high-value (>{threshold}) records per {group_col2}, calculates percentage of records exceeding threshold",
            'tables_needed': ['{table1}', '{table2}'],
            'requires': ['foreign_key_relationship', 'metric_col', 'grouping_col', 'date_col'],
            'verb_style': 'analyze'
        },
        
        # ============ TEMPLATE 9 ============
        {
            'sql': """SELECT t1.{group_col1}, t2.{group_col2},
    MIN(t1.{date_col}) as first_date,
    MAX(t1.{date_col}) as last_date,
    DATEDIFF(MAX(t1.{date_col}), MIN(t1.{date_col})) as days_span,
    SUM(t1.{metric_col}) as total_metric
FROM {table1} t1
INNER JOIN {table2} t2 ON t1.{fk1} = t2.{pk2}
WHERE t1.{date_col} >= '{year}-01-01'
GROUP BY t1.{group_col1}, t2.{group_col2}
HAVING DATEDIFF(MAX(t1.{date_col}), MIN(t1.{date_col})) >= 30
ORDER BY days_span DESC;""",
            'prompt_templates': [
                "{verb} {group_col1}-{group_col2} pairs with sustained activity",
                "{verb} long-running relationships between {group_col1} and {group_col2}",
                "{verb} {group_col1} and {group_col2} with extended engagement periods",
                "{verb} enduring {group_col1}-{group_col2} associations",
                "{verb} {group_col1} and {group_col2} showing 30+ day activity spans"
            ],
            'explanation': "Joins tables to find {group_col1}-{group_col2} combinations with activity spanning at least 30 days, showing first/last dates and totals",
            'tables_needed': ['{table1}', '{table2}'],
            'requires': ['foreign_key_relationship', 'metric_col', 'grouping_col', 'date_col'],
            'verb_style': 'identify'
        },
        
        # ============ TEMPLATE 10 ============
        {
            'sql': """SELECT t2.{group_col2}, 
    t1.{group_col1},
    t1.{metric_col},
    RANK() OVER (PARTITION BY t2.{group_col2} ORDER BY t1.{metric_col} DESC) as rank_within_group
FROM {table1} t1
INNER JOIN {table2} t2 ON t1.{fk1} = t2.{pk2}
WHERE t1.{date_col} >= '{year}-01-01'
QUALIFY rank_within_group <= {top_n};""",
            'prompt_templates': [
                "{verb} top {top_n} {group_col1} within each {group_col2} by {metric_col}",
                "{verb} highest {top_n} performers per {group_col2} category",
                "{verb} leading {top_n} {group_col1} grouped by {group_col2}",
                "{verb} best {top_n} {group_col1} in every {group_col2}",
                "{verb} {top_n} top-ranked {group_col1} per {group_col2} group"
            ],
            'explanation': "Ranks {group_col1} by {metric_col} within each {group_col2} partition, uses QUALIFY to filter top {top_n} per group efficiently",
            'tables_needed': ['{table1}', '{table2}'],
            'requires': ['foreign_key_relationship', 'metric_col', 'grouping_col', 'date_col'],
            'verb_style': 'rank'
        },
        
        # ============ TEMPLATE 11 ============
        {
            'sql': """SELECT t1.{group_col1}, 
    COUNT(DISTINCT t2.{group_col2}) as distinct_categories,
    STRING_AGG(DISTINCT t2.{group_col2}, ', ') as category_list,
    AVG(t1.{metric_col}) as avg_metric
FROM {table1} t1
LEFT JOIN {table2} t2 ON t1.{fk1} = t2.{pk2}
WHERE t1.{date_col} >= '{year}-01-01'
GROUP BY t1.{group_col1}
HAVING COUNT(DISTINCT t2.{group_col2}) >= 2
ORDER BY distinct_categories DESC;""",
            'prompt_templates': [
                "{verb} {group_col1} associated with multiple {group_col2} categories",
                "{verb} multi-category {group_col1} with {group_col2} lists",
                "{verb} {group_col1} spanning 2+ {group_col2} types",
                "{verb} diversified {group_col1} across {group_col2} categories",
                "{verb} {group_col1} with varied {group_col2} associations"
            ],
            'explanation': "Left joins to find {group_col1} linked to multiple {group_col2} values, aggregates distinct categories and creates comma-separated list",
            'tables_needed': ['{table1}', '{table2}'],
            'requires': ['foreign_key_relationship', 'metric_col', 'grouping_col', 'date_col'],
            'verb_style': 'find'
        },
        
        # ============ TEMPLATE 12 ============
        {
            'sql': """SELECT t2.{group_col2},
    SUM(CASE WHEN t1.{metric_col} BETWEEN 0 AND {threshold}/3 THEN 1 ELSE 0 END) as low_tier,
    SUM(CASE WHEN t1.{metric_col} BETWEEN {threshold}/3 AND 2*{threshold}/3 THEN 1 ELSE 0 END) as mid_tier,
    SUM(CASE WHEN t1.{metric_col} > 2*{threshold}/3 THEN 1 ELSE 0 END) as high_tier
FROM {table1} t1
INNER JOIN {table2} t2 ON t1.{fk1} = t2.{pk2}
WHERE t1.{date_col} >= '{year}-01-01'
GROUP BY t2.{group_col2};""",
            'prompt_templates': [
                "{verb} tier distribution of {metric_col} across {group_col2}",
                "{verb} {group_col2} with segmented {metric_col} breakdown",
                "{verb} low/mid/high tier counts per {group_col2}",
                "{verb} {metric_col} stratification by {group_col2}",
                "{verb} {group_col2} showing tiered {metric_col} composition"
            ],
            'explanation': "Joins tables and segments {metric_col} into three tiers based on {threshold}, counts records in each tier per {group_col2}",
            'tables_needed': ['{table1}', '{table2}'],
            'requires': ['foreign_key_relationship', 'metric_col', 'grouping_col', 'date_col'],
            'verb_style': 'classify'
        },
        
        # ============ TEMPLATE 13 ============
        {
            'sql': """SELECT t1.{group_col1}, t2.{group_col2},
    t1.{metric_col},
    t1.{metric_col} - AVG(t1.{metric_col}) OVER (PARTITION BY t2.{group_col2}) as deviation_from_group_avg
FROM {table1} t1
INNER JOIN {table2} t2 ON t1.{fk1} = t2.{pk2}
WHERE t1.{date_col} >= '{year}-01-01'
AND ABS(t1.{metric_col} - AVG(t1.{metric_col}) OVER (PARTITION BY t2.{group_col2})) > {threshold}
ORDER BY ABS(deviation_from_group_avg) DESC;""",
            'prompt_templates': [
                "{verb} outlier {group_col1} deviating from {group_col2} averages",
                "{verb} {group_col1} with unusual {metric_col} within {group_col2} groups",
                "{verb} extreme {group_col1} values compared to {group_col2} norms",
                "{verb} {group_col1} showing significant deviation in {metric_col}",
                "{verb} anomalous {group_col1} by {group_col2} category"
            ],
            'explanation': "Joins tables and identifies {group_col1} records where {metric_col} deviates more than {threshold} from their {group_col2} group average",
            'tables_needed': ['{table1}', '{table2}'],
            'requires': ['foreign_key_relationship', 'metric_col', 'grouping_col', 'date_col'],
            'verb_style': 'detect'
        },
        
        # ============ TEMPLATE 14 ============
        {
            'sql': """SELECT t2.{group_col2},
    MIN(t1.{metric_col}) as min_value,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY t1.{metric_col}) as q1,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY t1.{metric_col}) as median,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY t1.{metric_col}) as q3,
    MAX(t1.{metric_col}) as max_value
FROM {table1} t1
INNER JOIN {table2} t2 ON t1.{fk1} = t2.{pk2}
WHERE t1.{date_col} >= '{year}-01-01'
GROUP BY t2.{group_col2}
ORDER BY median DESC;""",
            'prompt_templates': [
                "{verb} five-number summary of {metric_col} by {group_col2}",
                "{verb} quartile analysis of {metric_col} across {group_col2}",
                "{verb} distribution statistics for {metric_col} per {group_col2}",
                "{verb} {group_col2} with complete {metric_col} percentile breakdown",
                "{verb} statistical profile of {metric_col} by {group_col2}"
            ],
            'explanation': "Joins tables and calculates five-number summary (min, Q1, median, Q3, max) of {metric_col} for each {group_col2}, providing comprehensive distribution view",
            'tables_needed': ['{table1}', '{table2}'],
            'requires': ['foreign_key_relationship', 'metric_col', 'grouping_col', 'date_col'],
            'verb_style': 'summarize'
        },
        
        # ============ TEMPLATE 15 ============
        {
            'sql': """WITH recent_activity AS (
    SELECT t1.{group_col1}, t2.{group_col2}, MAX(t1.{date_col}) as last_activity
    FROM {table1} t1
    INNER JOIN {table2} t2 ON t1.{fk1} = t2.{pk2}
    GROUP BY t1.{group_col1}, t2.{group_col2}
)
SELECT {group_col2}, COUNT(*) as active_count
FROM recent_activity
WHERE last_activity >= DATE_SUB(CURDATE(), INTERVAL 90 DAY)
GROUP BY {group_col2}
ORDER BY active_count DESC;""",
            'prompt_templates': [
                "{verb} recently active {group_col1} counts by {group_col2}",
                "{verb} {group_col2} with recent 90-day engagement levels",
                "{verb} active {group_col1} distribution across {group_col2}",
                "{verb} {group_col2} ranked by recent activity volume",
                "{verb} current engagement metrics by {group_col2}"
            ],
            'explanation': "Joins tables to identify {group_col1} active in last 90 days, counts active records per {group_col2} for recency analysis",
            'tables_needed': ['{table1}', '{table2}'],
            'requires': ['foreign_key_relationship', 'grouping_col', 'date_col'],
            'verb_style': 'count'
        }
    ]