from typing import List,Dict

def _get_window_function_templates() -> List[Dict]:
    """15 diverse window function templates"""
    return [
        # ============ TEMPLATE 1 ============
        {
            'sql': """SELECT {group_col}, {metric_col},
    RANK() OVER (PARTITION BY {partition_col} ORDER BY {metric_col} DESC) as rank,
    AVG({metric_col}) OVER (PARTITION BY {partition_col}) as avg_{metric_col}
FROM {table1}
WHERE {date_col} >= '{year}-01-01'
HAVING rank <= {top_n};""",
            'prompt_templates': [
                "{verb} top {top_n} records by {metric_col} within each {partition_col} since {year}",
                "{verb} highest {top_n} {group_col} per {partition_col} showing {metric_col}",
                "{verb} ranked {group_col} by {metric_col} in each {partition_col}, limited to top {top_n}",
                "{verb} leading {top_n} entries per {partition_col} based on {metric_col}",
                "{verb} best {top_n} performers by {metric_col} within {partition_col} groups"
            ],
            'explanation': "Uses RANK() to rank records by {metric_col} within each {partition_col}, calculates partition-level average {metric_col}, filters for {year} and shows top {top_n} per partition",
            'tables_needed': ['{table1}'],
            'requires': ['grouping_col', 'metric_col', 'date_col'],
            'verb_style': 'rank'
        },
        
        # ============ TEMPLATE 2 ============
        {
            'sql': """SELECT {group_col}, {date_col}, {metric_col},
    LAG({metric_col}, 1) OVER (PARTITION BY {group_col} ORDER BY {date_col}) as prev_{metric_col},
    LEAD({metric_col}, 1) OVER (PARTITION BY {group_col} ORDER BY {date_col}) as next_{metric_col},
    {metric_col} - LAG({metric_col}, 1) OVER (PARTITION BY {group_col} ORDER BY {date_col}) as change
FROM {table1}
WHERE {date_col} >= '{year}-01-01'
ORDER BY {group_col}, {date_col};""",
            'prompt_templates': [
                "{verb} {metric_col} trends for each {group_col} with previous and next period values",
                "{verb} period-over-period {metric_col} changes by {group_col} with adjacent comparisons",
                "{verb} time-series analysis of {metric_col} per {group_col} including lag and lead",
                "{verb} sequential {metric_col} progression for {group_col} showing transitions",
                "{verb} temporal {metric_col} patterns by {group_col} with period comparisons"
            ],
            'explanation': "Uses LAG() and LEAD() window functions to access previous and next period {metric_col} values for each {group_col}, calculates period-over-period change, orders chronologically for records since {year}",
            'tables_needed': ['{table1}'],
            'requires': ['grouping_col', 'metric_col', 'date_col'],
            'verb_style': 'analyze'
        },
        
        # ============ TEMPLATE 3 ============
        {
            'sql': """SELECT {group_col}, {metric_col},
    ROW_NUMBER() OVER (ORDER BY {metric_col} DESC) as overall_rank,
    PERCENT_RANK() OVER (PARTITION BY {partition_col} ORDER BY {metric_col}) as percentile,
    CUME_DIST() OVER (PARTITION BY {partition_col} ORDER BY {metric_col}) as cumulative_dist
FROM {table1}
WHERE {date_col} >= '{year}-01-01';""",
            'prompt_templates': [
                "{verb} {group_col} with overall ranking and percentile within {partition_col} for {year}",
                "{verb} {group_col} by {metric_col} showing absolute rank and relative percentile",
                "{verb} ranked {group_col} with percentile distribution per {partition_col}",
                "{verb} {group_col} with comprehensive ranking metrics across {partition_col}",
                "{verb} {group_col} performance showing global rank and local percentiles"
            ],
            'explanation': "Calculates overall ROW_NUMBER rank by {metric_col}, PERCENT_RANK percentile within each {partition_col}, and CUME_DIST cumulative distribution, filtered for records from {year}",
            'tables_needed': ['{table1}'],
            'requires': ['grouping_col', 'metric_col', 'date_col'],
            'verb_style': 'rank'
        },
        
        # ============ TEMPLATE 4 ============
        {
            'sql': """SELECT {group_col}, {metric_col}, {date_col},
    SUM({metric_col}) OVER (PARTITION BY {partition_col} ORDER BY {date_col} ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as running_total,
    AVG({metric_col}) OVER (PARTITION BY {partition_col} ORDER BY {date_col} ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) as moving_avg_3
FROM {table1}
WHERE {date_col} >= '{year}-01-01'
ORDER BY {partition_col}, {date_col};""",
            'prompt_templates': [
                "{verb} running total and 3-period moving average of {metric_col} by {partition_col}",
                "{verb} cumulative {metric_col} and rolling average for each {partition_col}",
                "{verb} {group_col} with running sum and moving average of {metric_col}",
                "{verb} progressive {metric_col} totals and trends per {partition_col}",
                "{verb} cumulative and smoothed {metric_col} metrics by {partition_col}"
            ],
            'explanation': "Calculates running total of {metric_col} from start to current row, and 3-period moving average (current + 2 preceding rows) within each {partition_col}, ordered chronologically for records since {year}",
            'tables_needed': ['{table1}'],
            'requires': ['grouping_col', 'metric_col', 'date_col'],
            'verb_style': 'calculate'
        },
        
        # ============ TEMPLATE 5 ============
        {
            'sql': """SELECT {group_col}, {partition_col}, {metric_col},
    NTILE(4) OVER (PARTITION BY {partition_col} ORDER BY {metric_col}) as quartile,
    DENSE_RANK() OVER (PARTITION BY {partition_col} ORDER BY {metric_col} DESC) as dense_rank,
    RANK() OVER (PARTITION BY {partition_col} ORDER BY {metric_col} DESC) as rank
FROM {table1}
WHERE {date_col} >= '{year}-01-01';""",
            'prompt_templates': [
                "{verb} {group_col} divided into quartiles by {metric_col} within each {partition_col}",
                "{verb} quartile distribution and ranking of {group_col} per {partition_col}",
                "{verb} {group_col} categorized into 4 groups based on {metric_col}",
                "{verb} {group_col} with quartile assignment and multiple ranking schemes",
                "{verb} {group_col} segmented by {metric_col} quartiles within {partition_col}"
            ],
            'explanation': "Uses NTILE(4) to divide records into quartiles based on {metric_col} within each {partition_col}, also provides DENSE_RANK and RANK for comprehensive ranking, filtered for records since {year}",
            'tables_needed': ['{table1}'],
            'requires': ['grouping_col', 'metric_col', 'date_col'],
            'verb_style': 'categorize'
        },
        
        # ============ TEMPLATE 6 ============
        {
            'sql': """SELECT {group_col}, {date_col}, {metric_col},
    FIRST_VALUE({metric_col}) OVER (PARTITION BY {partition_col} ORDER BY {date_col}) as period_start_value,
    LAST_VALUE({metric_col}) OVER (PARTITION BY {partition_col} ORDER BY {date_col} ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as period_end_value,
    {metric_col} - FIRST_VALUE({metric_col}) OVER (PARTITION BY {partition_col} ORDER BY {date_col}) as change_from_start
FROM {table1}
WHERE {date_col} >= '{year}-01-01';""",
            'prompt_templates': [
                "{verb} {group_col} showing {metric_col} changes from period start",
                "{verb} {metric_col} evolution per {group_col} relative to initial values",
                "{verb} {group_col} with first-to-current {metric_col} comparisons",
                "{verb} {metric_col} progression tracking for {group_col} from baseline",
                "{verb} {group_col} performance versus period starting {metric_col}"
            ],
            'explanation': "Uses FIRST_VALUE and LAST_VALUE to capture period start and end {metric_col} values for each {partition_col}, calculates change from start for every record, filtered for {year} onwards",
            'tables_needed': ['{table1}'],
            'requires': ['grouping_col', 'metric_col', 'date_col'],
            'verb_style': 'track'
        },
        
        # ============ TEMPLATE 7 ============
        {
            'sql': """SELECT {group_col}, {partition_col}, {metric_col}, {date_col},
    {metric_col} - AVG({metric_col}) OVER (PARTITION BY {partition_col}) as deviation_from_avg,
    CASE 
        WHEN {metric_col} > AVG({metric_col}) OVER (PARTITION BY {partition_col}) + STDDEV({metric_col}) OVER (PARTITION BY {partition_col}) THEN 'Above Avg'
        WHEN {metric_col} < AVG({metric_col}) OVER (PARTITION BY {partition_col}) - STDDEV({metric_col}) OVER (PARTITION BY {partition_col}) THEN 'Below Avg'
        ELSE 'Within Avg'
    END as performance_category
FROM {table1}
WHERE {date_col} >= '{year}-01-01';""",
            'prompt_templates': [
                "{verb} {group_col} categorized by {metric_col} deviation within {partition_col}",
                "{verb} performance classification of {group_col} relative to {partition_col} average",
                "{verb} {group_col} with statistical categorization by {metric_col}",
                "{verb} {group_col} segmented into performance tiers within {partition_col}",
                "{verb} {group_col} classified by {metric_col} relative to group statistics"
            ],
            'explanation': "Calculates deviation of each record's {metric_col} from its partition average, categorizes records as above/below/within average based on standard deviation thresholds, for records since {year}",
            'tables_needed': ['{table1}'],
            'requires': ['grouping_col', 'metric_col', 'date_col'],
            'verb_style': 'classify'
        },
        
        # ============ TEMPLATE 8 ============
        {
            'sql': """SELECT {group_col}, {date_col}, {metric_col},
    SUM({metric_col}) OVER (PARTITION BY {partition_col} ORDER BY {date_col}) / 
    SUM({metric_col}) OVER (PARTITION BY {partition_col}) * 100 as pct_of_total_to_date,
    ROW_NUMBER() OVER (PARTITION BY {partition_col} ORDER BY {date_col}) as period_number
FROM {table1}
WHERE {date_col} >= '{year}-01-01';""",
            'prompt_templates': [
                "{verb} cumulative percentage of total {metric_col} by {group_col}",
                "{verb} running {metric_col} contribution as percentage per {group_col}",
                "{verb} progressive {metric_col} share analysis by {partition_col}",
                "{verb} {group_col} showing cumulative {metric_col} as portion of total",
                "{verb} period-by-period {metric_col} accumulation percentage for {group_col}"
            ],
            'explanation': "Calculates running sum of {metric_col} as percentage of partition total, assigns period numbers, showing how much of total {metric_col} has accumulated by each point in time within {partition_col}, for records since {year}",
            'tables_needed': ['{table1}'],
            'requires': ['grouping_col', 'metric_col', 'date_col'],
            'verb_style': 'monitor'
        },
        
        # ============ TEMPLATE 9 ============
        {
            'sql': """SELECT {group_col}, {partition_col}, {metric_col}, {date_col},
    {metric_col} / SUM({metric_col}) OVER (PARTITION BY {partition_col}) * 100 as pct_of_partition,
    SUM({metric_col}) OVER (PARTITION BY {partition_col} ORDER BY {metric_col} DESC) / 
    SUM({metric_col}) OVER (PARTITION BY {partition_col}) * 100 as cumulative_pct
FROM {table1}
WHERE {date_col} >= '{year}-01-01'
ORDER BY {partition_col}, {metric_col} DESC;""",
            'prompt_templates': [
                "{verb} contribution percentage and cumulative share by {group_col}",
                "{verb} {group_col} showing individual and running percentages",
                "{verb} partition-wise {metric_col} distribution with cumulative view",
                "{verb} {group_col} contribution analysis within {partition_col}",
                "{verb} progressive percentage accumulation by {group_col}"
            ],
            'explanation': "Calculates each record's percentage of partition total and cumulative percentage when ordered by {metric_col}, showing contribution patterns within {partition_col}",
            'tables_needed': ['{table1}'],
            'requires': ['grouping_col', 'metric_col', 'date_col'],
            'verb_style': 'calculate'
        },
        
        # ============ TEMPLATE 10 ============
        {
            'sql': """SELECT {group_col}, {metric_col}, {date_col},
    AVG({metric_col}) OVER (ORDER BY {date_col} ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as moving_avg_7day,
    {metric_col} - AVG({metric_col}) OVER (ORDER BY {date_col} ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as deviation_from_ma
FROM {table1}
WHERE {date_col} >= '{year}-01-01'
ORDER BY {date_col};""",
            'prompt_templates': [
                "{verb} 7-day moving average with deviation analysis",
                "{verb} {metric_col} trends with smoothed moving average comparison",
                "{verb} {group_col} showing short-term average and variations",
                "{verb} rolling 7-day mean and deviation patterns",
                "{verb} {metric_col} relative to recent 7-day average"
            ],
            'explanation': "Calculates 7-day moving average of {metric_col} and deviation from this average for each record, useful for trend smoothing and anomaly detection",
            'tables_needed': ['{table1}'],
            'requires': ['grouping_col', 'metric_col', 'date_col'],
            'verb_style': 'analyze'
        },
        
        # ============ TEMPLATE 11 ============
        {
            'sql': """SELECT {group_col}, {partition_col}, {metric_col},
    NTILE(10) OVER (PARTITION BY {partition_col} ORDER BY {metric_col}) as decile,
    MIN({metric_col}) OVER (PARTITION BY {partition_col}, NTILE(10) OVER (PARTITION BY {partition_col} ORDER BY {metric_col})) as decile_min,
    MAX({metric_col}) OVER (PARTITION BY {partition_col}, NTILE(10) OVER (PARTITION BY {partition_col} ORDER BY {metric_col})) as decile_max
FROM {table1}
WHERE {date_col} >= '{year}-01-01';""",
            'prompt_templates': [
                "{verb} decile distribution of {metric_col} within {partition_col}",
                "{verb} {group_col} segmented into 10 groups by {metric_col}",
                "{verb} {partition_col} with {metric_col} decile boundaries",
                "{verb} 10-tier classification of {group_col} performance",
                "{verb} {group_col} distributed across {metric_col} deciles"
            ],
            'explanation': "Divides records into 10 equal groups (deciles) based on {metric_col} within each {partition_col}, shows min/max values for each decile",
            'tables_needed': ['{table1}'],
            'requires': ['grouping_col', 'metric_col', 'date_col'],
            'verb_style': 'segment'
        },
        
        # ============ TEMPLATE 12 ============
        {
            'sql': """SELECT {group_col}, {date_col}, {metric_col},
    LEAD({metric_col}, 1) OVER (PARTITION BY {group_col} ORDER BY {date_col}) - {metric_col} as next_period_change,
    CASE 
        WHEN LEAD({metric_col}, 1) OVER (PARTITION BY {group_col} ORDER BY {date_col}) > {metric_col} THEN 'Increasing'
        WHEN LEAD({metric_col}, 1) OVER (PARTITION BY {group_col} ORDER BY {date_col}) < {metric_col} THEN 'Decreasing'
        ELSE 'Stable'
    END as trend_direction
FROM {table1}
WHERE {date_col} >= '{year}-01-01'
ORDER BY {group_col}, {date_col};""",
            'prompt_templates': [
                "{verb} period-to-period {metric_col} changes with trend direction",
                "{verb} {group_col} showing next-period forecast and movement",
                "{verb} forward-looking {metric_col} changes by {group_col}",
                "{verb} {group_col} with upcoming period comparisons",
                "{verb} next-period {metric_col} deltas and trend classification"
            ],
            'explanation': "Uses LEAD to compare current {metric_col} with next period's value, calculates change magnitude and classifies trend direction for each {group_col}",
            'tables_needed': ['{table1}'],
            'requires': ['grouping_col', 'metric_col', 'date_col'],
            'verb_style': 'forecast'
        },
        
        # ============ TEMPLATE 13 ============
        {
            'sql': """SELECT {group_col}, {partition_col}, {metric_col},
    ROW_NUMBER() OVER (PARTITION BY {partition_col} ORDER BY {metric_col} DESC) as row_num,
    COUNT(*) OVER (PARTITION BY {partition_col}) as partition_size,
    ROUND(ROW_NUMBER() OVER (PARTITION BY {partition_col} ORDER BY {metric_col} DESC) * 100.0 / 
        COUNT(*) OVER (PARTITION BY {partition_col}), 2) as percentile_rank
FROM {table1}
WHERE {date_col} >= '{year}-01-01'
AND ROW_NUMBER() OVER (PARTITION BY {partition_col} ORDER BY {metric_col} DESC) <= {top_n};""",
            'prompt_templates': [
                "{verb} top {top_n} {group_col} per {partition_col} with percentile ranks",
                "{verb} leading {group_col} showing relative position in {partition_col}",
                "{verb} {top_n} highest {group_col} with ranking context",
                "{verb} top performers per {partition_col} with percentile scores",
                "{verb} {group_col} elite tier with partition-relative ranks"
            ],
            'explanation': "Ranks {group_col} within each {partition_col}, calculates percentile rank, filters to show top {top_n} per partition with their relative standing",
            'tables_needed': ['{table1}'],
            'requires': ['grouping_col', 'metric_col', 'date_col'],
            'verb_style': 'rank'
        },
        
        # ============ TEMPLATE 14 ============
        {
            'sql': """SELECT {group_col}, {date_col}, {metric_col},
    SUM({metric_col}) OVER (PARTITION BY {group_col} ORDER BY {date_col} 
                            ROWS BETWEEN CURRENT ROW AND 2 FOLLOWING) as forward_3period_sum,
    AVG({metric_col}) OVER (PARTITION BY {group_col} ORDER BY {date_col} 
                            ROWS BETWEEN CURRENT ROW AND 2 FOLLOWING) as forward_3period_avg
FROM {table1}
WHERE {date_col} >= '{year}-01-01'
ORDER BY {group_col}, {date_col};""",
            'prompt_templates': [
                "{verb} forward-looking 3-period aggregates for {group_col}",
                "{verb} {group_col} with next 3 periods summed and averaged",
                "{verb} upcoming {metric_col} totals by {group_col}",
                "{verb} {group_col} showing near-term {metric_col} accumulation",
                "{verb} 3-period forward window metrics per {group_col}"
            ],
            'explanation': "Calculates forward-looking 3-period sum and average of {metric_col} (current + next 2 periods) for each {group_col}, useful for short-term forecasting",
            'tables_needed': ['{table1}'],
            'requires': ['grouping_col', 'metric_col', 'date_col'],
            'verb_style': 'project'
        },
        
        # ============ TEMPLATE 15 ============
        {
            'sql': """SELECT {group_col}, {partition_col}, {metric_col}, {date_col},
    {metric_col} - MIN({metric_col}) OVER (PARTITION BY {partition_col}) as distance_from_min,
    MAX({metric_col}) OVER (PARTITION BY {partition_col}) - {metric_col} as distance_from_max,
    ({metric_col} - MIN({metric_col}) OVER (PARTITION BY {partition_col})) / 
    NULLIF(MAX({metric_col}) OVER (PARTITION BY {partition_col}) - MIN({metric_col}) OVER (PARTITION BY {partition_col}), 0) as normalized_score
FROM {table1}
WHERE {date_col} >= '{year}-01-01'
ORDER BY {partition_col}, normalized_score DESC;""",
            'prompt_templates': [
                "{verb} normalized {metric_col} scores within {partition_col} ranges",
                "{verb} {group_col} with distance from partition extremes",
                "{verb} min-max normalized {metric_col} by {partition_col}",
                "{verb} {group_col} showing relative position in {partition_col} range",
                "{verb} scaled {metric_col} values per {partition_col}"
            ],
            'explanation': "Calculates distance from partition minimum and maximum, applies min-max normalization to scale {metric_col} to 0-1 range within each {partition_col}",
            'tables_needed': ['{table1}'],
            'requires': ['grouping_col', 'metric_col', 'date_col'],
            'verb_style': 'normalize'
        }
    ]