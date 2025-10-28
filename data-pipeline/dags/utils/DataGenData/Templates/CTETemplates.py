from typing import List,Dict

def _get_cte_templates() -> List[Dict]:
    """15 diverse CTE templates"""
    return [
        # ============ TEMPLATE 1 ============
        {
            'sql': """WITH {cte_name} AS (
    SELECT {group_col}, SUM({metric_col}) as total_{metric_col}
    FROM {table1}
    WHERE {date_col} >= '{year}-01-01'
    GROUP BY {group_col}
)
SELECT {group_col}, total_{metric_col}
FROM {cte_name}
WHERE total_{metric_col} > {threshold}
ORDER BY total_{metric_col} DESC;""",
            'prompt_templates': [
                "{verb} {group_col} with total {metric_col} exceeding {threshold} in {year}",
                "{verb} {group_col} where total {metric_col} is greater than {threshold} for {year}",
                "{verb} all {group_col} that have total {metric_col} above {threshold} since {year}",
                "{verb} {group_col} with cumulative {metric_col} over {threshold} during {year}",
                "{verb} {group_col} having aggregate {metric_col} surpassing {threshold} in {year}"
            ],
            'explanation': "Uses a CTE to aggregate {metric_col} by {group_col} for records from {year}, then filters results to show only those with totals exceeding {threshold}",
            'tables_needed': ['{table1}'],
            'requires': ['grouping_col', 'metric_col', 'date_col'],
            'verb_style': 'find'
        },
        
        # ============ TEMPLATE 2 ============
        {
            'sql': """WITH ranked_{table1} AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY {group_col} ORDER BY {metric_col} DESC) as rank
    FROM {table1}
),
top_{table1} AS (
    SELECT * FROM ranked_{table1} WHERE rank <= {top_n}
)
SELECT {group_col}, AVG({metric_col}) as avg_{metric_col}, COUNT(*) as count
FROM top_{table1}
GROUP BY {group_col};""",
            'prompt_templates': [
                "{verb} average {metric_col} for top {top_n} records per {group_col}",
                "{verb} mean {metric_col} across the highest {top_n} entries in each {group_col}",
                "{verb} the average {metric_col} for the top {top_n} items grouped by {group_col}",
                "{verb} {metric_col} averages from the leading {top_n} records by {group_col}",
                "{verb} mean {metric_col} values for the best {top_n} in each {group_col}"
            ],
            'explanation': "Uses multiple CTEs: first ranks all records by {metric_col} within each {group_col} partition, filters to keep only top {top_n}, then calculates average {metric_col} per group",
            'tables_needed': ['{table1}'],
            'requires': ['grouping_col', 'metric_col'],
            'verb_style': 'calculate'
        },
        
        # ============ TEMPLATE 3 ============
        {
            'sql': """WITH summary AS (
    SELECT {group_col}, COUNT(*) as count, MAX({metric_col}) as max_{metric_col}, MIN({metric_col}) as min_{metric_col}
    FROM {table1}
    WHERE {date_col} BETWEEN '{year}-01-01' AND '{year}-12-31'
    GROUP BY {group_col}
    HAVING COUNT(*) > 2
)
SELECT * FROM summary
ORDER BY max_{metric_col} DESC;""",
            'prompt_templates': [
                "{verb} {group_col} with statistics for {year} having at least 3 records",
                "{verb} summary metrics by {group_col} for {year} with sufficient data",
                "{verb} {group_col} aggregates during {year} where count exceeds 2",
                "{verb} statistical breakdown of {group_col} in {year} with multiple entries",
                "{verb} {group_col} summary for {year} filtering low-count groups"
            ],
            'explanation': "CTE calculates count, maximum, and minimum {metric_col} per {group_col} for {year}, filters groups with more than 2 records using HAVING, then orders by max value",
            'tables_needed': ['{table1}'],
            'requires': ['grouping_col', 'metric_col', 'date_col'],
            'verb_style': 'show'
        },
        
        # ============ TEMPLATE 4 ============
        {
            'sql': """WITH recent_data AS (
    SELECT {group_col}, {metric_col}, {date_col}
    FROM {table1}
    WHERE {date_col} >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
),
aggregated AS (
    SELECT {group_col}, AVG({metric_col}) as avg_{metric_col}, MIN({metric_col}) as min_{metric_col}, MAX({metric_col}) as max_{metric_col}
    FROM recent_data
    GROUP BY {group_col}
)
SELECT * FROM aggregated
WHERE avg_{metric_col} > min_{metric_col} * 1.5;""",
            'prompt_templates': [
                "{verb} {group_col} where average {metric_col} exceeds 1.5x minimum in last 6 months",
                "{verb} recent {group_col} with significant {metric_col} variation",
                "{verb} {group_col} showing high {metric_col} spread over past half-year",
                "{verb} {group_col} with notable {metric_col} divergence in recent data",
                "{verb} {group_col} demonstrating substantial {metric_col} variability recently"
            ],
            'explanation': "First CTE filters last 6 months of data, second CTE calculates average, minimum, and maximum {metric_col} per {group_col}, final query shows groups where average is 50% higher than minimum",
            'tables_needed': ['{table1}'],
            'requires': ['grouping_col', 'metric_col', 'date_col'],
            'verb_style': 'analyze'
        },
        
        # ============ TEMPLATE 5 ============
        {
            'sql': """WITH monthly_totals AS (
    SELECT {group_col}, DATE_FORMAT({date_col}, '%Y-%m') as month, SUM({metric_col}) as monthly_total
    FROM {table1}
    WHERE {date_col} >= '{year}-01-01'
    GROUP BY {group_col}, month
),
ranked_months AS (
    SELECT *, RANK() OVER (PARTITION BY {group_col} ORDER BY monthly_total DESC) as month_rank
    FROM monthly_totals
)
SELECT {group_col}, month, monthly_total
FROM ranked_months
WHERE month_rank = 1;""",
            'prompt_templates': [
                "{verb} the best performing month for each {group_col} in {year}",
                "{verb} peak month by {metric_col} for every {group_col} during {year}",
                "{verb} highest revenue month per {group_col} in {year}",
                "{verb} top monthly performance for each {group_col} in {year}",
                "{verb} strongest month by {metric_col} for all {group_col} in {year}"
            ],
            'explanation': "First CTE aggregates {metric_col} by month and {group_col} for {year}, second CTE ranks months within each group, final query returns only the highest-performing month per group",
            'tables_needed': ['{table1}'],
            'requires': ['grouping_col', 'metric_col', 'date_col'],
            'verb_style': 'identify'
        },
        
        # ============ TEMPLATE 6 ============
        {
            'sql': """WITH base_metrics AS (
    SELECT {group_col}, AVG({metric_col}) as avg_{metric_col}, STDDEV({metric_col}) as stddev_{metric_col}
    FROM {table1}
    WHERE {date_col} >= '{year}-01-01'
    GROUP BY {group_col}
),
outlier_detection AS (
    SELECT b.*, t.{metric_col},
        CASE WHEN t.{metric_col} > b.avg_{metric_col} + 2 * b.stddev_{metric_col} THEN 'High Outlier'
                WHEN t.{metric_col} < b.avg_{metric_col} - 2 * b.stddev_{metric_col} THEN 'Low Outlier'
                ELSE 'Normal' END as outlier_status
    FROM base_metrics b
    JOIN {table1} t ON b.{group_col} = t.{group_col}
    WHERE t.{date_col} >= '{year}-01-01'
)
SELECT {group_col}, COUNT(*) as outlier_count
FROM outlier_detection
WHERE outlier_status != 'Normal'
GROUP BY {group_col};""",
            'prompt_templates': [
                "{verb} {group_col} with outlier counts in {metric_col} for {year}",
                "{verb} statistical anomalies by {group_col} during {year}",
                "{verb} {group_col} showing unusual {metric_col} patterns in {year}",
                "{verb} outlier frequency per {group_col} for {year}",
                "{verb} abnormal {metric_col} occurrences by {group_col} in {year}"
            ],
            'explanation': "First CTE calculates mean and standard deviation of {metric_col} by {group_col}, second CTE identifies records more than 2 standard deviations from mean, final query counts outliers per group",
            'tables_needed': ['{table1}'],
            'requires': ['grouping_col', 'metric_col', 'date_col'],
            'verb_style': 'detect'
        },
        
        # ============ TEMPLATE 7 ============
        {
            'sql': """WITH period_comparison AS (
    SELECT {group_col},
        SUM(CASE WHEN {date_col} >= '{year}-01-01' AND {date_col} < '{year}-07-01' THEN {metric_col} ELSE 0 END) as first_half,
        SUM(CASE WHEN {date_col} >= '{year}-07-01' AND {date_col} <= '{year}-12-31' THEN {metric_col} ELSE 0 END) as second_half
    FROM {table1}
    WHERE {date_col} BETWEEN '{year}-01-01' AND '{year}-12-31'
    GROUP BY {group_col}
)
SELECT {group_col}, first_half, second_half, 
    (second_half - first_half) as difference,
    ROUND(((second_half - first_half) / NULLIF(first_half, 0)) * 100, 2) as pct_change
FROM period_comparison
WHERE first_half > 0
ORDER BY pct_change DESC;""",
            'prompt_templates': [
                "{verb} year-over-year comparison of {metric_col} by {group_col} for {year}",
                "{verb} first vs second half {metric_col} growth by {group_col} in {year}",
                "{verb} period-over-period {metric_col} changes for {group_col} during {year}",
                "{verb} half-year {metric_col} comparison across {group_col} in {year}",
                "{verb} {group_col} performance comparing H1 vs H2 of {year}"
            ],
            'explanation': "CTE calculates {metric_col} totals for first and second half of {year} by {group_col}, main query computes absolute and percentage changes between periods, orders by growth rate",
            'tables_needed': ['{table1}'],
            'requires': ['grouping_col', 'metric_col', 'date_col'],
            'verb_style': 'compare'
        },
        
        # ============ TEMPLATE 8 ============
        {
            'sql': """WITH year_totals AS (
    SELECT YEAR({date_col}) as year, {group_col}, SUM({metric_col}) as yearly_total
    FROM {table1}
    GROUP BY year, {group_col}
),
growth_calc AS (
    SELECT year, {group_col}, yearly_total,
        LAG(yearly_total) OVER (PARTITION BY {group_col} ORDER BY year) as prev_year_total,
        yearly_total - LAG(yearly_total) OVER (PARTITION BY {group_col} ORDER BY year) as growth
    FROM year_totals
)
SELECT * FROM growth_calc
WHERE year >= {year} AND growth IS NOT NULL
ORDER BY growth DESC;""",
            'prompt_templates': [
                "{verb} year-over-year growth in {metric_col} by {group_col} since {year}",
                "{verb} annual {metric_col} increases for each {group_col} from {year}",
                "{verb} {group_col} showing yearly {metric_col} growth starting {year}",
                "{verb} YoY {metric_col} changes by {group_col} for {year} onwards",
                "{verb} annual growth trends in {metric_col} per {group_col}"
            ],
            'explanation': "First CTE aggregates {metric_col} by year and {group_col}, second CTE calculates year-over-year growth using LAG, final query shows growth trends from {year}",
            'tables_needed': ['{table1}'],
            'requires': ['grouping_col', 'metric_col', 'date_col'],
            'verb_style': 'calculate'
        },
        
        # ============ TEMPLATE 9 ============
        {
            'sql': """WITH category_metrics AS (
    SELECT {group_col}, 
        COUNT(*) as record_count,
        AVG({metric_col}) as avg_metric,
        STDDEV({metric_col}) as std_metric
    FROM {table1}
    WHERE {date_col} >= '{year}-01-01'
    GROUP BY {group_col}
),
significant_categories AS (
    SELECT * FROM category_metrics
    WHERE record_count >= 5 AND std_metric > avg_metric * 0.2
)
SELECT * FROM significant_categories
ORDER BY std_metric DESC;""",
            'prompt_templates': [
                "{verb} {group_col} with high variability in {metric_col} for {year}",
                "{verb} categories showing significant {metric_col} variation since {year}",
                "{verb} {group_col} with notable {metric_col} spread in {year}",
                "{verb} volatile {group_col} based on {metric_col} statistics",
                "{verb} {group_col} demonstrating high {metric_col} standard deviation"
            ],
            'explanation': "First CTE calculates statistical metrics per {group_col}, second CTE filters for categories with sufficient data and high variability (std dev > 20% of mean), for records since {year}",
            'tables_needed': ['{table1}'],
            'requires': ['grouping_col', 'metric_col', 'date_col'],
            'verb_style': 'identify'
        },
        
        # ============ TEMPLATE 10 ============
        {
            'sql': """WITH quarterly_data AS (
    SELECT {group_col}, 
        CONCAT(YEAR({date_col}), '-Q', QUARTER({date_col})) as quarter,
        SUM({metric_col}) as quarterly_total,
        COUNT(*) as transaction_count
    FROM {table1}
    WHERE YEAR({date_col}) = {year}
    GROUP BY {group_col}, quarter
),
ranked_quarters AS (
    SELECT *, 
        ROW_NUMBER() OVER (PARTITION BY {group_col} ORDER BY quarterly_total DESC) as rank
    FROM quarterly_data
)
SELECT {group_col}, quarter, quarterly_total, transaction_count
FROM ranked_quarters
WHERE rank <= 2;""",
            'prompt_templates': [
                "{verb} top 2 quarters by {metric_col} for each {group_col} in {year}",
                "{verb} best performing quarters per {group_col} during {year}",
                "{verb} highest {metric_col} quarters for all {group_col} in {year}",
                "{verb} peak quarterly performance by {group_col} for {year}",
                "{verb} leading quarters in {metric_col} per {group_col}"
            ],
            'explanation': "First CTE aggregates {metric_col} by quarter and {group_col} for {year}, second CTE ranks quarters within each group, final query returns top 2 quarters per group",
            'tables_needed': ['{table1}'],
            'requires': ['grouping_col', 'metric_col', 'date_col'],
            'verb_style': 'show'
        },
        
        # ============ TEMPLATE 11 ============
        {
            'sql': """WITH cohort_analysis AS (
    SELECT {group_col},
        DATE_FORMAT({date_col}, '%Y-%m') as cohort_month,
        COUNT(*) as cohort_size,
        AVG({metric_col}) as cohort_avg
    FROM {table1}
    WHERE {date_col} >= '{year}-01-01'
    GROUP BY {group_col}, cohort_month
),
cohort_comparison AS (
    SELECT cohort_month,
        AVG(cohort_avg) as overall_avg,
        MAX(cohort_avg) as max_avg,
        MIN(cohort_avg) as min_avg
    FROM cohort_analysis
    GROUP BY cohort_month
)
SELECT * FROM cohort_comparison
ORDER BY cohort_month;""",
            'prompt_templates': [
                "{verb} monthly cohort analysis of {metric_col} since {year}",
                "{verb} cohort performance metrics by month for {year}",
                "{verb} time-based cohort comparison of {metric_col}",
                "{verb} monthly {metric_col} cohort statistics from {year}",
                "{verb} cohort-level {metric_col} trends by month"
            ],
            'explanation': "First CTE creates monthly cohorts with average {metric_col}, second CTE calculates cross-cohort statistics per month, showing overall trends from {year}",
            'tables_needed': ['{table1}'],
            'requires': ['grouping_col', 'metric_col', 'date_col'],
            'verb_style': 'analyze'
        },
        
        # ============ TEMPLATE 12 ============
        {
            'sql': """WITH cumulative_metrics AS (
    SELECT {group_col}, {date_col}, {metric_col},
        SUM({metric_col}) OVER (PARTITION BY {group_col} ORDER BY {date_col}) as running_sum
    FROM {table1}
    WHERE {date_col} >= '{year}-01-01'
),
milestone_check AS (
    SELECT {group_col}, MIN({date_col}) as milestone_date
    FROM cumulative_metrics
    WHERE running_sum >= {threshold}
    GROUP BY {group_col}
)
SELECT * FROM milestone_check
ORDER BY milestone_date;""",
            'prompt_templates': [
                "{verb} when each {group_col} reached {threshold} cumulative {metric_col}",
                "{verb} milestone dates for {group_col} hitting {threshold} in {metric_col}",
                "{verb} first occurrence of {group_col} surpassing {threshold} cumulative",
                "{verb} {group_col} achievement dates for {threshold} {metric_col} milestone",
                "{verb} date each {group_col} crossed {threshold} cumulative {metric_col}"
            ],
            'explanation': "First CTE calculates running sum of {metric_col} per {group_col}, second CTE identifies earliest date each group reached {threshold} milestone, for records since {year}",
            'tables_needed': ['{table1}'],
            'requires': ['grouping_col', 'metric_col', 'date_col'],
            'verb_style': 'find'
        },
        
        # ============ TEMPLATE 13 ============
        {
            'sql': """WITH daily_metrics AS (
    SELECT {group_col}, DATE({date_col}) as date, SUM({metric_col}) as daily_total
    FROM {table1}
    WHERE {date_col} >= '{year}-01-01'
    GROUP BY {group_col}, date
),
volatility_calc AS (
    SELECT {group_col},
        AVG(daily_total) as avg_daily,
        STDDEV(daily_total) as std_daily,
        MAX(daily_total) - MIN(daily_total) as range_daily
    FROM daily_metrics
    GROUP BY {group_col}
)
SELECT *, 
    CASE 
        WHEN std_daily / NULLIF(avg_daily, 0) > 0.5 THEN 'High Volatility'
        WHEN std_daily / NULLIF(avg_daily, 0) > 0.2 THEN 'Medium Volatility'
        ELSE 'Low Volatility'
    END as volatility_category
FROM volatility_calc
ORDER BY std_daily / NULLIF(avg_daily, 0) DESC;""",
            'prompt_templates': [
                "{verb} {group_col} volatility patterns in {metric_col} for {year}",
                "{verb} stability classification of {group_col} by {metric_col}",
                "{verb} {group_col} grouped by {metric_col} variability levels",
                "{verb} volatility analysis of {group_col} performance",
                "{verb} {group_col} categorized by {metric_col} consistency"
            ],
            'explanation': "First CTE aggregates daily {metric_col} per {group_col}, second CTE calculates volatility metrics, final query classifies groups by coefficient of variation into volatility categories",
            'tables_needed': ['{table1}'],
            'requires': ['grouping_col', 'metric_col', 'date_col'],
            'verb_style': 'classify'
        },
        
        # ============ TEMPLATE 14 ============
        {
            'sql': """WITH top_performers AS (
    SELECT {group_col}, SUM({metric_col}) as total
    FROM {table1}
    WHERE {date_col} >= '{year}-01-01'
    GROUP BY {group_col}
    ORDER BY total DESC
    LIMIT {top_n}
),
bottom_performers AS (
    SELECT {group_col}, SUM({metric_col}) as total
    FROM {table1}
    WHERE {date_col} >= '{year}-01-01'
    GROUP BY {group_col}
    ORDER BY total ASC
    LIMIT {top_n}
)
SELECT 'Top Performers' as category, * FROM top_performers
UNION ALL
SELECT 'Bottom Performers' as category, * FROM bottom_performers
ORDER BY category DESC, total DESC;""",
            'prompt_templates': [
                "{verb} top and bottom {top_n} {group_col} by {metric_col} in {year}",
                "{verb} best and worst performing {group_col} for {year}",
                "{verb} extreme performers in {group_col} based on {metric_col}",
                "{verb} highest and lowest {top_n} {group_col} by {metric_col}",
                "{verb} performance extremes across {group_col} for {year}"
            ],
            'explanation': "Two CTEs identify top and bottom {top_n} performers by {metric_col}, union combines them with category labels, providing comprehensive view of performance distribution from {year}",
            'tables_needed': ['{table1}'],
            'requires': ['grouping_col', 'metric_col', 'date_col'],
            'verb_style': 'compare'
        },
        
        # ============ TEMPLATE 15 ============
        {
            'sql': """WITH weekly_aggregates AS (
    SELECT {group_col},
        YEARWEEK({date_col}) as year_week,
        SUM({metric_col}) as weekly_total,
        AVG({metric_col}) as weekly_avg
    FROM {table1}
    WHERE {date_col} >= '{year}-01-01'
    GROUP BY {group_col}, year_week
),
trend_analysis AS (
    SELECT {group_col},
        CORR(year_week, weekly_total) as trend_correlation,
        COUNT(*) as week_count
    FROM weekly_aggregates
    GROUP BY {group_col}
    HAVING COUNT(*) >= 10
)
SELECT {group_col}, trend_correlation,
    CASE 
        WHEN trend_correlation > 0.5 THEN 'Strong Upward'
        WHEN trend_correlation > 0 THEN 'Weak Upward'
        WHEN trend_correlation > -0.5 THEN 'Weak Downward'
        ELSE 'Strong Downward'
    END as trend_direction
FROM trend_analysis
ORDER BY ABS(trend_correlation) DESC;""",
            'prompt_templates': [
                "{verb} weekly trend patterns for {group_col} in {metric_col}",
                "{verb} {group_col} with strongest {metric_col} trends since {year}",
                "{verb} directional analysis of {group_col} by {metric_col}",
                "{verb} trend strength classification for {group_col}",
                "{verb} {group_col} showing clear {metric_col} trends"
            ],
            'explanation': "First CTE aggregates weekly {metric_col} per {group_col}, second CTE calculates correlation coefficient to determine trend strength/direction, requires minimum 10 weeks of data from {year}",
            'tables_needed': ['{table1}'],
            'requires': ['grouping_col', 'metric_col', 'date_col'],
            'verb_style': 'analyze'
        }
    ]