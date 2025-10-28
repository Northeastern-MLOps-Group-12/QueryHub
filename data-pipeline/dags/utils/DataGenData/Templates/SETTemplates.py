from typing import List , Dict

def _get_set_operation_templates() -> List[Dict]:
    """15 diverse set operation templates"""
    return [
        # ============ TEMPLATE 1 ============
        {
            'sql': """SELECT {group_col} as value FROM {table1} WHERE {group_col} IS NOT NULL
UNION
SELECT {group_col} as value FROM {table2} WHERE {group_col} IS NOT NULL;""",
            'prompt_templates': [
                "{verb} all unique values from {group_col} in {table1} and {table2}",
                "{verb} combined distinct values of {group_col} from both tables",
                "{verb} union of {group_col} from {table1} with {table2}",
                "{verb} merged unique values from {table1}.{group_col} and {table2}.{group_col}",
                "{verb} consolidated list of {group_col} across tables"
            ],
            'explanation': "Combines unique values from {group_col} in {table1} and {group_col} in {table2} using UNION, which automatically removes duplicates",
            'tables_needed': ['{table1}', '{table2}'],
            'requires': ['grouping_col'],
            'verb_style': 'find'
        },
        
        # ============ TEMPLATE 2 ============
        {
            'sql': """SELECT {group_col} FROM {table1}
INTERSECT
SELECT {group_col} FROM {table2};""",
            'prompt_templates': [
                "{verb} common {group_col} values appearing in both {table1} and {table2}",
                "{verb} {group_col} that exist in both {table1} and {table2}",
                "{verb} overlapping {group_col} between {table1} and {table2}",
                "{verb} shared {group_col} present in {table1} and {table2}",
                "{verb} {group_col} intersection across {table1} and {table2}"
            ],
            'explanation': "Returns only {group_col} values that are present in both {table1} and {table2} tables using INTERSECT",
            'tables_needed': ['{table1}', '{table2}'],
            'requires': ['grouping_col'],
            'verb_style': 'find'
        },
        
        # ============ TEMPLATE 3 ============
        {
            'sql': """SELECT {group_col} FROM {table1}
EXCEPT
SELECT {group_col} FROM {table2};""",
            'prompt_templates': [
                "{verb} {group_col} values in {table1} but not in {table2}",
                "{verb} {group_col} that appear in {table1} but are missing from {table2}",
                "{verb} {group_col} exclusive to {table1}",
                "{verb} {group_col} unique to {table1} and absent from {table2}",
                "{verb} {group_col} difference between {table1} and {table2}"
            ],
            'explanation': "Returns {group_col} values that exist in {table1} but not in {table2} using EXCEPT set operation",
            'tables_needed': ['{table1}', '{table2}'],
            'requires': ['grouping_col'],
            'verb_style': 'find'
        },
        
        # ============ TEMPLATE 4 ============
        {
            'sql': """SELECT {group_col}, COUNT(*) as count FROM {table1}
GROUP BY {group_col}
UNION ALL
SELECT {group_col}, COUNT(*) as count FROM {table2}
GROUP BY {group_col}
ORDER BY count DESC;""",
            'prompt_templates': [
                "{verb} counts of {group_col} from both {table1} and {table2} combined",
                "{verb} aggregated {group_col} frequencies across {table1} and {table2}",
                "{verb} combined frequency distribution of {group_col} from both tables",
                "{verb} merged {group_col} counts preserving duplicates from {table1} and {table2}",
                "{verb} total {group_col} occurrences across {table1} and {table2}"
            ],
            'explanation': "Uses UNION ALL to combine counts of {group_col} from both tables, preserving all records including duplicates, then sorts by frequency",
            'tables_needed': ['{table1}', '{table2}'],
            'requires': ['grouping_col'],
            'verb_style': 'show'
        },
        
        # ============ TEMPLATE 5 ============
        {
            'sql': """(SELECT {group_col}, {metric_col} FROM {table1} WHERE {date_col} >= '{year}-01-01' ORDER BY {metric_col} DESC LIMIT {top_n})
UNION ALL
(SELECT {group_col}, {metric_col} FROM {table2} WHERE {date_col} >= '{year}-01-01' ORDER BY {metric_col} DESC LIMIT {top_n})
ORDER BY {metric_col} DESC;""",
            'prompt_templates': [
                "{verb} top {top_n} records by {metric_col} from {table1} and {table2} for {year}",
                "{verb} highest {metric_col} entries combining {table1} and {table2} since {year}",
                "{verb} leading {top_n} from each table merged by {metric_col}",
                "{verb} combined top performers across {table1} and {table2} in {year}",
                "{verb} best {top_n} results from both tables based on {metric_col}"
            ],
            'explanation': "Retrieves top {top_n} records by {metric_col} from each table for records since {year}, combines them using UNION ALL, and orders the merged result by {metric_col}",
            'tables_needed': ['{table1}', '{table2}'],
            'requires': ['grouping_col', 'metric_col', 'date_col'],
            'verb_style': 'combine'
        },
        
        # ============ TEMPLATE 6 ============
        {
            'sql': """SELECT 'Only in {table1}' as source, COUNT(*) as count FROM (
    SELECT {group_col} FROM {table1}
    EXCEPT
    SELECT {group_col} FROM {table2}
) t1
UNION ALL
SELECT 'Only in {table2}' as source, COUNT(*) as count FROM (
    SELECT {group_col} FROM {table2}
    EXCEPT
    SELECT {group_col} FROM {table1}
) t2
UNION ALL
SELECT 'In both tables' as source, COUNT(*) as count FROM (
    SELECT {group_col} FROM {table1}
    INTERSECT
    SELECT {group_col} FROM {table2}
) t3;""",
            'prompt_templates': [
                "{verb} distribution of {group_col} across {table1} and {table2}",
                "{verb} {group_col} overlap analysis between {table1} and {table2}",
                "{verb} set membership counts for {group_col} in both tables",
                "{verb} breakdown of unique vs shared {group_col} values",
                "{verb} {group_col} presence statistics across tables"
            ],
            'explanation': "Performs comprehensive set analysis: counts {group_col} values exclusive to {table1}, exclusive to {table2}, and present in both, providing complete overlap picture",
            'tables_needed': ['{table1}', '{table2}'],
            'requires': ['grouping_col'],
            'verb_style': 'analyze'
        },
        
        # ============ TEMPLATE 7 ============
        {
            'sql': """SELECT {group_col}, 'Active in {table1}' as status FROM {table1}
WHERE {date_col} >= '{year}-01-01'
UNION
SELECT {group_col}, 'Active in {table2}' as status FROM {table2}
WHERE {date_col} >= '{year}-01-01'
ORDER BY {group_col}, status;""",
            'prompt_templates': [
                "{verb} activity status of {group_col} across both tables for {year}",
                "{verb} {group_col} presence in {table1} and {table2} since {year}",
                "{verb} combined activity records for {group_col} from {year}",
                "{verb} {group_col} engagement across tables in {year}",
                "{verb} unified {group_col} status from both sources"
            ],
            'explanation': "Combines {group_col} values with status indicators from both tables for records since {year}, using UNION to deduplicate identical entries",
            'tables_needed': ['{table1}', '{table2}'],
            'requires': ['grouping_col', 'date_col'],
            'verb_style': 'show'
        },
        
        # ============ TEMPLATE 8 ============
        {
            'sql': """(SELECT {group_col}, MAX({metric_col}) as max_value FROM {table1} GROUP BY {group_col})
UNION ALL
(SELECT {group_col}, MAX({metric_col}) as max_value FROM {table2} GROUP BY {group_col})
ORDER BY {group_col}, max_value DESC;""",
            'prompt_templates': [
                "{verb} maximum {metric_col} for {group_col} from both tables",
                "{verb} peak {metric_col} values per {group_col} across sources",
                "{verb} highest {metric_col} for each {group_col} in combined data",
                "{verb} {group_col} with their maximum {metric_col} from all tables",
                "{verb} merged maximum {metric_col} by {group_col}"
            ],
            'explanation': "Retrieves maximum {metric_col} for each {group_col} from both tables separately, combines using UNION ALL to preserve all maximums even for same {group_col}",
            'tables_needed': ['{table1}', '{table2}'],
            'requires': ['grouping_col', 'metric_col'],
            'verb_style': 'find'
        },
        
        # ============ TEMPLATE 9 ============
        {
            'sql': """SELECT {group_col} FROM {table1} WHERE {metric_col} > {threshold}
INTERSECT
SELECT {group_col} FROM {table2} WHERE {metric_col} > {threshold};""",
            'prompt_templates': [
                "{verb} {group_col} exceeding {threshold} in both {table1} and {table2}",
                "{verb} high-performing {group_col} common to both tables",
                "{verb} {group_col} above {threshold} threshold in all sources",
                "{verb} consistently strong {group_col} across tables",
                "{verb} {group_col} meeting {threshold} criteria in both datasets"
            ],
            'explanation': "Finds {group_col} values that exceed {threshold} in {metric_col} in both {table1} and {table2}, using INTERSECT to show only common high performers",
            'tables_needed': ['{table1}', '{table2}'],
            'requires': ['grouping_col', 'metric_col'],
            'verb_style': 'identify'
        },
        
        # ============ TEMPLATE 10 ============
        {
            'sql': """SELECT {group_col}, COUNT(*) as frequency FROM (
    SELECT {group_col} FROM {table1}
    UNION ALL
    SELECT {group_col} FROM {table2}
) combined
GROUP BY {group_col}
HAVING COUNT(*) = 1;""",
            'prompt_templates': [
                "{verb} {group_col} appearing in only one table",
                "{verb} unique {group_col} values exclusive to single source",
                "{verb} {group_col} with no overlap between tables",
                "{verb} non-duplicate {group_col} across both datasets",
                "{verb} {group_col} found in exactly one table"
            ],
            'explanation': "Combines all {group_col} from both tables with UNION ALL, counts occurrences, filters for those appearing exactly once (exclusive to one table)",
            'tables_needed': ['{table1}', '{table2}'],
            'requires': ['grouping_col'],
            'verb_style': 'find'
        },
        
        # ============ TEMPLATE 11 ============
        {
            'sql': """SELECT '{table1}' as source, {group_col}, SUM({metric_col}) as total FROM {table1}
WHERE {date_col} >= '{year}-01-01'
GROUP BY {group_col}
UNION ALL
SELECT '{table2}' as source, {group_col}, SUM({metric_col}) as total FROM {table2}
WHERE {date_col} >= '{year}-01-01'
GROUP BY {group_col}
ORDER BY {group_col}, source;""",
            'prompt_templates': [
                "{verb} {metric_col} totals by {group_col} and source table for {year}",
                "{verb} source-attributed {metric_col} aggregates per {group_col}",
                "{verb} combined {group_col} totals with source tracking",
                "{verb} {metric_col} breakdown by {group_col} and origin table",
                "{verb} merged {group_col} metrics preserving source information"
            ],
            'explanation': "Aggregates {metric_col} by {group_col} from each table with source labels, combines with UNION ALL to show all entries with their origin, filtered for {year}",
            'tables_needed': ['{table1}', '{table2}'],
            'requires': ['grouping_col', 'metric_col', 'date_col'],
            'verb_style': 'show'
        },
        
        # ============ TEMPLATE 12 ============
        {
            'sql': """SELECT {group_col} FROM {table1} WHERE {date_col} >= '{year}-01-01'
EXCEPT
SELECT {group_col} FROM {table2} WHERE {date_col} < '{year}-01-01';""",
            'prompt_templates': [
                "{verb} {group_col} new to {table1} since {year}",
                "{verb} {group_col} in recent {table1} but not historical {table2}",
                "{verb} newly appeared {group_col} starting {year}",
                "{verb} {group_col} absent from {table2} history but present in {table1}",
                "{verb} {group_col} emerging in {year} across datasets"
            ],
            'explanation': "Finds {group_col} present in {table1} since {year} but not in {table2} before {year}, identifying new entries using EXCEPT",
            'tables_needed': ['{table1}', '{table2}'],
            'requires': ['grouping_col', 'date_col'],
            'verb_style': 'identify'
        },
        
        # ============ TEMPLATE 13 ============
        {
            'sql': """SELECT {group_col}, AVG({metric_col}) as avg_metric FROM {table1}
GROUP BY {group_col}
HAVING AVG({metric_col}) > {threshold}
UNION
SELECT {group_col}, AVG({metric_col}) as avg_metric FROM {table2}
GROUP BY {group_col}
HAVING AVG({metric_col}) > {threshold}
ORDER BY avg_metric DESC;""",
            'prompt_templates': [
                "{verb} {group_col} with average {metric_col} above {threshold} in either table",
                "{verb} high-performing {group_col} by average {metric_col} across sources",
                "{verb} {group_col} exceeding {threshold} average in any dataset",
                "{verb} combined {group_col} meeting {threshold} average threshold",
                "{verb} merged {group_col} with strong average {metric_col}"
            ],
            'explanation': "Calculates average {metric_col} per {group_col} in each table, filters for averages above {threshold}, combines unique results with UNION",
            'tables_needed': ['{table1}', '{table2}'],
            'requires': ['grouping_col', 'metric_col'],
            'verb_style': 'find'
        },
        
        # ============ TEMPLATE 14 ============
        {
            'sql': """SELECT {group_col}, 'Common' as category, COUNT(*) as record_count FROM (
    SELECT {group_col} FROM {table1}
    INTERSECT
    SELECT {group_col} FROM {table2}
) common_values
GROUP BY {group_col}
UNION ALL
SELECT {group_col}, 'Unique to {table1}' as category, COUNT(*) as record_count FROM (
    SELECT {group_col} FROM {table1}
    EXCEPT
    SELECT {group_col} FROM {table2}
) table1_only
GROUP BY {group_col};""",
            'prompt_templates': [
                "{verb} {group_col} categorized by presence in tables",
                "{verb} {group_col} distribution across common and unique sets",
                "{verb} membership classification of {group_col} values",
                "{verb} {group_col} breakdown by table presence pattern",
                "{verb} categorized {group_col} showing overlap status"
            ],
            'explanation': "Categorizes {group_col} values as either common to both tables or unique to {table1}, providing counts for each category using combination of INTERSECT and EXCEPT",
            'tables_needed': ['{table1}', '{table2}'],
            'requires': ['grouping_col'],
            'verb_style': 'classify'
        },
        
        # ============ TEMPLATE 15 ============
        {
            'sql': """(SELECT DISTINCT {group_col}, {metric_col} FROM {table1} WHERE {metric_col} >= {threshold} AND {date_col} >= '{year}-01-01')
UNION
(SELECT DISTINCT {group_col}, {metric_col} FROM {table2} WHERE {metric_col} >= {threshold} AND {date_col} >= '{year}-01-01')
ORDER BY {metric_col} DESC
LIMIT {top_n};""",
            'prompt_templates': [
                "{verb} top {top_n} {group_col} by {metric_col} across tables since {year}",
                "{verb} highest {top_n} {group_col} meeting {threshold} from all sources",
                "{verb} leading {top_n} {group_col} with {metric_col} above {threshold}",
                "{verb} best {top_n} performers across combined datasets",
                "{verb} merged top {top_n} {group_col} by {metric_col} threshold"
            ],
            'explanation': "Retrieves distinct {group_col} with {metric_col} >= {threshold} from both tables since {year}, combines with UNION, returns top {top_n} by {metric_col}",
            'tables_needed': ['{table1}', '{table2}'],
            'requires': ['grouping_col', 'metric_col', 'date_col'],
            'verb_style': 'rank'
        }
    ]