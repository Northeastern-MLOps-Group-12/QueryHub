import re
from typing import List, Dict
from .state import AgentState


class SQLComplexityAnalyzer:
    """Analyzes SQL queries to determine their complexity level"""
    
    def __init__(self):
        self.complexity_priority = [
            'window_functions',
            'set_operations',
            'multiple_joins',
            'subqueries',
            'aggregations',
            'single_join',
            'simple'
        ]
        
        self.aggregate_functions = [
            'COUNT', 'SUM', 'AVG', 'MAX', 'MIN',
            'STRING_AGG', 'LISTAGG', 'GROUP_CONCAT',
            'STDDEV', 'VARIANCE', 'ARRAY_AGG'
        ]
        
        self.window_functions = [
            'ROW_NUMBER', 'RANK', 'DENSE_RANK', 'NTILE',
            'LAG', 'LEAD', 'FIRST_VALUE', 'LAST_VALUE',
            'PERCENT_RANK', 'CUME_DIST', 'PERCENTILE_CONT',
            'PERCENTILE_DISC'
        ]
        
        self.set_operations = ['UNION', 'INTERSECT', 'EXCEPT', 'MINUS']
    
    def analyze(self, sql: str) -> Dict:
        """
        Analyze a SQL query and return complexity information.
        
        Returns:
            Dictionary with primary_complexity, all_complexities, details, complexity_count
        """
        sql_upper = sql.upper()
        
        complexities = []
        details = {}
        
        # 1. Check for Window Functions
        window_detected = self._detect_window_functions(sql_upper)
        if window_detected['found']:
            complexities.append('window_functions')
            details['window_functions'] = window_detected['details']
        
        # 2. Check for Set Operations
        set_ops_detected = self._detect_set_operations(sql_upper)
        if set_ops_detected['found']:
            complexities.append('set_operations')
            details['set_operations'] = set_ops_detected['details']
        
        # 3. Check for Joins
        joins_detected = self._detect_joins(sql_upper)
        if joins_detected['count'] >= 2:
            complexities.append('multiple_joins')
            details['multiple_joins'] = joins_detected['details']
        elif joins_detected['count'] == 1:
            complexities.append('single_join')
            details['single_join'] = joins_detected['details']
        
        # 4. Check for Subqueries
        subquery_detected = self._detect_subqueries(sql_upper)
        if subquery_detected['found']:
            complexities.append('subqueries')
            details['subqueries'] = subquery_detected['details']
        
        # 5. Check for Aggregations
        agg_detected = self._detect_aggregations(sql_upper)
        if agg_detected['found']:
            complexities.append('aggregations')
            details['aggregations'] = agg_detected['details']
        
        # 6. Default to simple if no complexity found
        if not complexities or complexities == ['single_join']:
            if not complexities:
                complexities.append('simple')
                details['simple'] = {'type': 'basic SELECT query'}
        
        # Determine primary complexity (highest priority)
        primary_complexity = self._determine_primary_complexity(complexities)
        
        return {
            'primary_complexity': primary_complexity,
            'all_complexities': complexities,
            'details': details,
            'complexity_count': len(complexities)
        }
    
    def _detect_window_functions(self, sql: str) -> Dict:
        """Detect window functions like ROW_NUMBER() OVER (...)"""
        found_functions = []
        
        for func in self.window_functions:
            pattern = rf'\b{func}\s*\([^)]*\)\s+OVER\s*\('
            if re.search(pattern, sql):
                found_functions.append(func)
        
        return {
            'found': len(found_functions) > 0,
            'details': {
                'functions': found_functions,
                'count': len(found_functions)
            }
        }
    
    def _detect_set_operations(self, sql: str) -> Dict:
        """Detect set operations like UNION, INTERSECT, EXCEPT"""
        found_operations = []
        
        for op in self.set_operations:
            pattern = rf'\b{op}\b'
            if re.search(pattern, sql):
                found_operations.append(op)
        
        return {
            'found': len(found_operations) > 0,
            'details': {
                'operations': found_operations,
                'count': len(found_operations)
            }
        }
    
    def _detect_joins(self, sql: str) -> Dict:
        """Detect JOIN operations and count them"""
        join_types = []
        
        join_patterns = [
            r'\bINNER\s+JOIN\b',
            r'\bLEFT\s+(?:OUTER\s+)?JOIN\b',
            r'\bRIGHT\s+(?:OUTER\s+)?JOIN\b',
            r'\bFULL\s+(?:OUTER\s+)?JOIN\b',
            r'\bCROSS\s+JOIN\b',
            r'\bJOIN\b'
        ]
        
        join_count = 0
        for pattern in join_patterns:
            matches = re.findall(pattern, sql)
            join_count += len(matches)
            if matches:
                join_types.extend(matches)
        
        return {
            'count': join_count,
            'found': join_count > 0,
            'details': {
                'join_count': join_count,
                'join_types': list(set(join_types))
            }
        }
    
    def _detect_subqueries(self, sql: str) -> Dict:
        """Detect subqueries (nested SELECT statements)"""
        select_count = len(re.findall(r'\bSELECT\b', sql))
        has_subqueries = select_count > 1
        
        subquery_types = []
        
        if re.search(r'\bFROM\s*\(\s*SELECT\b', sql):
            subquery_types.append('FROM clause subquery')
        
        if re.search(r'\bWHERE\s+.*\bIN\s*\(\s*SELECT\b', sql):
            subquery_types.append('WHERE IN subquery')
        
        if re.search(r'\bEXISTS\s*\(\s*SELECT\b', sql):
            subquery_types.append('EXISTS subquery')
        
        if re.search(r'\bSELECT\s+.*\(\s*SELECT\b', sql):
            subquery_types.append('SELECT clause subquery')
        
        return {
            'found': has_subqueries,
            'details': {
                'select_count': select_count,
                'subquery_types': subquery_types,
                'nesting_level': select_count - 1 if has_subqueries else 0
            }
        }
    
    def _detect_aggregations(self, sql: str) -> Dict:
        """Detect aggregation functions and GROUP BY"""
        found_functions = []
        
        for func in self.aggregate_functions:
            pattern = rf'\b{func}\s*\('
            if re.search(pattern, sql):
                found_functions.append(func)
        
        has_group_by = bool(re.search(r'\bGROUP\s+BY\b', sql))
        has_having = bool(re.search(r'\bHAVING\b', sql))
        
        return {
            'found': len(found_functions) > 0 or has_group_by,
            'details': {
                'aggregate_functions': found_functions,
                'has_group_by': has_group_by,
                'has_having': has_having,
                'function_count': len(found_functions)
            }
        }
    
    def _determine_primary_complexity(self, complexities: List[str]) -> str:
        """Determine the primary (most complex) complexity type"""
        for complexity in self.complexity_priority:
            if complexity in complexities:
                return complexity
        return 'simple'
    
    def get_complexity_score(self, primary_complexity: str) -> int:
        """Get a numeric score for complexity (1-7, higher = more complex)"""
        complexity_scores = {
            'window_functions': 7,
            'set_operations': 6,
            'multiple_joins': 5,
            'subqueries': 4,
            'aggregations': 3,
            'single_join': 2,
            'simple': 1
        }
        return complexity_scores.get(primary_complexity, 1)


# Global analyzer instance
_analyzer = SQLComplexityAnalyzer()


def analyze_sql_complexity(state: AgentState) -> Dict:
    """
    Workflow node to analyze SQL complexity.
    Called after SQL generation but before execution.
    Adds complexity information to state AND tracks metrics in Prometheus.
    """
    print("_____________________analyze_sql_complexity______________________")
    
    sql = state.generated_sql
    db_type = state.db_config.get('db_type', 'unknown')
    
    if not sql or sql.strip() == "":
        return {
            "sql_complexity": {
                'primary_complexity': 'unknown',
                'all_complexities': [],
                'details': {},
                'complexity_count': 0,
                'complexity_score': 0
            }
        }
    
    try:
        # Analyze the SQL
        analysis = _analyzer.analyze(sql)
        
        # Add complexity score
        analysis['complexity_score'] = _analyzer.get_complexity_score(
            analysis['primary_complexity']
        )
        
        # Track metrics in Prometheus
        from backend.monitoring import track_sql_complexity
        track_sql_complexity(analysis, db_type)
        
        # Print analysis results
        print(f"\n{'='*70}")
        print(f"📊 SQL Complexity Analysis")
        print(f"{'='*70}")
        print(f"Primary Complexity: {analysis['primary_complexity']}")
        print(f"Complexity Score: {analysis['complexity_score']}/7")
        print(f"All Complexities: {', '.join(analysis['all_complexities'])}")
        print(f"Details:")
        for complexity_type, details in analysis['details'].items():
            print(f"  - {complexity_type}: {details}")
        print(f"{'='*70}\n")
        
        return {
            "sql_complexity": analysis
        }
        
    except Exception as e:
        print(f"⚠️ Error analyzing SQL complexity: {e}")
        
        from backend.monitoring import track_workflow_error
        track_workflow_error('sql_complexity_analysis', 'analysis_error')
        
        return {
            "sql_complexity": {
                'primary_complexity': 'error',
                'all_complexities': [],
                'details': {'error': str(e)},
                'complexity_count': 0,
                'complexity_score': 0
            }
        }


def get_complexity_description(primary_complexity: str) -> str:
    """Get a human-readable description of the complexity type"""
    descriptions = {
        'window_functions': 'Advanced analytical query with window functions',
        'set_operations': 'Query combining multiple result sets',
        'multiple_joins': 'Query joining multiple tables (2+ joins)',
        'subqueries': 'Query with nested SELECT statements',
        'aggregations': 'Query with aggregate functions and/or GROUP BY',
        'single_join': 'Query joining two tables',
        'simple': 'Basic SELECT query',
        'unknown': 'Unable to analyze complexity',
        'error': 'Error during complexity analysis'
    }
    return descriptions.get(primary_complexity, 'Unknown complexity type')