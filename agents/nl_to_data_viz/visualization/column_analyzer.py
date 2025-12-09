import pandas as pd
import json
from typing import Dict, List, Set

class ColumnAnalyzer:
    """Analyze columns to detect original vs generated, IDs, etc."""
    
    def __init__(self):
        self.agg_functions = [
            "SUM", "AVG", "COUNT", "MAX", "MIN", 
            "TOTAL", "AVERAGE",
            "STRING_AGG", "GROUP_CONCAT", "LISTAGG", "ARRAY_AGG",
            "CONCAT_WS", "XMLAGG",
            "STDDEV", "VARIANCE", "MEDIAN"
        ]
        self.gen_keywords = ["total", "sum", "avg", "average", "count", "max", "min", "calculated", "computed"]
        self.id_patterns = {
            "exact": ["id", "ID", "index", "Index"],
            "contains": ["_id", "_ID", "id_", "ID_"],
            "ends_with": ["Id", "ID"]
        }
    
    def get_source_columns(self, table_details: List[Dict]) -> Set[str]:
        """Extract all column names from source tables"""
        source_cols = set()
        
        for table in table_details:
            try:
                columns_json = table.get('columns', '[]')
                if isinstance(columns_json, str):
                    columns = json.loads(columns_json)
                else:
                    columns = columns_json
                
                for col in columns:
                    source_cols.add(col.get('name', ''))
            except:
                continue
        
        return source_cols
    
    def detect_generated_columns(self, df: pd.DataFrame, sql_query: str, table_details: List[Dict]) -> Dict[str, List[str]]:
        """
        Detect which columns are generated (aggregated/calculated) vs original
        
        Returns:
            {
                'original': [col1, col2],
                'generated': [col3, col4]
            }
        """
        source_columns = self.get_source_columns(table_details)
        sql_upper = sql_query.upper()
        
        original = []
        generated = []
        
        for col in df.columns:
            # Method 1: Check if column exists in source tables
            if col in source_columns:
                original.append(col)
                continue
            
            # Method 2: Check SQL for aggregation functions
            has_aggregation = any(f"{func}(" in sql_upper for func in self.agg_functions)
            
            # Method 3: Check column name for generated keywords
            col_lower = col.lower()
            has_gen_keyword = any(keyword in col_lower for keyword in self.gen_keywords)
            
            # If either SQL has aggregation or column name suggests generation
            if has_aggregation or has_gen_keyword:
                generated.append(col)
            else:
                # Default: if not in source and no clear signals, consider it original
                original.append(col)
        

        return {
            'original': original,
            'generated': generated
        }
    
    def is_id_column(self, col_name: str) -> bool:
        """Check if column is an ID/index column"""
        col_lower = col_name.lower()
        
        # Exact match
        if col_name in self.id_patterns["exact"]:
            return True
        
        # Contains pattern
        if any(pattern.lower() in col_lower for pattern in self.id_patterns["contains"]):
            return True
        
        # Ends with pattern
        if any(col_name.endswith(pattern) for pattern in self.id_patterns["ends_with"]):
            return True
        
        return False
    
    def categorize_columns(self, df: pd.DataFrame) -> Dict:
        """Categorize columns by type for EDA"""
        categories = {
            'numeric': [],
            'categorical': [],
            'datetime': [],
            'ids': []
        }
        
        for col in df.columns:
            # Check if ID first
            if self.is_id_column(col):
                categories['ids'].append(col)
                continue
            
            dtype = df[col].dtype
            
            if pd.api.types.is_numeric_dtype(dtype):
                categories['numeric'].append(col)
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                categories['datetime'].append(col)
            else:
                categories['categorical'].append(col)
        
        return categories