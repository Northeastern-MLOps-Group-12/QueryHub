# ============================================================================
# FILE: AgentFiles/Visualization/eda_visualizer.py (FIXED)
# ============================================================================

from typing import List, Dict
import pandas as pd
import numpy as np
from scipy import stats
from .column_analyzer import ColumnAnalyzer
from ..utils.config_loader import VisualizationConfig


class EDAVisualizer:
    """Generate comprehensive EDA visualizations"""
    
    def __init__(self, analyzer: ColumnAnalyzer):
        self.analyzer = analyzer
        self.config = VisualizationConfig()
    
    def generate(self, df: pd.DataFrame) -> List[Dict]:
        """Generate comprehensive EDA visualizations including text stats"""
        categories = self.analyzer.categorize_columns(df)
        
        numeric_cols = [c for c in categories['numeric'] if c not in categories['ids']]
        categorical_cols = [c for c in categories['categorical'] if c not in categories['ids']]
        datetime_cols = categories['datetime']
        
        visualizations = []
        
        # 1. DataFrame Overview (df.info)
        visualizations.append(self._dataframe_overview(df))
        
        # 2. Statistical Summary (df.describe) - only if we have numeric columns
        if numeric_cols:
            stat_summary = self._statistical_summary(df, numeric_cols)
            if stat_summary:
                visualizations.append(stat_summary)
        
        # 3. Missing Data Analysis
        visualizations.extend(self._missing_data_analysis(df))
        
        # 4. Univariate Analysis
        visualizations.extend(self._univariate_analysis(df, numeric_cols, categorical_cols))
        
        # 5. Text Statistics for Excluded High-Cardinality Columns
        visualizations.extend(self._text_stats_for_excluded(df, categorical_cols))
        
        # 6. Bivariate Analysis
        visualizations.extend(self._bivariate_analysis(df, numeric_cols, categorical_cols, datetime_cols))
        
        # 7. Multivariate Analysis
        visualizations.extend(self._multivariate_analysis(df, numeric_cols, categorical_cols))
        
        max_total = self.config.get('eda_limits', 'max_total_graphs')
        return visualizations[:max_total]
    
    def _dataframe_overview(self, df: pd.DataFrame) -> Dict:
        """Generate DataFrame overview (df.info equivalent)"""
        
        column_info = []
        memory_usage = 0
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null = int(df[col].notna().sum())
            null_count = int(df[col].isna().sum())
            memory = df[col].memory_usage(deep=True)
            memory_usage += memory
            
            column_info.append({
                'column': col,
                'dtype': dtype,
                'non_null': non_null,
                'null_count': null_count,
                'memory_kb': float(memory / 1024)
            })
        
        return {
            'type': 'text_stats',
            'subtype': 'dataframe_info',
            'title': 'DataFrame Overview',
            'stats': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'total_memory_mb': float(memory_usage / (1024 * 1024)),
                'columns': column_info
            },
            'columns': list(df.columns)
        }
    
    def _statistical_summary(self, df: pd.DataFrame, numeric_cols: List[str]) -> Dict:
        """Generate statistical summary (df.describe)"""
        
        if not numeric_cols:
            return None
        
        try:
            # Only describe numeric columns
            describe_df = df[numeric_cols].describe()
            
            # Check if describe worked properly
            if 'mean' not in describe_df.index:
                return None
            
            stats_data = []
            for col in numeric_cols:
                try:
                    col_stats = {
                        'column': col,
                        'count': int(describe_df.loc['count', col]),
                        'mean': float(describe_df.loc['mean', col]),
                        'std': float(describe_df.loc['std', col]),
                        'min': float(describe_df.loc['min', col]),
                        'q25': float(describe_df.loc['25%', col]),
                        'median': float(describe_df.loc['50%', col]),
                        'q75': float(describe_df.loc['75%', col]),
                        'max': float(describe_df.loc['max', col])
                    }
                    stats_data.append(col_stats)
                except (KeyError, ValueError):
                    continue
            
            if not stats_data:
                return None
            
            return {
                'type': 'text_stats',
                'subtype': 'statistical_summary',
                'title': 'Statistical Summary (Numeric Columns)',
                'stats': stats_data,
                'columns': numeric_cols
            }
        except Exception as e:
            print(f"Failed to generate statistical summary: {e}")
            return None
    
    def _missing_data_analysis(self, df: pd.DataFrame) -> List[Dict]:
        """Generate missing data visualizations"""
        vizs = []
        
        missing_counts = df.isnull().sum()
        total_rows = len(df)
        
        columns_with_missing = []
        for col in df.columns:
            missing_count = missing_counts[col]
            if missing_count > 0:
                missing_pct = (missing_count / total_rows) * 100
                min_missing_pct = self.config.get('missing_data', 'min_missing_percentage')
                
                if missing_pct >= min_missing_pct:
                    columns_with_missing.append({
                        'column': col,
                        'missing_count': int(missing_count),
                        'missing_percentage': float(missing_pct)
                    })
        
        if columns_with_missing:
            vizs.append({
                'type': 'text_stats',
                'subtype': 'missing_data',
                'title': 'Missing Data Analysis',
                'stats': columns_with_missing,
                'total_rows': total_rows,
                'columns': [item['column'] for item in columns_with_missing]
            })
        
        return vizs
    
    def _text_stats_for_excluded(self, df: pd.DataFrame, categorical_cols: List[str]) -> List[Dict]:
        """Generate text-based stats for high-cardinality excluded columns"""
        vizs = []
        
        high_cardinality_threshold = self.config.get('categorical', 'high_cardinality_threshold')
        
        for col in categorical_cols:
            n_unique = df[col].nunique()
            
            if n_unique > high_cardinality_threshold:
                value_counts = df[col].value_counts()
                
                stats_data = {
                    'column_name': col,
                    'total_unique': int(n_unique),
                    'total_rows': len(df),
                    'top_5': [
                        {'value': str(val), 'count': int(count), 'percentage': float((count/len(df))*100)}
                        for val, count in value_counts.head(5).items()
                    ],
                    'bottom_5': [
                        {'value': str(val), 'count': int(count), 'percentage': float((count/len(df))*100)}
                        for val, count in value_counts.tail(5).items()
                    ],
                    'singleton_count': int(sum(value_counts == 1)),
                    'most_frequent': str(value_counts.index[0]),
                    'most_frequent_count': int(value_counts.iloc[0])
                }
                
                vizs.append({
                    'type': 'text_stats',
                    'subtype': 'high_cardinality',
                    'title': f'Statistical Summary: {col.replace("_", " ").title()}',
                    'stats': stats_data,
                    'columns': [col]
                })
        
        return vizs
    
    def _univariate_analysis(self, df: pd.DataFrame, numeric_cols: List[str], 
                            categorical_cols: List[str]) -> List[Dict]:
        """Generate univariate visualizations"""
        vizs = []
        
        max_numeric = self.config.get('eda_limits', 'max_numeric_columns')
        max_categorical = self.config.get('eda_limits', 'max_categorical_columns')
        min_cardinality = self.config.get('eda_limits', 'min_categorical_cardinality')
        max_cardinality = self.config.get('eda_limits', 'max_categorical_cardinality')
        
        for col in numeric_cols[:max_numeric]:
            vizs.append({
                'type': 'histogram',
                'title': f'Distribution of {col.replace("_", " ").title()}',
                'column': col,
                'columns': [col],
                'bins': 'auto'
            })
            
            vizs.append({
                'type': 'box',
                'title': f'{col.replace("_", " ").title()} - Outlier Detection',
                'column': col,
                'columns': [col]
            })
        
        for col in categorical_cols[:max_categorical]:
            n_unique = df[col].nunique()
            if min_cardinality <= n_unique <= max_cardinality:
                vizs.append({
                    'type': 'bar',
                    'title': f'{col.replace("_", " ").title()} Frequency',
                    'x_column': col,
                    'y_column': 'count',
                    'columns': [col],
                    'is_frequency': True
                })
        
        return vizs
    
    def _bivariate_analysis(self, df: pd.DataFrame, numeric_cols: List[str], 
                           categorical_cols: List[str], datetime_cols: List[str]) -> List[Dict]:
        """Generate bivariate visualizations with smart pairing"""
        vizs = []
        
        if len(numeric_cols) >= 2:
            vizs.append({
                'type': 'heatmap',
                'title': 'Correlation Matrix',
                'columns': numeric_cols,
                'heatmap_type': 'correlation'
            })
        
        max_pairs = self.config.get('eda_limits', 'max_numeric_pairs')
        uniqueness_threshold = self.config.get('correlation', 'high_uniqueness_threshold')
        min_corr_scatter = self.config.get('correlation', 'min_correlation_for_scatter')
        min_corr_bivariate = self.config.get('correlation', 'min_correlation_for_bivariate')
        
        for i, col1 in enumerate(numeric_cols[:max_pairs]):
            for col2 in numeric_cols[i+1:max_pairs]:
                col1_unique_ratio = df[col1].nunique() / len(df)
                col2_unique_ratio = df[col2].nunique() / len(df)
                
                if col1_unique_ratio > uniqueness_threshold and col2_unique_ratio > uniqueness_threshold:
                    corr = float(df[col1].corr(df[col2]))
                    if abs(corr) > min_corr_scatter:
                        vizs.append({
                            'type': 'scatter',
                            'title': f'{col2.replace("_", " ").title()} vs {col1.replace("_", " ").title()}',
                            'x_column': col1,
                            'y_column': col2,
                            'correlation': corr,
                            'columns': [col1, col2]
                        })
                else:
                    corr = float(df[col1].corr(df[col2]))
                    if abs(corr) > min_corr_bivariate:
                        vizs.append({
                            'type': 'scatter',
                            'title': f'{col2} vs {col1} (r={corr:.2f})',
                            'x_column': col1,
                            'y_column': col2,
                            'correlation': corr,
                            'columns': [col1, col2],
                            'show_regression': True
                        })
        
        max_num_cols = self.config.get('eda_limits', 'max_numeric_columns') // 2
        max_cat_cols = self.config.get('eda_limits', 'max_categorical_columns') // 2
        min_cat_card = self.config.get('eda_limits', 'min_categorical_cardinality')
        max_cat_card = self.config.get('eda_limits', 'max_categorical_cardinality') // 2
        
        for num_col in numeric_cols[:max_num_cols]:
            for cat_col in categorical_cols[:max_cat_cols]:
                if min_cat_card <= df[cat_col].nunique() <= max_cat_card:
                    vizs.append({
                        'type': 'box_by_category',
                        'title': f'{num_col.replace("_", " ").title()} by {cat_col.replace("_", " ").title()}',
                        'numeric_column': num_col,
                        'categorical_column': cat_col,
                        'columns': [num_col, cat_col]
                    })
        
        for date_col in datetime_cols[:2]:
            for num_col in numeric_cols[:3]:
                vizs.append({
                    'type': 'line',
                    'title': f'{num_col.replace("_", " ").title()} Over Time',
                    'x_column': date_col,
                    'y_column': num_col,
                    'columns': [date_col, num_col]
                })
        
        return vizs
    
    def _multivariate_analysis(self, df: pd.DataFrame, numeric_cols: List[str], 
                               categorical_cols: List[str]) -> List[Dict]:
        """Generate 3-variable visualizations"""
        vizs = []
        
        max_color_cats = self.config.get('eda_limits', 'max_color_categories')
        
        if len(numeric_cols) >= 2 and len(categorical_cols) >= 1:
            for cat_col in categorical_cols[:2]:
                if 2 <= df[cat_col].nunique() <= max_color_cats:
                    vizs.append({
                        'type': 'scatter_colored',
                        'title': f'{numeric_cols[1]} vs {numeric_cols[0]} (colored by {cat_col})',
                        'x_column': numeric_cols[0],
                        'y_column': numeric_cols[1],
                        'color_column': cat_col,
                        'columns': [numeric_cols[0], numeric_cols[1], cat_col]
                    })
        
        return vizs