# ============================================================================
# FILE: AgentFiles/Visualization/bi_visualizer.py (UPDATED)
# ============================================================================

from typing import List, Dict
import pandas as pd
import numpy as np
from .column_analyzer import ColumnAnalyzer
from ..utils.config_loader import VisualizationConfig


class BIVisualizer:
    """Generate BI visualizations: Original × Generated only"""
    
    def __init__(self, analyzer: ColumnAnalyzer):
        self.analyzer = analyzer
        self.config = VisualizationConfig()
    
    def generate(self, df: pd.DataFrame, sql_query: str, table_details: List[Dict]) -> List[Dict]:
        """Generate BI visualizations"""
        columns = self.analyzer.detect_generated_columns(df, sql_query, table_details)
        original_cols = columns['original']
        generated_cols = columns['generated']
        
        visualizations = []
        
        original_cols = [col for col in original_cols if not self.analyzer.is_id_column(col)]
        
        if not generated_cols:
            categories = self.analyzer.categorize_columns(df)
            generated_cols = [col for col in categories['numeric'] if not self.analyzer.is_id_column(col)]
            original_cols = [col for col in original_cols if col not in generated_cols]
        
        wordcloud_threshold = self.config.get('bi_limits', 'wordcloud_threshold')
        max_categories = self.config.get('bi_limits', 'max_categories_for_bar')
        top_n = self.config.get('bi_limits', 'wordcloud_top_n')
        
        for gen_col in generated_cols:
            if pd.api.types.is_numeric_dtype(df[gen_col].dtype):
                visualizations.append(self._create_histogram(df, gen_col))
            else:
                n_unique = df[gen_col].nunique()
                
                if n_unique >= wordcloud_threshold:
                    visualizations.append(self._create_word_cloud(df, gen_col))
                    visualizations.append(self._create_top_n_bar(df, gen_col, top_n=top_n))
                
                if 2 <= n_unique <= max_categories:
                    visualizations.append(self._create_category_bar(df, gen_col))
        
        for orig_col in original_cols:
            for gen_col in generated_cols:
                viz = self._create_combination_viz(df, orig_col, gen_col)
                if viz:
                    visualizations.append(viz)
        
        max_graphs = self.config.get('bi_limits', 'max_graphs')
        return visualizations[:max_graphs]
    
    def _create_combination_viz(self, df: pd.DataFrame, x_col: str, y_col: str) -> Dict:
        """Create visualization for original × generated combination"""
        x_dtype = df[x_col].dtype
        y_dtype = df[y_col].dtype
        
        if not pd.api.types.is_numeric_dtype(x_dtype) and pd.api.types.is_numeric_dtype(y_dtype):
            return {
                'type': 'bar',
                'title': f'{y_col.replace("_", " ").title()} by {x_col.replace("_", " ").title()}',
                'x_column': x_col,
                'y_column': y_col,
                'columns': [x_col, y_col],
                'sort_by': y_col,
                'sort_descending': True
            }
        
        elif pd.api.types.is_datetime64_any_dtype(x_dtype) and pd.api.types.is_numeric_dtype(y_dtype):
            return {
                'type': 'line',
                'title': f'{y_col.replace("_", " ").title()} Over Time',
                'x_column': x_col,
                'y_column': y_col,
                'columns': [x_col, y_col]
            }
        
        elif pd.api.types.is_numeric_dtype(x_dtype) and pd.api.types.is_numeric_dtype(y_dtype):
            corr = float(df[x_col].corr(df[y_col]))
            return {
                'type': 'scatter',
                'title': f'{y_col.replace("_", " ").title()} vs {x_col.replace("_", " ").title()}',
                'x_column': x_col,
                'y_column': y_col,
                'correlation': corr,
                'columns': [x_col, y_col]
            }
        
        return None
    
    def _create_histogram(self, df: pd.DataFrame, col: str) -> Dict:
        """Create distribution histogram for generated column"""
        return {
            'type': 'histogram',
            'title': f'Distribution of {col.replace("_", " ").title()}',
            'column': col,
            'columns': [col],
            'bins': 'auto'
        }
    
    def _create_word_cloud(self, df: pd.DataFrame, col: str) -> Dict:
        """Create word cloud for high-cardinality categorical column"""
        value_counts = df[col].value_counts().to_dict()
        
        return {
            'type': 'wordcloud',
            'title': f'Word Cloud: {col.replace("_", " ").title()}',
            'column': col,
            'columns': [col],
            'word_frequencies': value_counts,
            'max_words': 200
        }
    
    def _create_top_n_bar(self, df: pd.DataFrame, col: str, top_n: int = 20) -> Dict:
        """Create top N bar chart for high-cardinality categorical column"""
        return {
            'type': 'bar',
            'title': f'Top {top_n} {col.replace("_", " ").title()}',
            'x_column': col,
            'y_column': 'count',
            'columns': [col],
            'is_frequency': True,
            'top_n': top_n,
            'sort_descending': True
        }
    
    def _create_category_bar(self, df: pd.DataFrame, col: str) -> Dict:
        """Create bar chart for categorical column frequency"""
        return {
            'type': 'bar',
            'title': f'{col.replace("_", " ").title()} Distribution',
            'x_column': col,
            'y_column': 'count',
            'columns': [col],
            'is_frequency': True,
            'sort_descending': True
        }

