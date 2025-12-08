# ============================================================================
# FILE: AgentFiles/Utils/configLoader/config_loader.py (UPDATED)
# ============================================================================

import yaml
from pathlib import Path
from typing import Dict, Any

class VisualizationConfig:
    """Load and manage visualization configuration"""
    
    def __init__(self, config_path: str = "config/visualization_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration as fallback"""
        return {
            'id_patterns': {
                'exact_match': ["id", "ID", "index", "Index"],
                'contains': ["_id", "_ID", "id_", "ID_"],
                'ends_with': ["Id", "ID"]
            },
            'bi_limits': {
                'max_graphs': 10,
                'wordcloud_threshold': 200,
                'wordcloud_top_n': 20,
                'max_categories_for_bar': 50
            },
            'eda_limits': {
                'max_univariate': 15,
                'max_bivariate': 20,
                'max_total_graphs': 35,
                'max_numeric_columns': 10,
                'max_categorical_columns': 5,
                'max_categorical_cardinality': 30,
                'min_categorical_cardinality': 2,
                'max_color_categories': 10,
                'max_numeric_pairs': 8
            },
            'correlation': {
                'high_uniqueness_threshold': 0.95,
                'min_correlation_for_scatter': 0.3,
                'min_correlation_for_bivariate': 0.5
            },
            'categorical': {
                'high_cardinality_threshold': 50,
                'text_stats_min_cardinality': 51
            },
            'missing_data': {
                'min_missing_percentage': 1.0
            },
            'sampling': {
                'scatter_threshold': 1000,
                'scatter_ratio': 0.50,
                'histogram_threshold': 10000,
                'histogram_ratio': 0.30
            }
        }
    
    def get(self, *keys):
        """Get nested configuration value"""
        value = self.config
        for key in keys:
            value = value.get(key, {})
        return value