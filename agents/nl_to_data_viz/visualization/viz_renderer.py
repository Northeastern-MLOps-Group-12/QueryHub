# ============================================================================
# FILE: AgentFiles/Visualization/viz_renderer.py
# ============================================================================
"""
Render visualizations to HTML/PNG files
"""

import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import json
import numpy as np
import os

class VisualizationRenderer:
    """Render and save visualizations locally"""
    
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            self.base_dir = Path(os.getcwd()) / "query_visualizations"
        else:
            self.base_dir = Path(base_dir)
        
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_session_folder(self, user_id: str, db_name: str, 
                             session_id: str) -> Path:
        """Create unique folder for this visualization session"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{db_name}_{session_id}_{timestamp}"
        
        session_path = self.base_dir / user_id / folder_name
        session_path.mkdir(parents=True, exist_ok=True)
        
        return session_path
    
    def render_all(self, df: pd.DataFrame, visualizations: List[Dict], 
                   session_path: Path) -> Dict:
        """Render all visualizations and save to session folder"""
        rendered_files = []
        errors = []
        
        for i, viz in enumerate(visualizations, 1):
            try:
                file_path = self._render_single(df, viz, session_path, i)
                if file_path:
                    rendered_files.append(str(file_path))
            except Exception as e:
                errors.append({
                    'viz_index': i,
                    'viz_type': viz.get('type'),
                    'error': str(e)
                })
                print(f"Error rendering viz {i}: {e}")
        
        # Save metadata
        metadata = {
            'total_visualizations': len(visualizations),
            'rendered': len(rendered_files),
            'errors': errors,
            'timestamp': datetime.now().isoformat(),
            'files': rendered_files
        }
        
        metadata_path = session_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            'files': rendered_files,
            'metadata': metadata,
            'session_path': str(session_path)
        }
    
    def _render_single(self, df: pd.DataFrame, viz: Dict, 
                      session_path: Path, index: int) -> Path:
        """Render a single visualization"""
        viz_type = viz['type']
        title = viz['title'].replace('/', '_').replace('\\', '_').replace(':', '_')
        
        # Word cloud needs PNG, others use HTML
        if viz_type == 'wordcloud':
            filename = f"viz_{index:02d}_{viz_type}_{title[:50]}.png"
        else:
            filename = f"viz_{index:02d}_{viz_type}_{title[:50]}.html"
        
        file_path = session_path / filename
        
        # Render based on type
        if viz_type == 'bar':
            self._render_bar(df, viz, file_path)
        elif viz_type == 'histogram':
            self._render_histogram(df, viz, file_path)
        elif viz_type == 'scatter':
            self._render_scatter(df, viz, file_path)
        elif viz_type == 'line':
            self._render_line(df, viz, file_path)
        elif viz_type == 'box':
            self._render_box(df, viz, file_path)
        elif viz_type == 'box_by_category':
            self._render_box_by_category(df, viz, file_path)
        elif viz_type == 'heatmap':
            self._render_heatmap(df, viz, file_path)
        elif viz_type == 'wordcloud':
            self._render_wordcloud(df, viz, file_path)
        elif viz_type == 'scatter_colored':
            self._render_scatter_colored(df, viz, file_path)
        else:
            return None
        
        return file_path
    
    def _render_bar(self, df: pd.DataFrame, viz: Dict, file_path: Path):
        """Render bar chart"""
        if viz.get('is_frequency', False):
            value_counts = df[viz['x_column']].value_counts()
            if 'top_n' in viz:
                value_counts = value_counts.head(viz['top_n'])
            
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=viz['title'],
                labels={'x': viz['x_column'], 'y': 'Count'}
            )
            fig.update_layout(xaxis_tickangle=-45)
        else:
            plot_df = df.copy()
            if viz.get('sort_by') and viz.get('sort_descending'):
                plot_df = plot_df.sort_values(viz['sort_by'], ascending=False).head(20)
            
            fig = px.bar(plot_df, x=viz['x_column'], y=viz['y_column'], title=viz['title'])
            fig.update_layout(xaxis_tickangle=-45)
        
        fig.write_html(file_path)
    
    def _render_histogram(self, df: pd.DataFrame, viz: Dict, file_path: Path):
        fig = px.histogram(df, x=viz['column'], title=viz['title'], nbins=30)
        fig.write_html(file_path)
    
    def _render_scatter(self, df: pd.DataFrame, viz: Dict, file_path: Path):
        fig = px.scatter(df, x=viz['x_column'], y=viz['y_column'], title=viz['title'])
        
        if viz.get('show_regression', False):
            z = np.polyfit(df[viz['x_column']].dropna(), df[viz['y_column']].dropna(), 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=df[viz['x_column']], 
                y=p(df[viz['x_column']]),
                mode='lines',
                name='Trend',
                line=dict(color='red', dash='dash')
            ))
        
        fig.write_html(file_path)
    
    def _render_line(self, df: pd.DataFrame, viz: Dict, file_path: Path):
        fig = px.line(df, x=viz['x_column'], y=viz['y_column'], title=viz['title'])
        fig.write_html(file_path)
    
    def _render_box(self, df: pd.DataFrame, viz: Dict, file_path: Path):
        fig = px.box(df, y=viz['column'], title=viz['title'])
        fig.write_html(file_path)
    
    def _render_box_by_category(self, df: pd.DataFrame, viz: Dict, file_path: Path):
        fig = px.box(df, x=viz['categorical_column'], y=viz['numeric_column'], title=viz['title'])
        fig.write_html(file_path)
    
    def _render_heatmap(self, df: pd.DataFrame, viz: Dict, file_path: Path):
        corr_matrix = df[viz['columns']].corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        fig.update_layout(title=viz['title'])
        fig.write_html(file_path)
    
    def _render_wordcloud(self, df: pd.DataFrame, viz: Dict, file_path: Path):
        """Render word cloud - saves as PNG"""
        word_freq = viz['word_frequencies']
        
        wordcloud = WordCloud(
            width=1200, 
            height=600,
            background_color='white',
            max_words=viz.get('max_words', 200),
            colormap='viridis'
        ).generate_from_frequencies(word_freq)
        
        plt.figure(figsize=(15, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(viz['title'], fontsize=20)
        plt.tight_layout(pad=0)
        plt.savefig(file_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _render_scatter_colored(self, df: pd.DataFrame, viz: Dict, file_path: Path):
        fig = px.scatter(
            df, 
            x=viz['x_column'], 
            y=viz['y_column'], 
            color=viz['color_column'],
            title=viz['title']
        )
        fig.write_html(file_path)