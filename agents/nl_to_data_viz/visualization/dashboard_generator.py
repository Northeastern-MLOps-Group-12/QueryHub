# ============================================================================
# FILE: AgentFiles/Visualization/dashboard_generator.py (COMPLETE)
# ============================================================================

from pathlib import Path
from typing import List, Dict
import base64


class DashboardGenerator:
    """Generate responsive HTML dashboards with all visualizations"""
    
    def generate_dashboard(self, session_path: Path, visualizations: List[Dict], 
                          metadata: Dict) -> Path:
        """Create local dashboard with relative file paths"""
        dashboard_path = session_path / "dashboard.html"
        
        html_content = self._generate_local_html(session_path, visualizations, metadata)
        
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return dashboard_path
    
    def _generate_local_html(self, session_path: Path, visualizations: List[Dict], 
                            metadata: Dict) -> str:
        """Generate HTML dashboard with relative paths for local viewing"""
        
        viz_cards = []
        
        for i, viz in enumerate(visualizations, 1):
            if viz['type'] == 'text_stats':
                card_html = self._create_text_stats_card(viz, i)
            else:
                card_html = self._create_local_viz_card(session_path, viz, i)
            
            if card_html:
                viz_cards.append(card_html)
        
        cards_html = '\n'.join(viz_cards)
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Query Visualization Dashboard</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
               min-height: 100vh; padding: 20px; }}
        .container {{ max-width: 1600px; margin: 0 auto; }}
        .header {{ background: white; border-radius: 12px; padding: 30px; margin-bottom: 30px;
                   box-shadow: 0 10px 30px rgba(0,0,0,0.2); }}
        .header h1 {{ color: #2d3748; font-size: 32px; margin-bottom: 10px; }}
        .header .meta {{ color: #718096; font-size: 14px; }}
        .header .stats {{ display: flex; gap: 30px; margin-top: 20px; flex-wrap: wrap; }}
        .stat {{ background: #f7fafc; padding: 15px 25px; border-radius: 8px; border-left: 4px solid #667eea; }}
        .stat-label {{ color: #718096; font-size: 12px; text-transform: uppercase; }}
        .stat-value {{ color: #2d3748; font-size: 24px; font-weight: bold; margin-top: 5px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(550px, 1fr));
                 gap: 25px; margin-bottom: 30px; }}
        @media (max-width: 768px) {{ .grid {{ grid-template-columns: 1fr; }} }}
        .viz-card {{ background: white; border-radius: 12px; overflow: hidden;
                     box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                     transition: transform 0.3s ease, box-shadow 0.3s ease; }}
        .viz-card:hover {{ transform: translateY(-5px); box-shadow: 0 8px 25px rgba(0,0,0,0.15); }}
        .viz-header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                       color: white; padding: 20px; font-size: 16px; font-weight: 600; }}
        .viz-content {{ padding: 0; background: white; }}
        .viz-content iframe {{ width: 100%; height: 550px; border: none; display: block; }}
        .viz-content img {{ width: 100%; height: auto; display: block; }}
        .text-content {{ padding: 25px; font-size: 14px; line-height: 1.6; color: #2d3748; }}
        .text-content h3 {{ color: #2d3748; margin-bottom: 15px; font-size: 18px; }}
        .text-content p {{ margin: 10px 0; }}
        .text-content table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        .text-content th {{ background: #f7fafc; padding: 12px; text-align: left; 
                           border-bottom: 2px solid #e2e8f0; font-weight: 600; }}
        .text-content td {{ padding: 10px; border-bottom: 1px solid #e2e8f0; }}
        .text-content .highlight {{ background: #fef3c7; padding: 2px 6px; border-radius: 3px; 
                                    font-weight: 600; }}
        .text-content .metric {{ font-weight: bold; color: #667eea; }}
        .full-width {{ grid-column: 1 / -1; }}
        .footer {{ text-align: center; color: white; padding: 20px; margin-top: 30px; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Query Visualization Dashboard</h1>
            <div class="meta">
                <strong>Mode:</strong> {metadata.get('intent', 'N/A').upper()} | 
                <strong>Data:</strong> {metadata.get('data_rows', 0)} rows × {metadata.get('data_columns', 0)} columns
            </div>
            <div class="stats">
                <div class="stat">
                    <div class="stat-label">Total Visualizations</div>
                    <div class="stat-value">{metadata.get('total_graphs', 0)}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Rendered</div>
                    <div class="stat-value">{metadata.get('rendered_files', 0)}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Mode</div>
                    <div class="stat-value">{metadata.get('intent', 'N/A').upper()}</div>
                </div>
            </div>
        </div>
        <div class="grid">
            {cards_html}
        </div>
        <div class="footer">Generated at {metadata.get('timestamp', 'N/A')}</div>
    </div>
</body>
</html>"""
        
        return html
    
    def _create_text_stats_card(self, viz: Dict, index: int) -> str:
        """Create text-based statistics card"""
        subtype = viz.get('subtype', '')
        
        if subtype == 'missing_data':
            return self._create_missing_data_card(viz)
        elif subtype == 'high_cardinality':
            return self._create_high_cardinality_card(viz)
        elif subtype == 'dataframe_info':
            return self._create_dataframe_info_card(viz)
        elif subtype == 'statistical_summary':
            return self._create_statistical_summary_card(viz)
        
        return ""
    
    def _create_dataframe_info_card(self, viz: Dict) -> str:
        """Create DataFrame info card (df.info equivalent)"""
        stats = viz['stats']
        
        rows_html = []
        for item in stats['columns']:
            rows_html.append(f"""
                <tr>
                    <td>{item['column']}</td>
                    <td>{item['dtype']}</td>
                    <td class="metric">{item['non_null']:,}</td>
                    <td>{item['null_count']:,}</td>
                    <td>{item['memory_kb']:.1f} KB</td>
                </tr>
            """)
        
        table_html = '\n'.join(rows_html)
        
        return f"""
        <div class="viz-card full-width">
            <div class="viz-header">{viz['title']}</div>
            <div class="viz-content">
                <div class="text-content">
                    <p><strong>Shape:</strong> <span class="metric">{stats['total_rows']:,} rows × {stats['total_columns']} columns</span></p>
                    <p><strong>Total Memory:</strong> <span class="metric">{stats['total_memory_mb']:.2f} MB</span></p>
                    
                    <h3 style="margin-top: 20px;">Column Details</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Column</th>
                                <th>Data Type</th>
                                <th>Non-Null Count</th>
                                <th>Null Count</th>
                                <th>Memory</th>
                            </tr>
                        </thead>
                        <tbody>
                            {table_html}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        """
    
    def _create_statistical_summary_card(self, viz: Dict) -> str:
        """Create statistical summary card (df.describe equivalent)"""
        stats_data = viz['stats']
        
        rows_html = []
        for item in stats_data:
            rows_html.append(f"""
                <tr>
                    <td><strong>{item['column']}</strong></td>
                    <td>{item['count']:,}</td>
                    <td>{item['mean']:.2f}</td>
                    <td>{item['std']:.2f}</td>
                    <td>{item['min']:.2f}</td>
                    <td>{item['q25']:.2f}</td>
                    <td>{item['median']:.2f}</td>
                    <td>{item['q75']:.2f}</td>
                    <td>{item['max']:.2f}</td>
                </tr>
            """)
        
        table_html = '\n'.join(rows_html)
        
        return f"""
        <div class="viz-card full-width">
            <div class="viz-header">{viz['title']}</div>
            <div class="viz-content">
                <div class="text-content">
                    <h3>Descriptive Statistics for Numeric Columns</h3>
                    <div style="overflow-x: auto;">
                        <table>
                            <thead>
                                <tr>
                                    <th>Column</th>
                                    <th>Count</th>
                                    <th>Mean</th>
                                    <th>Std Dev</th>
                                    <th>Min</th>
                                    <th>25%</th>
                                    <th>Median</th>
                                    <th>75%</th>
                                    <th>Max</th>
                                </tr>
                            </thead>
                            <tbody>
                                {table_html}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
        """
    
    def _create_missing_data_card(self, viz: Dict) -> str:
        """Create missing data analysis card"""
        stats = viz['stats']
        total_rows = viz['total_rows']
        
        rows_html = []
        for item in stats:
            rows_html.append(f"""
                <tr>
                    <td>{item['column']}</td>
                    <td class="metric">{item['missing_count']:,}</td>
                    <td><span class="highlight">{item['missing_percentage']:.2f}%</span></td>
                </tr>
            """)
        
        table_html = '\n'.join(rows_html)
        
        return f"""
        <div class="viz-card full-width">
            <div class="viz-header">{viz['title']}</div>
            <div class="viz-content">
                <div class="text-content">
                    <h3>Columns with Missing Values (Total Rows: {total_rows:,})</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Column Name</th>
                                <th>Missing Count</th>
                                <th>Missing %</th>
                            </tr>
                        </thead>
                        <tbody>
                            {table_html}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        """
    
    def _create_high_cardinality_card(self, viz: Dict) -> str:
        """Create high-cardinality column statistics card"""
        stats = viz['stats']
        
        top_5_html = []
        for item in stats['top_5']:
            top_5_html.append(f"""
                <tr>
                    <td>{item['value']}</td>
                    <td class="metric">{item['count']:,}</td>
                    <td>{item['percentage']:.2f}%</td>
                </tr>
            """)
        
        top_table = '\n'.join(top_5_html)
        
        return f"""
        <div class="viz-card full-width">
            <div class="viz-header">{viz['title']}</div>
            <div class="viz-content">
                <div class="text-content">
                    <h3>Column: {stats['column_name']}</h3>
                    <p><strong>Total Unique Values:</strong> <span class="metric">{stats['total_unique']:,}</span> out of {stats['total_rows']:,} rows</p>
                    <p><strong>Most Frequent:</strong> {stats['most_frequent']} (appears {stats['most_frequent_count']:,} times)</p>
                    <p><strong>Singleton Values:</strong> {stats['singleton_count']:,} values appear only once</p>
                    
                    <h3 style="margin-top: 20px;">Top 5 Most Frequent Values</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Value</th>
                                <th>Count</th>
                                <th>Percentage</th>
                            </tr>
                        </thead>
                        <tbody>
                            {top_table}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        """
    
    def _create_local_viz_card(self, session_path: Path, viz: Dict, index: int) -> str:
        """Create HTML card with relative file paths"""
        viz_type = viz['type']
        title = viz['title'].replace('/', '_').replace('\\', '_').replace(':', '_')
        
        if viz_type == 'wordcloud':
            filename = f"viz_{index:02d}_{viz_type}_{title[:50]}.png"
        else:
            filename = f"viz_{index:02d}_{viz_type}_{title[:50]}.html"
        
        file_path = session_path / filename
        
        if not file_path.exists():
            return ""
        
        if viz_type == 'wordcloud':
            with open(file_path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')
            
            return f"""
        <div class="viz-card">
            <div class="viz-header">{viz['title']}</div>
            <div class="viz-content">
                <img src="data:image/png;base64,{img_data}" alt="{viz['title']}">
            </div>
        </div>
            """
        else:
            return f"""
        <div class="viz-card">
            <div class="viz-header">{viz['title']}</div>
            <div class="viz-content">
                <iframe src="{filename}"></iframe>
            </div>
        </div>
            """
    
    def generate_cloud_dashboard(self, cloud_files: List[Dict], visualizations: List[Dict],
                                metadata: Dict, output_path: Path) -> Path:
        """Generate dashboard with absolute cloud URLs for public sharing"""
        
        viz_cards = []
        file_map = {f['filename']: f['url'] for f in cloud_files}
        
        for i, viz in enumerate(visualizations, 1):
            if viz['type'] == 'text_stats':
                card_html = self._create_text_stats_card(viz, i)
                viz_cards.append(card_html)
            else:
                viz_type = viz['type']
                title = viz['title'].replace('/', '_').replace('\\', '_').replace(':', '_')
                
                if viz_type == 'wordcloud':
                    filename = f"viz_{i:02d}_{viz_type}_{title[:50]}.png"
                else:
                    filename = f"viz_{i:02d}_{viz_type}_{title[:50]}.html"
                
                url = file_map.get(filename)
                
                if url:
                    if filename.endswith('.png'):
                        card = f"""
        <div class="viz-card">
            <div class="viz-header">{viz['title']}</div>
            <div class="viz-content">
                <img src="{url}" alt="{viz['title']}">
            </div>
        </div>
                        """
                    else:
                        card = f"""
        <div class="viz-card">
            <div class="viz-header">{viz['title']}</div>
            <div class="viz-content">
                <iframe src="{url}"></iframe>
            </div>
        </div>
                        """
                    
                    viz_cards.append(card)
        
        cards_html = '\n'.join(viz_cards)
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Query Visualization Dashboard</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
               min-height: 100vh; padding: 20px; }}
        .container {{ max-width: 1600px; margin: 0 auto; }}
        .header {{ background: white; border-radius: 12px; padding: 30px; margin-bottom: 30px;
                   box-shadow: 0 10px 30px rgba(0,0,0,0.2); }}
        .header h1 {{ color: #2d3748; font-size: 32px; margin-bottom: 10px; }}
        .header .meta {{ color: #718096; font-size: 14px; }}
        .header .stats {{ display: flex; gap: 30px; margin-top: 20px; flex-wrap: wrap; }}
        .stat {{ background: #f7fafc; padding: 15px 25px; border-radius: 8px; border-left: 4px solid #667eea; }}
        .stat-label {{ color: #718096; font-size: 12px; text-transform: uppercase; }}
        .stat-value {{ color: #2d3748; font-size: 24px; font-weight: bold; margin-top: 5px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(550px, 1fr));
                 gap: 25px; margin-bottom: 30px; }}
        @media (max-width: 768px) {{ .grid {{ grid-template-columns: 1fr; }} }}
        .viz-card {{ background: white; border-radius: 12px; overflow: hidden;
                     box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                     transition: transform 0.3s ease, box-shadow 0.3s ease; }}
        .viz-card:hover {{ transform: translateY(-5px); box-shadow: 0 8px 25px rgba(0,0,0,0.15); }}
        .viz-header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                       color: white; padding: 20px; font-size: 16px; font-weight: 600; }}
        .viz-content {{ padding: 0; background: white; }}
        .viz-content iframe {{ width: 100%; height: 550px; border: none; display: block; }}
        .viz-content img {{ width: 100%; height: auto; display: block; }}
        .text-content {{ padding: 25px; font-size: 14px; line-height: 1.6; color: #2d3748; }}
        .text-content h3 {{ color: #2d3748; margin-bottom: 15px; font-size: 18px; }}
        .text-content p {{ margin: 10px 0; }}
        .text-content table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        .text-content th {{ background: #f7fafc; padding: 12px; text-align: left; 
                           border-bottom: 2px solid #e2e8f0; font-weight: 600; }}
        .text-content td {{ padding: 10px; border-bottom: 1px solid #e2e8f0; }}
        .text-content .highlight {{ background: #fef3c7; padding: 2px 6px; border-radius: 3px; 
                                    font-weight: 600; }}
        .text-content .metric {{ font-weight: bold; color: #667eea; }}
        .full-width {{ grid-column: 1 / -1; }}
        .footer {{ text-align: center; color: white; padding: 20px; margin-top: 30px; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Query Visualization Dashboard</h1>
            <div class="meta">
                <strong>Mode:</strong> {metadata.get('intent', 'N/A').upper()} | 
                <strong>Data:</strong> {metadata.get('data_rows', 0)} rows × {metadata.get('data_columns', 0)} columns
            </div>
            <div class="stats">
                <div class="stat">
                    <div class="stat-label">Total Visualizations</div>
                    <div class="stat-value">{metadata.get('total_graphs', 0)}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Rendered</div>
                    <div class="stat-value">{metadata.get('rendered_files', 0)}</div>
                </div>
            </div>
        </div>
        <div class="grid">
            {cards_html}
        </div>
        <div class="footer">Generated at {metadata.get('timestamp', 'N/A')}</div>
    </div>
</body>
</html>"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return output_path
