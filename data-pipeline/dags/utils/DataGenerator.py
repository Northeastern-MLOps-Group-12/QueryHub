import random
import pandas as pd
from typing import List, Dict, Tuple
import os
import glob
import logging
import datetime

from utils.DataGenData.Templates.CTETemplates import _get_cte_templates
from utils.DataGenData.Templates.SETTemplates import _get_set_operation_templates
from utils.DataGenData.Templates.MultipleJoinsTemplates import _get_multiple_join_templates
from utils.DataGenData.Templates.SubQueryTemplates import _get_subquery_templates
from utils.DataGenData.Templates.WindowFunctionTemplates import _get_window_function_templates

from utils.DataGenData.DomainData.Ecommerce import _get_ecommerce_domain
from utils.DataGenData.DomainData.Education import _get_education_domain
from utils.DataGenData.DomainData.Finance import _get_finance_domain
from utils.DataGenData.DomainData.Gaming import _get_gaming_domain
from utils.DataGenData.DomainData.Healthcare import _get_healthcare_domain
from utils.DataGenData.DomainData.Hospitality import _get_hospitality_domain
from utils.DataGenData.DomainData.Logistics import _get_logistics_domain
from utils.DataGenData.DomainData.Manufacturing import _get_manufacturing_domain
from utils.DataGenData.DomainData.RealEstate import _get_real_estate_domain
from utils.DataGenData.DomainData.Retail import _get_retail_domain
from utils.DataGenData.DomainData.SocialMedia import _get_social_media_domain



class UltraDiverseSQLGenerator:
    """Generate maximum diversity synthetic text-to-SQL data"""
    
    def __init__(self):
        # 11 comprehensive domains
        self.domains = {
            'retail': _get_retail_domain(),
            'healthcare': _get_healthcare_domain(),
            'finance': _get_finance_domain(),
            'education': _get_education_domain(),
            'logistics': _get_logistics_domain(),
            'ecommerce': _get_ecommerce_domain(),
            'manufacturing': _get_manufacturing_domain(),
            'real_estate': _get_real_estate_domain(),
            'hospitality': _get_hospitality_domain(),
            'social_media': _get_social_media_domain(),
            'gaming': _get_gaming_domain()
        }
        
        self.templates = {
            'CTEs': _get_cte_templates(),
            'set operations': _get_set_operation_templates(),
            'multiple_joins': _get_multiple_join_templates(),
            'window functions': _get_window_function_templates(),
            'subqueries': _get_subquery_templates()
        }
        
        self.prompt_verbs = {
            'find': [
                'Find', 'Locate', 'Identify', 'Discover', 'Retrieve', 'Get', 'Fetch', 
                'Search for', 'Look up', 'Extract', 'Pull', 'Obtain', 'Detect', 'Pinpoint',
                'Uncover', 'Seek out', 'Hunt for', 'Track down'
            ],  
            
            'show': [
                'Show', 'Display', 'Present', 'Exhibit', 'Reveal', 'Demonstrate',
                'Illustrate', 'Showcase', 'Highlight', 'Expose', 'Visualize', 'Output',
                'Print', 'Render', 'Surface', 'Bring up'
            ],  
            
            'list': [
                'List', 'Enumerate', 'Show all', 'Display all', 'Catalog', 'Present all',
                'Itemize', 'Detail', 'Outline', 'Index', 'Compile', 'Record',
                'Register', 'Chronicle', 'Document'
            ],  
            
            'calculate': [
                'Calculate', 'Compute', 'Determine', 'Find', 'Derive', 'Evaluate',
                'Figure out', 'Work out', 'Estimate', 'Quantify', 'Measure', 'Assess',
                'Analyze', 'Tally', 'Sum up', 'Total', 'Count up'
            ],  
            
            'analyze': [
                'Analyze', 'Examine', 'Review', 'Investigate', 'Study', 'Assess',
                'Inspect', 'Scrutinize', 'Evaluate', 'Explore', 'Probe', 'Survey',
                'Dissect', 'Break down', 'Look into', 'Dive into'
            ],  
            
            'rank': [
                'Rank', 'Order', 'Sort', 'Arrange', 'Organize', 'Sequence',
                'Prioritize', 'Grade', 'Rate', 'Classify', 'Categorize', 'Position',
                'Place', 'Tier', 'Structure', 'Hierarchy'
            ],  
            
            'compare': [
                'Compare', 'Contrast', 'Evaluate', 'Assess', 'Measure',
                'Benchmark', 'Match up', 'Weigh', 'Stack up', 'Juxtapose',
                'Differentiate', 'Distinguish', 'Relate', 'Correlate'
            ],
            
            'identify': [
                'Identify', 'Spot', 'Recognize', 'Determine', 'Detect', 'Pinpoint',
                'Single out', 'Pick out', 'Flag', 'Mark', 'Note', 'Point out'
            ],  
            
            'filter': [
                'Filter', 'Select', 'Choose', 'Pick', 'Screen', 'Isolate',
                'Narrow down', 'Refine', 'Sift', 'Trim', 'Cull', 'Winnow'
            ],
            
            'summarize': [
                'Summarize', 'Aggregate', 'Consolidate', 'Combine', 'Merge', 'Total',
                'Roll up', 'Compile', 'Group', 'Collect', 'Gather', 'Assemble'
            ], 
            
            'classify': [
                'Classify', 'Categorize', 'Group', 'Segment', 'Divide', 'Partition',
                'Separate', 'Sort into', 'Break into', 'Organize into', 'Label'
            ], 
            
            'detect': [
                'Detect', 'Spot', 'Catch', 'Notice', 'Observe', 'Discover',
                'Uncover', 'Flag', 'Identify', 'Find', 'Locate', 'Pinpoint'
            ], 
            
            'track': [
                'Track', 'Monitor', 'Follow', 'Trace', 'Watch', 'Observe',
                'Survey', 'Keep tabs on', 'Chart', 'Record', 'Log'
            ],
            
            'project': [
                'Project', 'Forecast', 'Predict', 'Estimate', 'Anticipate', 'Expect',
                'Extrapolate', 'Model', 'Simulate', 'Envision'
            ],
            
            'segment': [
                'Segment', 'Divide', 'Split', 'Partition', 'Break down', 'Separate',
                'Slice', 'Bucket', 'Bin', 'Group into'
            ], 
            
            'normalize': [
                'Normalize', 'Standardize', 'Scale', 'Adjust', 'Balance', 'Calibrate',
                'Equalize', 'Level', 'Regularize', 'Transform'
            ], 
            
            'monitor': [
                'Monitor', 'Track', 'Watch', 'Observe', 'Follow', 'Survey',
                'Oversee', 'Supervise', 'Keep track of', 'Check'
            ],
            
            'forecast': [
                'Forecast', 'Predict', 'Project', 'Anticipate', 'Estimate', 'Expect',
                'Foresee', 'Envision', 'Extrapolate', 'Model'
            ],
            
            'count': [
                'Count', 'Tally', 'Enumerate', 'Number', 'Sum', 'Total',
                'Quantify', 'Calculate', 'Add up', 'Compute'
            ], 
            
            'highlight': [
                'Highlight', 'Emphasize', 'Spotlight', 'Feature', 'Showcase', 'Focus on',
                'Accentuate', 'Draw attention to', 'Point out', 'Single out'
            ], 
            
            'extract': [
                'Extract', 'Pull', 'Retrieve', 'Get', 'Fetch', 'Obtain',
                'Draw out', 'Take out', 'Isolate', 'Select'
            ],
            
            'combine': [
                'Combine', 'Merge', 'Unite', 'Join', 'Blend', 'Consolidate',
                'Integrate', 'Fuse', 'Mix', 'Pool together'
            ]
        }

    def calculate_diversity_stats(self) -> Dict:
        """Calculate expected unique combinations and duplicate rates - ACCURATE"""
        
        # Dynamic counts
        total_templates = sum(len(t) for t in self.templates.values())  # All 75 templates
        avg_templates = total_templates / len(self.templates)  # ~15 per complexity
        
        # Count verb variations dynamically
        total_verbs = sum(len(v) for v in self.prompt_verbs.values())
        avg_verbs = total_verbs / len(self.prompt_verbs) if self.prompt_verbs else 5
        
        # REALISTIC multiplicative factors
        # These are the parameters that ACTUALLY vary independently per sample:
        total_combinations = (
            total_templates *        # 75 templates total
            len(self.domains) *      # 11 domains
            5 *                      # 5 prompt variations per template
            int(avg_verbs) *         # ~12 verbs average (will be ~12 with your expansion)
            (10 * 12 * 28) *        # 3,360 date combinations (year × month × day)
            100000 *                 # threshold range (1-100,000)
            50 *                     # top_n range (1-50)
            (5 * 5 * 5)             # 125 SQL variations (agg × operators × where styles)
        )
        
        variations = {
            'total_templates': total_templates,
            'avg_templates_per_complexity': int(avg_templates),
            'complexity_types': len(self.templates),
            'domains': len(self.domains),
            'prompt_variations_per_template': 5,
            'total_unique_verbs': total_verbs,
            'avg_verbs_per_style': int(avg_verbs),
            'verb_styles': len(self.prompt_verbs),
            'date_combinations': 3360,  # 10 years × 12 months × 28 days
            'threshold_range': 100000,
            'top_n_range': 50,
            'agg_functions': 5,
            'comparison_operators': 5,
            'where_clause_styles': 5,
            'sql_style_combinations': 125  # 5 × 5 × 5
        }
        
        return {
            'variations': variations,
            'total_combinations': total_combinations
        }

    def calculate_expected_duplicates(self, sample_size: int) -> Dict:
        """Calculate expected duplicate rate using birthday paradox"""
        stats = self.calculate_diversity_stats()
        total_combos = stats['total_combinations']
        
        # Birthday paradox: P(duplicate) ≈ n²/(2N)
        pairs = (sample_size * (sample_size - 1)) / 2
        expected_duplicate_pairs = pairs / total_combos
        expected_duplicate_rows = expected_duplicate_pairs * 2
        
        # Safety: can't have negative or >100% duplicates
        duplicate_percentage = min(100, max(0, (expected_duplicate_rows / sample_size * 100))) if sample_size > 0 else 0
        expected_duplicate_rows = min(sample_size, max(0, int(expected_duplicate_rows)))
        
        return {
            'sample_size': sample_size,
            'total_combinations': total_combos,
            'expected_duplicate_rows': expected_duplicate_rows,
            'duplicate_percentage': round(duplicate_percentage, 6),  # More precision for tiny values
            'unique_rows': sample_size - expected_duplicate_rows,
            'unique_percentage': round(100 - duplicate_percentage, 4)
        }

    # Helper methods
    def get_valid_columns(self, table_config: Dict, requirement: str) -> List[str]:
        """Get valid columns based on requirement type"""
        if requirement == 'grouping_col':
            cols = table_config.get('grouping_cols', [])
            return cols if cols else [table_config['primary_key']]
        elif requirement == 'metric_col':
            return table_config.get('metric_cols', [])
        elif requirement == 'date_col':
            return table_config.get('date_cols', [])
        return []
    
    def find_foreign_key_relationship(self, table1: str, table2: str, domain: str) -> Tuple[str, str, str]:
        """Find foreign key relationship between two tables"""
        domain_config = self.domains[domain]
        
        # Check if table1 has FK to table2
        table1_config = domain_config['tables'][table1]
        if 'foreign_keys' in table1_config:
            for fk, ref_table in table1_config['foreign_keys'].items():
                if ref_table == table2:
                    pk = domain_config['tables'][table2]['primary_key']
                    return (fk, pk, fk.replace('_id', ''))
        
        # Check if table2 has FK to table1
        table2_config = domain_config['tables'][table2]
        if 'foreign_keys' in table2_config:
            for fk, ref_table in table2_config['foreign_keys'].items():
                if ref_table == table1:
                    pk = domain_config['tables'][table1]['primary_key']
                    return (fk, pk, fk.replace('_id', ''))
        
        # Default fallback
        return ('id', 'id', 'id')
    
    def generate_sql_context(self, tables: List[str], domain: str) -> str:
        """Generate CREATE TABLE and INSERT statements"""
        domain_config = self.domains[domain]
        context_parts = []
        
        for table_name in tables:
            if table_name not in domain_config['tables']:
                continue
                
            table_config = domain_config['tables'][table_name]
            columns = table_config['columns']
            
            # CREATE TABLE statement
            col_definitions = []
            for col_name, (col_type, _) in columns.items():
                col_definitions.append(f"{col_name} {col_type}")
            
            create_stmt = f"CREATE TABLE {table_name} ({', '.join(col_definitions)});"
            context_parts.append(create_stmt)
            
            # INSERT statements
            col_names = list(columns.keys())
            values_list = []
            
            num_rows = min(len(values) for _, values in columns.values())
            
            for i in range(num_rows):
                row_values = []
                for col_name in col_names:
                    _, sample_values = columns[col_name]
                    value = sample_values[i]
                    if isinstance(value, str):
                        row_values.append(f"'{value}'")
                    else:
                        row_values.append(str(value))
                values_list.append(f"({', '.join(row_values)})")
            
            insert_stmt = f"INSERT INTO {table_name} ({', '.join(col_names)}) VALUES {', '.join(values_list)};"
            context_parts.append(insert_stmt)
        
        return ' '.join(context_parts)
    
    def fill_template(self, template: Dict, domain: str) -> Dict:
        """Fill a template with semantically correct values and maximum diversity"""
        domain_config = self.domains[domain]
        table_names = list(domain_config['tables'].keys())
        
        # Find valid tables
        valid_tables = []
        for tname in table_names:
            tconfig = domain_config['tables'][tname]
            has_all_requirements = True
            
            for req in template.get('requires', []):
                if req == 'grouping_col' and not self.get_valid_columns(tconfig, 'grouping_col'):
                    has_all_requirements = False
                elif req == 'metric_col' and not self.get_valid_columns(tconfig, 'metric_col'):
                    has_all_requirements = False
                elif req == 'date_col' and not self.get_valid_columns(tconfig, 'date_col'):
                    has_all_requirements = False
                    
            if has_all_requirements:
                valid_tables.append(tname)
        
        if not valid_tables:
            raise ValueError(f"No valid tables in {domain}")
        
        # Select tables
        num_tables = len([t for t in template.get('tables_needed', []) if '{table' in t])
        if num_tables > len(valid_tables):
            raise ValueError(f"Need {num_tables} tables but only {len(valid_tables)} available")
        
        selected_tables = random.sample(valid_tables, num_tables)
        table1 = selected_tables[0]
        table2 = selected_tables[1] if len(selected_tables) > 1 else table1
        
        # Get columns
        table1_config = domain_config['tables'][table1]
        table2_config = domain_config['tables'][table2]
        
        group_cols1 = self.get_valid_columns(table1_config, 'grouping_col')
        group_cols2 = self.get_valid_columns(table2_config, 'grouping_col')
        metric_cols1 = self.get_valid_columns(table1_config, 'metric_col')
        metric_cols2 = self.get_valid_columns(table2_config, 'metric_col')
        date_cols1 = self.get_valid_columns(table1_config, 'date_col')
        
        # Get FK relationship
        fk1, pk2, fk_base = self.find_foreign_key_relationship(table1, table2, domain)
        
        # Natural language variation
        verb_style = template.get('verb_style', 'show')
        verb = random.choice(self.prompt_verbs.get(verb_style, ['Show']))
        
        # 🔥 HIGH-VARIATION PARAMETERS (ENHANCED)
        
        # 1. Expand numeric ranges (10x more options)
        threshold = random.randint(1, 100000)  # was (100, 10000)
        threshold2 = random.randint(threshold, min(threshold + 50000, 150000))
        top_n = random.randint(1, 50)  # was (3, 10)
        
        # 2. Add month/day variation (336x more options)
        year = random.randint(2015, 2024)  # was (2020, 2024)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        date_string = f'{year}-{month:02d}-{day:02d}'
        
        # 3. Vary aggregation functions (5x more options)
        agg_func = random.choice(['SUM', 'AVG', 'MAX', 'MIN', 'COUNT'])
        
        # 4. Vary comparison operators (5x more options)
        comparison_operators = ['>', '>=', '<', '<=', '!=']
        comparison_op = random.choice(comparison_operators)
        
        # 5. Vary WHERE clause styles (5x more options)
        interval_days = random.randint(30, 365)
        where_styles = [
            f"WHERE {{{{date_col}}}} >= '{date_string}'",
            f"WHERE {{{{date_col}}}} BETWEEN '{year}-01-01' AND '{year}-12-31'",
            f"WHERE YEAR({{{{date_col}}}}) = {year}",
            f"WHERE {{{{date_col}}}} >= '{year}-{month:02d}-01'",
            f"WHERE {{{{date_col}}}} > DATE_SUB(CURDATE(), INTERVAL {interval_days} DAY)"
        ]
        where_clause_style = random.choice(where_styles)
        
        # Build replacements
        replacements = {
            '{table1}': table1,
            '{table2}': table2,
            '{group_col}': random.choice(group_cols1),
            '{group_col1}': random.choice(group_cols1),
            '{group_col2}': random.choice(group_cols2) if group_cols2 else random.choice(group_cols1),
            '{partition_col}': random.choice(group_cols1),
            '{metric_col}': random.choice(metric_cols1) if metric_cols1 else 'id',
            '{metric_col2}': random.choice(metric_cols2) if metric_cols2 else random.choice(metric_cols1) if metric_cols1 else 'id',
            '{date_col}': random.choice(date_cols1) if date_cols1 else 'created_date',
            '{year}': str(year),
            '{month}': f'{month:02d}',
            '{day}': f'{day:02d}',
            '{date_string}': date_string,
            '{threshold}': str(threshold),
            '{threshold2}': str(threshold2),
            '{top_n}': str(top_n),
            '{cte_name}': f"{table1}_summary",
            '{pk1}': table1_config['primary_key'],
            '{pk2}': table2_config['primary_key'],
            '{fk1}': fk1,
            '{verb}': verb,
            '{agg_func}': agg_func,
            '{comparison_op}': comparison_op
        }
        
        # Fill template
        filled_sql = template['sql']
        filled_prompt = random.choice(template['prompt_templates'])
        filled_explanation = template['explanation']
        
        # Dynamic replacements for variety
        # Replace SUM with random aggregation function
        filled_sql = filled_sql.replace('SUM(', f'{agg_func}(')
        
        # Replace WHERE date clauses with varied styles
        filled_sql = filled_sql.replace("WHERE {date_col} >= '{year}-01-01'", where_clause_style)
        
        # Apply all placeholder replacements
        for placeholder, value in replacements.items():
            filled_sql = filled_sql.replace(placeholder, value)
            filled_prompt = filled_prompt.replace(placeholder, value)
            filled_explanation = filled_explanation.replace(placeholder, value)
        
        # Generate sql_context
        tables_used = [table1]
        if num_tables > 1:
            tables_used.append(table2)
        
        sql_context = self.generate_sql_context(tables_used, domain)
        
        return {
            'sql_prompt': filled_prompt,
            'sql_context': sql_context,
            'sql': filled_sql,
            'sql_explanation': filled_explanation,
            'domain': domain,
            'domain_description': domain_config['description']
        }
    
    def generate_balanced_dataset(self, target_counts: Dict[str, int]) -> pd.DataFrame:
        """Generate synthetic data to balance classes"""
        generated_data = []
        
        for complexity, target_count in target_counts.items():
            if complexity not in self.templates:
                print(f"Warning: No templates for {complexity}")
                continue
            
            templates = self.templates[complexity]
            domains = list(self.domains.keys())
            
            successful = 0
            attempts = 0
            max_attempts = target_count * 5
            
            print(f"Generating {complexity}... ", end='', flush=True)
            
            while successful < target_count and attempts < max_attempts:
                attempts += 1
                template = random.choice(templates)
                domain = random.choice(domains)
                
                try:
                    sample = self.fill_template(template, domain)
                    sample['sql_complexity'] = complexity
                    sample['sql_task_type'] = 'analytics and reporting'
                    generated_data.append(sample)
                    successful += 1
                    
                    # Progress indicator
                    if successful % 100 == 0:
                        print(f"{successful}...", end='', flush=True)
                        
                except (ValueError, IndexError):
                    continue
                except Exception:
                    continue
            
            print(f" ✓ {successful}/{target_count}")
            
            if successful < target_count:
                print(f"  ⚠️  Only generated {successful}/{target_count}")
        
        return pd.DataFrame(generated_data)

def GenerateAdditionalData(target_counts):
    """
    Generate synthetic SQL data with specified target counts per complexity.
    
    Args:
        target_counts: Dictionary mapping complexity types to target counts
                      Example: {'CTEs': 40000, 'set operations': 20000, ...}
    
    Returns:
        pd.DataFrame: Generated synthetic data
    """
    # Validate input
    if not isinstance(target_counts, dict):
        raise ValueError("target_counts must be a dictionary")
    
    if not target_counts:
        raise ValueError("target_counts cannot be empty")
    
    generator = UltraDiverseSQLGenerator()
    
    # Validate complexity types
    valid_complexities = set(generator.templates.keys())
    invalid = set(target_counts.keys()) - valid_complexities
    if invalid:
        raise ValueError(f"Invalid complexity types: {invalid}. Valid types: {valid_complexities}")
    
    print("="*80)
    print("🚀 ULTRA-DIVERSE SQL TEMPLATE GENERATOR (ENHANCED)")
    print("="*80)
    
    # Calculate and display diversity statistics
    diversity_stats = generator.calculate_diversity_stats()
    
    print(f"\n📊 DIVERSITY CONFIGURATION:")
    print("-"*80)
    for param, count in diversity_stats['variations'].items():
        print(f"   • {param.replace('_', ' ').title()}: {count:,} options")
    
    print(f"\n🎯 TOTAL UNIQUE COMBINATIONS:")
    print("-"*80)
    total_combos = diversity_stats['total_combinations']
    if total_combos > 1e15:
        print(f"   {total_combos:.2e} ({total_combos/1e15:.2f} quadrillion)")
    elif total_combos > 1e12:
        print(f"   {total_combos:.2e} ({total_combos/1e12:.2f} trillion)")
    elif total_combos > 1e9:
        print(f"   {total_combos:.2e} ({total_combos/1e9:.2f} billion)")
    else:
        print(f"   {total_combos:,}")
    
    print(f"\n📈 EXPECTED DUPLICATE ANALYSIS:")
    print("-"*80)
    
    # Calculate for the actual target counts provided
    sample_sizes = list(target_counts.values())
    
    print(f"\n{'Sample Size':<15} {'Duplicates':<15} {'Dup %':<12} {'Unique Rows':<15} {'Unique %':<12}")
    print("-"*80)
    
    for size in sample_sizes:
        dup_stats = generator.calculate_expected_duplicates(size)
        print(f"{size:,}".ljust(15), end='')
        print(f"~{dup_stats['expected_duplicate_rows']:,}".ljust(15), end='')
        print(f"{dup_stats['duplicate_percentage']:.4f}%".ljust(12), end='')
        print(f"~{dup_stats['unique_rows']:,}".ljust(15), end='')
        print(f"{dup_stats['unique_percentage']:.2f}%")
    
    # Display target counts
    print("\n" + "="*80)
    print("🎯 TARGET GENERATION COUNTS")
    print("="*80 + "\n")
    
    total_target = sum(target_counts.values())
    for complexity, count in target_counts.items():
        print(f"   • {complexity}: {count:,} samples")
    print(f"\n   📊 Total: {total_target:,} samples")
    
    print("\n" + "="*80)
    print("🔄 GENERATING SYNTHETIC DATA")
    print("="*80 + "\n")
    
    synthetic_df = generator.generate_balanced_dataset(target_counts)

    output_dir = '/opt/airflow/data'
    os.makedirs(output_dir, exist_ok=True)

    path = f'{output_dir}/synthetic_data.csv'
    
    synthetic_df.to_csv(path,index=False)

    logging.info(f"Saved synthetic data at {path}")
