import random
import pandas as pd
from typing import List, Dict, Tuple
import os
import glob
import logging
import datetime

class UltraDiverseSQLGenerator:
    """Generate maximum diversity synthetic text-to-SQL data"""
    
    def __init__(self):
        # 11 comprehensive domains
        self.domains = {
            'retail': self._get_retail_domain(),
            'healthcare': self._get_healthcare_domain(),
            'finance': self._get_finance_domain(),
            'education': self._get_education_domain(),
            'logistics': self._get_logistics_domain(),
            'ecommerce': self._get_ecommerce_domain(),
            'manufacturing': self._get_manufacturing_domain(),
            'real_estate': self._get_real_estate_domain(),
            'hospitality': self._get_hospitality_domain(),
            'social_media': self._get_social_media_domain(),
            'gaming': self._get_gaming_domain()
        }
        
        self.templates = {
            'CTEs': self._get_cte_templates(),
            'set operations': self._get_set_operation_templates(),
            'multiple_joins': self._get_multiple_join_templates(),
            'window functions': self._get_window_function_templates(),
            'subqueries': self._get_subquery_templates()
        }
        
        # 5+ natural language variations per verb style
        self.prompt_verbs = {
            'find': ['Find', 'Locate', 'Identify', 'Discover', 'Retrieve', 'Get', 'Fetch'],
            'show': ['Show', 'Display', 'Present', 'Exhibit', 'Reveal', 'Demonstrate'],
            'list': ['List', 'Enumerate', 'Show all', 'Display all', 'Catalog', 'Present all'],
            'calculate': ['Calculate', 'Compute', 'Determine', 'Find', 'Derive', 'Evaluate'],
            'analyze': ['Analyze', 'Examine', 'Review', 'Investigate', 'Study', 'Assess'],
            'rank': ['Rank', 'Order', 'Sort', 'Arrange', 'Organize', 'Sequence'],
            'compare': ['Compare', 'Contrast', 'Evaluate', 'Assess', 'Measure']
        }
    
    def _get_retail_domain(self) -> Dict:
        return {
            'description': 'Comprehensive retail data covering sales, inventory, customers, and product analytics.',
            'tables': {
                'products': {
                    'columns': {
                        'product_id': ('INT', list(range(1, 16))),
                        'name': ('VARCHAR(255)', ['Laptop', 'Phone', 'Tablet', 'Monitor', 'Keyboard', 'Mouse', 'Headphones', 'Webcam', 'Speaker', 'Router', 'Hard Drive', 'Memory Card', 'USB Hub', 'Printer', 'Scanner']),
                        'price': ('DECIMAL(10,2)', [999.99, 699.50, 299.00, 450.25, 89.99, 45.50, 129.99, 79.99, 199.00, 150.75, 89.50, 39.99, 55.00, 299.99, 349.50]),
                        'category': ('VARCHAR(100)', ['Electronics', 'Electronics', 'Electronics', 'Electronics', 'Accessories', 'Accessories', 'Audio', 'Video', 'Audio', 'Network', 'Storage', 'Storage', 'Accessories', 'Office', 'Office']),
                        'stock': ('INT', [50, 120, 80, 35, 200, 150, 90, 65, 110, 45, 75, 300, 125, 40, 55]),
                        'supplier': ('VARCHAR(100)', ['SupplierA', 'SupplierB', 'SupplierA', 'SupplierC', 'SupplierB', 'SupplierA', 'SupplierC', 'SupplierB', 'SupplierA', 'SupplierC', 'SupplierB', 'SupplierA', 'SupplierC', 'SupplierB', 'SupplierA']),
                        'created_date': ('DATE', ['2022-01-10', '2022-03-15', '2022-06-20', '2022-09-05', '2023-01-12', '2023-02-18', '2023-04-22', '2023-07-30', '2023-09-15', '2023-11-08', '2022-05-14', '2022-08-22', '2023-03-10', '2023-06-25', '2023-10-15'])
                    },
                    'primary_key': 'product_id',
                    'grouping_cols': ['category', 'supplier'],
                    'metric_cols': ['price', 'stock'],
                    'date_cols': ['created_date']
                },
                'customers': {
                    'columns': {
                        'customer_id': ('INT', list(range(1, 16))),
                        'name': ('VARCHAR(255)', ['John Smith', 'Jane Doe', 'Bob Johnson', 'Alice Williams', 'Charlie Brown', 'Diana Prince', 'Eve Davis', 'Frank Miller', 'Grace Lee', 'Henry Wilson', 'Iris Chen', 'Jack Taylor', 'Kate Anderson', 'Leo Martinez', 'Mia Robinson']),
                        'city': ('VARCHAR(100)', ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'Austin', 'Seattle', 'Denver', 'Boston', 'Portland', 'Miami']),
                        'total_purchases': ('INT', [5, 12, 3, 8, 15, 7, 20, 4, 11, 6, 9, 14, 2, 18, 10]),
                        'member_status': ('VARCHAR(50)', ['Gold', 'Silver', 'Bronze', 'Gold', 'Platinum', 'Silver', 'Platinum', 'Bronze', 'Gold', 'Silver', 'Gold', 'Platinum', 'Bronze', 'Platinum', 'Silver']),
                        'signup_date': ('DATE', ['2020-01-15', '2020-03-22', '2020-06-10', '2021-02-18', '2021-08-05', '2021-11-12', '2022-01-30', '2022-05-15', '2022-09-20', '2023-03-10', '2020-07-25', '2021-04-18', '2022-06-22', '2023-01-14', '2023-08-30'])
                    },
                    'primary_key': 'customer_id',
                    'grouping_cols': ['city', 'member_status'],
                    'metric_cols': ['total_purchases'],
                    'date_cols': ['signup_date']
                },
                'orders': {
                    'columns': {
                        'order_id': ('INT', list(range(1, 16))),
                        'customer_id': ('INT', [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
                        'product_id': ('INT', [1, 2, 1, 3, 2, 4, 5, 3, 1, 2, 6, 7, 8, 9, 10]),
                        'quantity': ('INT', [1, 2, 1, 3, 1, 2, 4, 1, 2, 3, 1, 2, 1, 3, 2]),
                        'status': ('VARCHAR(50)', ['Delivered', 'Delivered', 'Processing', 'Shipped', 'Delivered', 'Processing', 'Shipped', 'Delivered', 'Cancelled', 'Processing', 'Delivered', 'Shipped', 'Processing', 'Delivered', 'Shipped']),
                        'order_date': ('DATE', ['2023-01-15', '2023-02-20', '2023-03-10', '2023-03-25', '2023-04-12', '2023-05-18', '2023-06-22', '2023-07-14', '2023-08-09', '2023-09-30', '2023-10-15', '2023-11-05', '2023-12-01', '2024-01-10', '2024-02-14'])
                    },
                    'primary_key': 'order_id',
                    'grouping_cols': ['status'],
                    'metric_cols': ['quantity'],
                    'date_cols': ['order_date'],
                    'foreign_keys': {'customer_id': 'customers', 'product_id': 'products'}
                },
                'sales': {
                    'columns': {
                        'sale_id': ('INT', list(range(1, 16))),
                        'product_id': ('INT', [1, 2, 3, 1, 2, 4, 5, 3, 1, 6, 7, 8, 9, 10, 11]),
                        'amount': ('DECIMAL(10,2)', [999.99, 1399.00, 299.00, 999.99, 699.50, 900.50, 179.96, 897.00, 1999.98, 89.99, 259.97, 159.96, 55.00, 299.99, 349.50]),
                        'region': ('VARCHAR(100)', ['North', 'South', 'East', 'West', 'North', 'South', 'East', 'West', 'North', 'South', 'East', 'West', 'North', 'South', 'East']),
                        'sale_date': ('DATE', ['2023-01-10', '2023-01-15', '2023-02-20', '2023-03-05', '2023-03-15', '2023-04-10', '2023-05-22', '2023-06-18', '2023-07-25', '2023-08-30', '2023-09-12', '2023-10-20', '2023-11-08', '2023-12-15', '2024-01-22'])
                    },
                    'primary_key': 'sale_id',
                    'grouping_cols': ['region'],
                    'metric_cols': ['amount'],
                    'date_cols': ['sale_date'],
                    'foreign_keys': {'product_id': 'products'}
                }
            }
        }
    
    def _get_healthcare_domain(self) -> Dict:
        return {
            'description': 'Healthcare system data including patient care, treatments, appointments, and clinical outcomes.',
            'tables': {
                'patients': {
                    'columns': {
                        'patient_id': ('INT', list(range(1, 16))),
                        'name': ('VARCHAR(255)', ['Sarah Connor', 'Michael Scott', 'Emily Chen', 'David Martinez', 'Lisa Anderson', 'Robert Taylor', 'Jennifer Brown', 'William Garcia', 'Maria Rodriguez', 'James Wilson', 'Patricia Moore', 'Thomas Jackson', 'Linda White', 'Charles Harris', 'Barbara Martin']),
                        'age': ('INT', [45, 52, 28, 67, 34, 41, 55, 29, 63, 38, 48, 59, 32, 71, 44]),
                        'condition': ('VARCHAR(255)', ['Diabetes', 'Hypertension', 'Asthma', 'Arthritis', 'Healthy', 'Diabetes', 'Heart Disease', 'Asthma', 'Hypertension', 'Healthy', 'Diabetes', 'Cancer', 'Asthma', 'Arthritis', 'Heart Disease']),
                        'city': ('VARCHAR(100)', ['Boston', 'Seattle', 'Austin', 'Denver', 'Portland', 'Atlanta', 'Miami', 'Detroit', 'Nashville', 'Phoenix', 'Chicago', 'Houston', 'Dallas', 'San Francisco', 'New York']),
                        'insurance_type': ('VARCHAR(50)', ['PPO', 'HMO', 'PPO', 'Medicare', 'HMO', 'PPO', 'Medicare', 'HMO', 'Medicare', 'PPO', 'HMO', 'Medicare', 'PPO', 'Medicare', 'HMO']),
                        'admission_date': ('DATE', ['2023-01-10', '2023-02-15', '2023-03-20', '2023-04-05', '2023-05-12', '2023-06-18', '2023-07-22', '2023-08-14', '2023-09-09', '2023-10-30', '2023-11-15', '2023-12-05', '2024-01-12', '2024-02-08', '2024-03-14'])
                    },
                    'primary_key': 'patient_id',
                    'grouping_cols': ['condition', 'city', 'insurance_type'],
                    'metric_cols': ['age'],
                    'date_cols': ['admission_date']
                },
                'doctors': {
                    'columns': {
                        'doctor_id': ('INT', list(range(1, 16))),
                        'name': ('VARCHAR(255)', ['Dr. Smith', 'Dr. Johnson', 'Dr. Williams', 'Dr. Brown', 'Dr. Davis', 'Dr. Miller', 'Dr. Wilson', 'Dr. Moore', 'Dr. Taylor', 'Dr. Anderson', 'Dr. Thomas', 'Dr. Jackson', 'Dr. White', 'Dr. Harris', 'Dr. Martin']),
                        'specialty': ('VARCHAR(100)', ['Cardiology', 'Neurology', 'Pediatrics', 'Orthopedics', 'General', 'Surgery', 'Radiology', 'Psychiatry', 'Dermatology', 'Oncology', 'Cardiology', 'Neurology', 'Emergency', 'Anesthesiology', 'Pathology']),
                        'department': ('VARCHAR(100)', ['Cardiology', 'Neurology', 'Pediatrics', 'Orthopedics', 'General Medicine', 'Surgery', 'Radiology', 'Mental Health', 'Dermatology', 'Oncology', 'Cardiology', 'Neurology', 'Emergency', 'Surgery', 'Laboratory']),
                        'years_experience': ('INT', [15, 12, 8, 10, 5, 20, 7, 13, 9, 18, 22, 11, 6, 16, 14]),
                        'hire_date': ('DATE', ['2015-03-20', '2017-06-15', '2018-09-10', '2019-01-25', '2020-11-30', '2010-05-12', '2021-02-18', '2016-07-22', '2019-10-05', '2012-08-14', '2009-04-18', '2018-11-22', '2022-01-15', '2014-09-08', '2017-03-30'])
                    },
                    'primary_key': 'doctor_id',
                    'grouping_cols': ['specialty', 'department'],
                    'metric_cols': ['years_experience'],
                    'date_cols': ['hire_date']
                },
                'appointments': {
                    'columns': {
                        'appointment_id': ('INT', list(range(1, 16))),
                        'patient_id': ('INT', [1, 2, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
                        'doctor_id': ('INT', [1, 2, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
                        'duration_minutes': ('INT', [30, 45, 60, 30, 90, 45, 60, 30, 45, 60, 30, 90, 45, 60, 30]),
                        'appointment_date': ('DATE', ['2023-06-01', '2023-06-02', '2023-06-03', '2023-06-15', '2023-06-20', '2023-07-05', '2023-07-18', '2023-08-10', '2023-08-25', '2023-09-12', '2023-10-05', '2023-10-20', '2023-11-15', '2023-12-08', '2024-01-10']),
                        'status': ('VARCHAR(50)', ['Completed', 'Completed', 'Scheduled', 'Completed', 'Cancelled', 'Scheduled', 'Completed', 'Completed', 'Scheduled', 'Completed', 'Completed', 'Scheduled', 'Completed', 'Completed', 'Scheduled'])
                    },
                    'primary_key': 'appointment_id',
                    'grouping_cols': ['status'],
                    'metric_cols': ['duration_minutes'],
                    'date_cols': ['appointment_date'],
                    'foreign_keys': {'patient_id': 'patients', 'doctor_id': 'doctors'}
                },
                'treatments': {
                    'columns': {
                        'treatment_id': ('INT', list(range(1, 16))),
                        'patient_id': ('INT', list(range(1, 16))),
                        'treatment_type': ('VARCHAR(100)', ['Surgery', 'Medication', 'Therapy', 'Checkup', 'Surgery', 'Medication', 'Therapy', 'Checkup', 'Surgery', 'Medication', 'Physical Therapy', 'Radiation', 'Chemotherapy', 'Dialysis', 'Immunotherapy']),
                        'cost': ('DECIMAL(10,2)', [5000.00, 150.00, 200.00, 100.00, 7500.00, 175.00, 225.00, 125.00, 6000.00, 200.00, 300.00, 8000.00, 12000.00, 1500.00, 10000.00]),
                        'treatment_date': ('DATE', ['2023-01-15', '2023-02-20', '2023-03-25', '2023-04-10', '2023-05-18', '2023-06-22', '2023-07-15', '2023-08-20', '2023-09-25', '2023-10-30', '2023-11-12', '2023-12-05', '2024-01-15', '2024-02-10', '2024-03-08'])
                    },
                    'primary_key': 'treatment_id',
                    'grouping_cols': ['treatment_type'],
                    'metric_cols': ['cost'],
                    'date_cols': ['treatment_date'],
                    'foreign_keys': {'patient_id': 'patients'}
                }
            }
        }
    
    def _get_finance_domain(self) -> Dict:
        return {
            'description': 'Banking and financial services data including accounts, transactions, loans, and investments.',
            'tables': {
                'accounts': {
                    'columns': {
                        'account_id': ('INT', list(range(1, 16))),
                        'customer_id': ('INT', list(range(1, 16))),
                        'account_type': ('VARCHAR(50)', ['Checking', 'Savings', 'Credit', 'Checking', 'Savings', 'Credit', 'Checking', 'Savings', 'Investment', 'Credit', 'Checking', 'Money Market', 'Savings', 'Investment', 'Credit']),
                        'balance': ('DECIMAL(10,2)', [5000.00, 12000.00, 1500.00, 8000.00, 25000.00, 3000.00, 6500.00, 18000.00, 50000.00, 2000.00, 7500.00, 30000.00, 15000.00, 45000.00, 2500.00]),
                        'status': ('VARCHAR(50)', ['Active', 'Active', 'Active', 'Active', 'Active', 'Frozen', 'Active', 'Active', 'Active', 'Closed', 'Active', 'Active', 'Active', 'Active', 'Frozen']),
                        'open_date': ('DATE', ['2020-01-15', '2020-05-20', '2021-03-10', '2021-07-22', '2021-11-05', '2022-02-14', '2022-06-18', '2022-09-25', '2023-01-30', '2023-05-12', '2020-08-15', '2021-04-20', '2022-10-10', '2023-07-05', '2024-01-15'])
                    },
                    'primary_key': 'account_id',
                    'grouping_cols': ['account_type', 'status'],
                    'metric_cols': ['balance'],
                    'date_cols': ['open_date']
                },
                'transactions': {
                    'columns': {
                        'transaction_id': ('INT', list(range(1, 16))),
                        'account_id': ('INT', [1, 2, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
                        'amount': ('DECIMAL(10,2)', [250.00, 1500.00, 75.50, 500.00, 2000.00, 125.75, 800.00, 350.25, 1200.00, 450.50, 600.00, 2500.00, 175.00, 1800.00, 325.50]),
                        'transaction_type': ('VARCHAR(50)', ['Deposit', 'Withdrawal', 'Transfer', 'Deposit', 'Withdrawal', 'Transfer', 'Deposit', 'Withdrawal', 'Transfer', 'Deposit', 'Withdrawal', 'Transfer', 'Deposit', 'Withdrawal', 'Transfer']),
                        'transaction_date': ('DATE', ['2023-01-10', '2023-02-15', '2023-03-20', '2023-04-05', '2023-05-12', '2023-06-18', '2023-07-22', '2023-08-14', '2023-09-09', '2023-10-30', '2023-11-15', '2023-12-05', '2024-01-12', '2024-02-08', '2024-03-14'])
                    },
                    'primary_key': 'transaction_id',
                    'grouping_cols': ['transaction_type'],
                    'metric_cols': ['amount'],
                    'date_cols': ['transaction_date'],
                    'foreign_keys': {'account_id': 'accounts'}
                },
                'loans': {
                    'columns': {
                        'loan_id': ('INT', list(range(1, 16))),
                        'account_id': ('INT', list(range(1, 16))),
                        'loan_amount': ('DECIMAL(10,2)', [25000.00, 50000.00, 15000.00, 100000.00, 30000.00, 45000.00, 20000.00, 75000.00, 35000.00, 60000.00, 40000.00, 85000.00, 28000.00, 55000.00, 32000.00]),
                        'interest_rate': ('DECIMAL(5,2)', [3.5, 4.2, 5.0, 3.8, 4.5, 3.9, 5.2, 4.0, 4.8, 3.6, 4.3, 3.7, 4.9, 4.1, 4.6]),
                        'loan_type': ('VARCHAR(50)', ['Personal', 'Mortgage', 'Auto', 'Mortgage', 'Personal', 'Auto', 'Personal', 'Mortgage', 'Auto', 'Personal', 'Business', 'Mortgage', 'Auto', 'Personal', 'Business']),
                        'issue_date': ('DATE', ['2022-01-15', '2022-03-20', '2022-06-10', '2022-09-05', '2023-01-12', '2023-03-18', '2023-05-22', '2023-07-14', '2023-09-08', '2023-11-20', '2022-04-15', '2022-10-12', '2023-02-25', '2023-08-18', '2024-01-10'])
                    },
                    'primary_key': 'loan_id',
                    'grouping_cols': ['loan_type'],
                    'metric_cols': ['loan_amount', 'interest_rate'],
                    'date_cols': ['issue_date'],
                    'foreign_keys': {'account_id': 'accounts'}
                }
            }
        }
    
    def _get_education_domain(self) -> Dict:
        return {
            'description': 'Academic institution data covering students, courses, enrollments, and performance metrics.',
            'tables': {
                'students': {
                    'columns': {
                        'student_id': ('INT', list(range(1, 16))),
                        'name': ('VARCHAR(255)', ['Alex Johnson', 'Beth Williams', 'Chris Davis', 'Dana Miller', 'Emma Wilson', 'Frank Moore', 'Grace Taylor', 'Henry Anderson', 'Iris Thomas', 'Jack Jackson', 'Kelly White', 'Leo Harris', 'Maya Martin', 'Noah Thompson', 'Olivia Garcia']),
                        'major': ('VARCHAR(100)', ['Computer Science', 'Mathematics', 'Physics', 'Engineering', 'Biology', 'Chemistry', 'Economics', 'Psychology', 'History', 'English', 'Business', 'Art', 'Music', 'Philosophy', 'Political Science']),
                        'gpa': ('DECIMAL(3,2)', [3.8, 3.5, 3.9, 3.2, 3.7, 3.4, 3.6, 3.1, 3.3, 3.9, 3.5, 3.7, 3.2, 3.8, 3.4]),
                        'year': ('VARCHAR(20)', ['Junior', 'Sophomore', 'Senior', 'Freshman', 'Junior', 'Sophomore', 'Senior', 'Freshman', 'Junior', 'Senior', 'Sophomore', 'Junior', 'Freshman', 'Senior', 'Sophomore']),
                        'enrollment_date': ('DATE', ['2020-09-01', '2021-09-01', '2019-09-01', '2022-09-01', '2020-09-01', '2021-09-01', '2019-09-01', '2022-09-01', '2020-09-01', '2019-09-01', '2021-09-01', '2020-09-01', '2022-09-01', '2019-09-01', '2021-09-01'])
                    },
                    'primary_key': 'student_id',
                    'grouping_cols': ['major', 'year'],
                    'metric_cols': ['gpa'],
                    'date_cols': ['enrollment_date']
                },
                'courses': {
                    'columns': {
                        'course_id': ('INT', list(range(1, 16))),
                        'course_name': ('VARCHAR(255)', ['Intro to Programming', 'Calculus I', 'Physics 101', 'Data Structures', 'Linear Algebra', 'Organic Chemistry', 'Microeconomics', 'Statistics', 'World History', 'Creative Writing', 'Marketing', 'Drawing', 'Music Theory', 'Ethics', 'American Government']),
                        'credits': ('INT', [3, 4, 4, 3, 3, 4, 3, 3, 3, 3, 3, 3, 2, 3, 3]),
                        'department': ('VARCHAR(100)', ['Computer Science', 'Mathematics', 'Physics', 'Computer Science', 'Mathematics', 'Chemistry', 'Economics', 'Mathematics', 'History', 'English', 'Business', 'Art', 'Music', 'Philosophy', 'Political Science']),
                        'semester': ('VARCHAR(50)', ['Fall', 'Fall', 'Spring', 'Spring', 'Fall', 'Spring', 'Fall', 'Spring', 'Fall', 'Spring', 'Fall', 'Spring', 'Fall', 'Spring', 'Fall'])
                    },
                    'primary_key': 'course_id',
                    'grouping_cols': ['department', 'semester'],
                    'metric_cols': ['credits'],
                    'date_cols': []
                },
                'enrollments': {
                    'columns': {
                        'enrollment_id': ('INT', list(range(1, 16))),
                        'student_id': ('INT', [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
                        'course_id': ('INT', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
                        'grade': ('DECIMAL(3,2)', [3.7, 4.0, 3.3, 3.8, 3.5, 3.9, 3.2, 3.6, 3.4, 4.0, 3.5, 3.7, 3.1, 3.8, 3.3]),
                        'enrollment_date': ('DATE', ['2023-01-15', '2023-01-15', '2023-01-18', '2023-01-20', '2023-01-22', '2023-08-25', '2023-08-27', '2023-08-28', '2023-08-30', '2023-09-01', '2023-01-16', '2023-08-26', '2023-01-19', '2023-08-29', '2023-09-02'])
                    },
                    'primary_key': 'enrollment_id',
                    'grouping_cols': [],
                    'metric_cols': ['grade'],
                    'date_cols': ['enrollment_date'],
                    'foreign_keys': {'student_id': 'students', 'course_id': 'courses'}
                }
            }
        }
    
    def _get_logistics_domain(self) -> Dict:
        return {
            'description': 'Supply chain and logistics data including warehouses, shipments, routes, and delivery tracking.',
            'tables': {
                'warehouses': {
                    'columns': {
                        'warehouse_id': ('INT', list(range(1, 16))),
                        'location': ('VARCHAR(100)', ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'Austin', 'Seattle', 'Denver', 'Boston', 'Atlanta', 'Miami']),
                        'capacity': ('INT', [10000, 15000, 12000, 8000, 9000, 11000, 7000, 13000, 14000, 10500, 16000, 9500, 11500, 8500, 12500]),
                        'manager': ('VARCHAR(255)', ['John Doe', 'Jane Smith', 'Bob Wilson', 'Alice Brown', 'Charlie Davis', 'Diana Miller', 'Eve Taylor', 'Frank Moore', 'Grace Lee', 'Henry Clark', 'Iris Robinson', 'Jack Lewis', 'Kate Walker', 'Leo Hall', 'Maya Allen']),
                        'warehouse_type': ('VARCHAR(50)', ['Distribution', 'Storage', 'Distribution', 'Storage', 'Distribution', 'Storage', 'Distribution', 'Fulfillment', 'Distribution', 'Storage', 'Fulfillment', 'Distribution', 'Storage', 'Fulfillment', 'Distribution']),
                        'open_date': ('DATE', ['2018-01-15', '2018-05-20', '2019-03-10', '2019-07-22', '2020-02-14', '2020-06-18', '2021-01-25', '2021-09-12', '2022-03-30', '2022-11-08', '2019-08-15', '2020-10-20', '2021-05-12', '2022-07-18', '2023-02-25'])
                    },
                    'primary_key': 'warehouse_id',
                    'grouping_cols': ['location', 'warehouse_type'],
                    'metric_cols': ['capacity'],
                    'date_cols': ['open_date']
                },
                'shipments': {
                    'columns': {
                        'shipment_id': ('INT', list(range(1, 16))),
                        'warehouse_id': ('INT', [1, 2, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
                        'weight': ('DECIMAL(10,2)', [150.50, 200.75, 125.00, 180.25, 220.50, 95.75, 300.00, 175.50, 250.25, 140.00, 190.75, 210.50, 160.25, 230.00, 145.50]),
                        'status': ('VARCHAR(50)', ['Delivered', 'In Transit', 'Processing', 'Delivered', 'In Transit', 'Delivered', 'Processing', 'Delivered', 'In Transit', 'Delivered', 'Processing', 'In Transit', 'Delivered', 'Processing', 'Delivered']),
                        'priority': ('VARCHAR(50)', ['Standard', 'Express', 'Standard', 'Priority', 'Express', 'Standard', 'Priority', 'Standard', 'Express', 'Standard', 'Priority', 'Express', 'Standard', 'Priority', 'Express']),
                        'ship_date': ('DATE', ['2023-01-10', '2023-02-15', '2023-03-20', '2023-04-05', '2023-05-12', '2023-06-18', '2023-07-22', '2023-08-14', '2023-09-09', '2023-10-30', '2023-11-15', '2023-12-05', '2024-01-10', '2024-02-08', '2024-03-12'])
                    },
                    'primary_key': 'shipment_id',
                    'grouping_cols': ['status', 'priority'],
                    'metric_cols': ['weight'],
                    'date_cols': ['ship_date'],
                    'foreign_keys': {'warehouse_id': 'warehouses'}
                },
                'routes': {
                    'columns': {
                        'route_id': ('INT', list(range(1, 16))),
                        'origin': ('VARCHAR(100)', ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'Austin', 'Seattle', 'Denver', 'Boston', 'Atlanta', 'Miami']),
                        'destination': ('VARCHAR(100)', ['Boston', 'San Francisco', 'Detroit', 'Miami', 'Seattle', 'Atlanta', 'Denver', 'Portland', 'Nashville', 'Las Vegas', 'Los Angeles', 'Chicago', 'New York', 'Houston', 'Dallas']),
                        'distance': ('INT', [215, 380, 280, 1200, 1420, 400, 920, 1100, 240, 1100, 1100, 920, 215, 800, 1050]),
                        'route_type': ('VARCHAR(50)', ['Ground', 'Air', 'Ground', 'Air', 'Air', 'Ground', 'Ground', 'Air', 'Ground', 'Air', 'Air', 'Ground', 'Ground', 'Air', 'Ground']),
                        'active_date': ('DATE', ['2022-01-01', '2022-02-01', '2022-03-01', '2022-04-01', '2022-05-01', '2022-06-01', '2022-07-01', '2022-08-01', '2022-09-01', '2022-10-01', '2022-11-01', '2022-12-01', '2023-01-01', '2023-02-01', '2023-03-01'])
                    },
                    'primary_key': 'route_id',
                    'grouping_cols': ['origin', 'destination', 'route_type'],
                    'metric_cols': ['distance'],
                    'date_cols': ['active_date']
                }
            }
        }
    
    def _get_ecommerce_domain(self) -> Dict:
        return {
            'description': 'Online marketplace data including products, orders, reviews, and customer interactions.',
            'tables': {
                'products': {
                    'columns': {
                        'product_id': ('INT', list(range(1, 16))),
                        'name': ('VARCHAR(255)', ['Wireless Earbuds', 'Smart Watch', 'Fitness Tracker', 'Bluetooth Speaker', 'Phone Case', 'Charger', 'Screen Protector', 'Power Bank', 'USB Cable', 'Car Mount', 'Tablet Stand', 'Laptop Bag', 'Wireless Mouse', 'Keyboard Cover', 'Stylus Pen']),
                        'price': ('DECIMAL(10,2)', [79.99, 299.99, 149.99, 59.99, 19.99, 29.99, 9.99, 49.99, 14.99, 24.99, 34.99, 59.99, 39.99, 29.99, 44.99]),
                        'category': ('VARCHAR(100)', ['Audio', 'Electronics', 'Electronics', 'Audio', 'Accessories', 'Accessories', 'Accessories', 'Electronics', 'Accessories', 'Accessories', 'Accessories', 'Accessories', 'Electronics', 'Accessories', 'Electronics']),
                        'rating': ('DECIMAL(2,1)', [4.5, 4.7, 4.3, 4.6, 4.2, 4.4, 4.1, 4.8, 4.0, 4.3, 4.5, 4.6, 4.4, 4.2, 4.7]),
                        'seller': ('VARCHAR(100)', ['TechStore', 'GadgetHub', 'TechStore', 'AudioPro', 'CaseMaster', 'TechStore', 'ScreenGuard', 'PowerPlus', 'CableCo', 'AutoTech', 'TechStore', 'BagWorld', 'GadgetHub', 'CaseMaster', 'PenTech']),
                        'list_date': ('DATE', ['2022-06-15', '2022-07-20', '2022-08-10', '2022-09-05', '2022-10-12', '2023-01-18', '2023-02-22', '2023-03-14', '2023-04-08', '2023-05-20', '2022-11-15', '2023-06-10', '2023-07-25', '2023-08-18', '2023-09-30'])
                    },
                    'primary_key': 'product_id',
                    'grouping_cols': ['category', 'seller'],
                    'metric_cols': ['price', 'rating'],
                    'date_cols': ['list_date']
                },
                'orders': {
                    'columns': {
                        'order_id': ('INT', list(range(1, 16))),
                        'product_id': ('INT', [1, 2, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
                        'customer_id': ('INT', list(range(1, 16))),
                        'quantity': ('INT', [2, 1, 1, 3, 2, 1, 4, 1, 2, 1, 2, 1, 3, 2, 1]),
                        'total_amount': ('DECIMAL(10,2)', [159.98, 299.99, 149.99, 239.97, 39.98, 29.99, 39.96, 49.99, 29.98, 24.99, 69.98, 59.99, 119.97, 59.98, 44.99]),
                        'order_date': ('DATE', ['2023-06-01', '2023-06-05', '2023-06-10', '2023-06-15', '2023-06-20', '2023-06-25', '2023-07-01', '2023-07-05', '2023-07-10', '2023-07-15', '2023-07-20', '2023-07-25', '2023-08-01', '2023-08-05', '2023-08-10'])
                    },
                    'primary_key': 'order_id',
                    'grouping_cols': [],
                    'metric_cols': ['quantity', 'total_amount'],
                    'date_cols': ['order_date'],
                    'foreign_keys': {'product_id': 'products'}
                },
                'reviews': {
                    'columns': {
                        'review_id': ('INT', list(range(1, 16))),
                        'product_id': ('INT', list(range(1, 16))),
                        'rating': ('INT', [5, 4, 5, 4, 3, 5, 4, 5, 4, 4, 5, 4, 5, 3, 4]),
                        'verified_purchase': ('VARCHAR(10)', ['Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'No']),
                        'review_date': ('DATE', ['2023-07-01', '2023-07-05', '2023-07-10', '2023-07-15', '2023-07-20', '2023-07-25', '2023-08-01', '2023-08-05', '2023-08-10', '2023-08-15', '2023-08-20', '2023-08-25', '2023-09-01', '2023-09-05', '2023-09-10'])
                    },
                    'primary_key': 'review_id',
                    'grouping_cols': ['verified_purchase'],
                    'metric_cols': ['rating'],
                    'date_cols': ['review_date'],
                    'foreign_keys': {'product_id': 'products'}
                }
            }
        }
    
    def _get_manufacturing_domain(self) -> Dict:
        return {
            'description': 'Manufacturing operations data including production lines, quality control, inventory, and maintenance.',
            'tables': {
                'production_lines': {
                    'columns': {
                        'line_id': ('INT', list(range(1, 16))),
                        'line_name': ('VARCHAR(100)', ['Assembly A', 'Assembly B', 'Packaging 1', 'Packaging 2', 'Quality Check 1', 'Quality Check 2', 'Assembly C', 'Packaging 3', 'Testing 1', 'Assembly D', 'Packaging 4', 'Quality Check 3', 'Testing 2', 'Assembly E', 'Packaging 5']),
                        'capacity_per_hour': ('INT', [100, 120, 200, 180, 150, 160, 110, 190, 80, 105, 195, 155, 85, 115, 185]),
                        'status': ('VARCHAR(50)', ['Operating', 'Operating', 'Maintenance', 'Operating', 'Operating', 'Maintenance', 'Operating', 'Operating', 'Operating', 'Maintenance', 'Operating', 'Operating', 'Operating', 'Operating', 'Maintenance']),
                        'department': ('VARCHAR(100)', ['Assembly', 'Assembly', 'Packaging', 'Packaging', 'Quality', 'Quality', 'Assembly', 'Packaging', 'Testing', 'Assembly', 'Packaging', 'Quality', 'Testing', 'Assembly', 'Packaging']),
                        'install_date': ('DATE', ['2020-01-15', '2020-03-20', '2020-06-10', '2020-09-15', '2021-01-20', '2021-04-12', '2021-07-18', '2021-10-25', '2022-01-30', '2022-05-15', '2022-08-20', '2022-11-10', '2023-02-18', '2023-06-22', '2023-10-15'])
                    },
                    'primary_key': 'line_id',
                    'grouping_cols': ['department', 'status'],
                    'metric_cols': ['capacity_per_hour'],
                    'date_cols': ['install_date']
                },
                'products': {
                    'columns': {
                        'product_id': ('INT', list(range(1, 16))),
                        'product_name': ('VARCHAR(255)', ['Widget A', 'Widget B', 'Component X', 'Component Y', 'Assembly Unit 1', 'Assembly Unit 2', 'Module P', 'Module Q', 'Part Z', 'Subassembly 1', 'Subassembly 2', 'Final Product A', 'Final Product B', 'Component K', 'Module R']),
                        'production_cost': ('DECIMAL(10,2)', [15.50, 22.75, 8.25, 12.00, 45.50, 52.25, 18.75, 25.00, 6.50, 35.00, 42.50, 125.00, 150.00, 10.75, 28.50]),
                        'defect_rate': ('DECIMAL(3,2)', [0.02, 0.01, 0.03, 0.02, 0.01, 0.02, 0.01, 0.02, 0.04, 0.01, 0.02, 0.01, 0.02, 0.03, 0.01]),
                        'category': ('VARCHAR(100)', ['Electronics', 'Electronics', 'Mechanical', 'Mechanical', 'Electronics', 'Electronics', 'Electrical', 'Electrical', 'Mechanical', 'Electronics', 'Electronics', 'Final', 'Final', 'Mechanical', 'Electrical']),
                        'launch_date': ('DATE', ['2021-01-10', '2021-03-15', '2021-06-20', '2021-09-05', '2022-01-12', '2022-03-18', '2022-06-22', '2022-09-14', '2022-12-08', '2023-03-15', '2023-06-20', '2023-09-10', '2023-12-05', '2022-04-15', '2023-01-22'])
                    },
                    'primary_key': 'product_id',
                    'grouping_cols': ['category'],
                    'metric_cols': ['production_cost', 'defect_rate'],
                    'date_cols': ['launch_date']
                },
                'production_runs': {
                    'columns': {
                        'run_id': ('INT', list(range(1, 16))),
                        'line_id': ('INT', [1, 2, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
                        'product_id': ('INT', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
                        'units_produced': ('INT', [950, 1150, 1950, 980, 1480, 1560, 1080, 1870, 780, 1030, 1920, 1520, 830, 1130, 1810]),
                        'run_date': ('DATE', ['2023-01-15', '2023-02-20', '2023-03-25', '2023-04-10', '2023-05-18', '2023-06-22', '2023-07-15', '2023-08-20', '2023-09-25', '2023-10-30', '2023-11-15', '2023-12-05', '2024-01-12', '2024-02-08', '2024-03-14'])
                    },
                    'primary_key': 'run_id',
                    'grouping_cols': [],
                    'metric_cols': ['units_produced'],
                    'date_cols': ['run_date'],
                    'foreign_keys': {'line_id': 'production_lines', 'product_id': 'products'}
                }
            }
        }
    
    def _get_real_estate_domain(self) -> Dict:
        return {
            'description': 'Real estate market data including properties, transactions, agents, and market analytics.',
            'tables': {
                'properties': {
                    'columns': {
                        'property_id': ('INT', list(range(1, 16))),
                        'address': ('VARCHAR(255)', ['123 Main St', '456 Oak Ave', '789 Pine Rd', '321 Elm St', '654 Maple Dr', '987 Cedar Ln', '147 Birch Way', '258 Spruce Ct', '369 Willow Pl', '741 Ash Blvd', '852 Cherry St', '963 Poplar Ave', '159 Hickory Rd', '357 Walnut Dr', '486 Beech Ln']),
                        'city': ('VARCHAR(100)', ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'Austin', 'Seattle', 'Denver', 'Boston', 'Portland', 'Miami']),
                        'price': ('DECIMAL(12,2)', [450000.00, 850000.00, 325000.00, 275000.00, 395000.00, 525000.00, 285000.00, 725000.00, 350000.00, 425000.00, 675000.00, 485000.00, 625000.00, 575000.00, 495000.00]),
                        'bedrooms': ('INT', [3, 4, 2, 2, 3, 4, 2, 4, 3, 3, 4, 3, 4, 3, 3]),
                        'property_type': ('VARCHAR(50)', ['House', 'House', 'Condo', 'Apartment', 'House', 'House', 'Condo', 'House', 'Townhouse', 'House', 'House', 'Townhouse', 'House', 'Condo', 'House']),
                        'list_date': ('DATE', ['2023-01-15', '2023-02-20', '2023-03-10', '2023-04-05', '2023-05-12', '2023-06-18', '2023-07-22', '2023-08-14', '2023-09-09', '2023-10-30', '2023-11-15', '2023-12-05', '2024-01-10', '2024-02-08', '2024-03-12'])
                    },
                    'primary_key': 'property_id',
                    'grouping_cols': ['city', 'property_type'],
                    'metric_cols': ['price', 'bedrooms'],
                    'date_cols': ['list_date']
                },
                'agents': {
                    'columns': {
                        'agent_id': ('INT', list(range(1, 16))),
                        'name': ('VARCHAR(255)', ['Sarah Mitchell', 'John Peterson', 'Emily Roberts', 'Michael Chang', 'Lisa Thompson', 'David Kim', 'Jennifer Lopez', 'Robert Wilson', 'Maria Garcia', 'James Anderson', 'Patricia Taylor', 'Thomas Brown', 'Linda Davis', 'Charles Martinez', 'Barbara White']),
                        'agency': ('VARCHAR(100)', ['Prime Realty', 'Urban Homes', 'Prime Realty', 'Skyline Properties', 'Urban Homes', 'Prime Realty', 'Coastal Real Estate', 'Skyline Properties', 'Urban Homes', 'Prime Realty', 'Coastal Real Estate', 'Urban Homes', 'Skyline Properties', 'Prime Realty', 'Urban Homes']),
                        'commission_rate': ('DECIMAL(3,2)', [0.03, 0.025, 0.03, 0.028, 0.025, 0.03, 0.027, 0.028, 0.025, 0.03, 0.027, 0.025, 0.028, 0.03, 0.025]),
                        'years_experience': ('INT', [12, 8, 15, 6, 10, 18, 7, 14, 9, 20, 11, 13, 5, 16, 8]),
                        'hire_date': ('DATE', ['2015-03-15', '2018-06-20', '2012-09-10', '2020-01-25', '2017-11-30', '2010-05-12', '2019-02-18', '2014-07-22', '2018-10-05', '2008-08-14', '2016-04-18', '2015-11-22', '2021-01-15', '2013-09-08', '2018-03-30'])
                    },
                    'primary_key': 'agent_id',
                    'grouping_cols': ['agency'],
                    'metric_cols': ['commission_rate', 'years_experience'],
                    'date_cols': ['hire_date']
                },
                'transactions': {
                    'columns': {
                        'transaction_id': ('INT', list(range(1, 16))),
                        'property_id': ('INT', list(range(1, 16))),
                        'agent_id': ('INT', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
                        'sale_price': ('DECIMAL(12,2)', [445000.00, 840000.00, 320000.00, 270000.00, 390000.00, 520000.00, 280000.00, 720000.00, 345000.00, 420000.00, 670000.00, 480000.00, 620000.00, 570000.00, 490000.00]),
                        'sale_date': ('DATE', ['2023-03-15', '2023-04-20', '2023-05-10', '2023-06-05', '2023-07-12', '2023-08-18', '2023-09-22', '2023-10-14', '2023-11-09', '2024-01-05', '2024-02-15', '2024-03-05', '2024-04-10', '2024-05-08', '2024-06-12'])
                    },
                    'primary_key': 'transaction_id',
                    'grouping_cols': [],
                    'metric_cols': ['sale_price'],
                    'date_cols': ['sale_date'],
                    'foreign_keys': {'property_id': 'properties', 'agent_id': 'agents'}
                }
            }
        }
    
    def _get_hospitality_domain(self) -> Dict:
        return {
            'description': 'Hotel and hospitality industry data including reservations, rooms, guests, and service ratings.',
            'tables': {
                'hotels': {
                    'columns': {
                        'hotel_id': ('INT', list(range(1, 16))),
                        'name': ('VARCHAR(255)', ['Grand Plaza', 'Ocean View Resort', 'Mountain Lodge', 'City Center Inn', 'Sunset Hotel', 'Riverside Suite', 'Garden Retreat', 'Downtown Marriott', 'Beach Paradise', 'Hilltop Hotel', 'Lakeside Resort', 'Urban Stay', 'Country Manor', 'Skyline Tower', 'Harbor Hotel']),
                        'city': ('VARCHAR(100)', ['New York', 'Los Angeles', 'Denver', 'Chicago', 'San Diego', 'Austin', 'Portland', 'Boston', 'Miami', 'Seattle', 'Orlando', 'Houston', 'Nashville', 'San Francisco', 'Baltimore']),
                        'star_rating': ('INT', [5, 5, 4, 3, 4, 4, 3, 5, 5, 4, 4, 3, 4, 5, 4]),
                        'room_count': ('INT', [250, 350, 120, 80, 200, 150, 100, 300, 280, 140, 220, 90, 110, 320, 180]),
                        'open_date': ('DATE', ['2010-05-15', '2012-08-20', '2015-03-10', '2018-06-25', '2013-11-12', '2016-09-18', '2019-01-22', '2011-07-14', '2014-04-08', '2017-10-30', '2015-12-15', '2020-02-05', '2018-08-10', '2012-11-20', '2016-05-25'])
                    },
                    'primary_key': 'hotel_id',
                    'grouping_cols': ['city', 'star_rating'],
                    'metric_cols': ['room_count'],
                    'date_cols': ['open_date']
                },
                'rooms': {
                    'columns': {
                        'room_id': ('INT', list(range(1, 16))),
                        'hotel_id': ('INT', [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8]),
                        'room_number': ('VARCHAR(10)', ['101', '102', '201', '202', '301', '302', '101', '102', '201', '202', '301', '302', '101', '102', '201']),
                        'room_type': ('VARCHAR(50)', ['Deluxe', 'Suite', 'Standard', 'Deluxe', 'Standard', 'Suite', 'Standard', 'Deluxe', 'Suite', 'Standard', 'Deluxe', 'Suite', 'Standard', 'Deluxe', 'Suite']),
                        'price_per_night': ('DECIMAL(10,2)', [250.00, 450.00, 150.00, 200.00, 120.00, 300.00, 100.00, 180.00, 380.00, 130.00, 220.00, 400.00, 110.00, 190.00, 420.00]),
                        'last_renovation': ('DATE', ['2020-01-15', '2020-01-15', '2019-06-20', '2019-06-20', '2021-03-10', '2021-03-10', '2022-08-05', '2022-08-05', '2020-11-12', '2020-11-12', '2021-09-18', '2021-09-18', '2023-01-22', '2023-01-22', '2020-07-14'])
                    },
                    'primary_key': 'room_id',
                    'grouping_cols': ['room_type'],
                    'metric_cols': ['price_per_night'],
                    'date_cols': ['last_renovation'],
                    'foreign_keys': {'hotel_id': 'hotels'}
                },
                'reservations': {
                    'columns': {
                        'reservation_id': ('INT', list(range(1, 16))),
                        'room_id': ('INT', list(range(1, 16))),
                        'guest_name': ('VARCHAR(255)', ['Alice Johnson', 'Bob Smith', 'Carol White', 'David Brown', 'Emma Davis', 'Frank Wilson', 'Grace Miller', 'Henry Moore', 'Iris Taylor', 'Jack Anderson', 'Kelly Thomas', 'Leo Jackson', 'Maya Harris', 'Noah Martin', 'Olivia Thompson']),
                        'nights': ('INT', [3, 5, 2, 4, 1, 7, 3, 2, 6, 4, 3, 5, 2, 4, 3]),
                        'total_cost': ('DECIMAL(10,2)', [750.00, 2250.00, 300.00, 800.00, 120.00, 2100.00, 300.00, 360.00, 2280.00, 520.00, 660.00, 2000.00, 220.00, 760.00, 1260.00]),
                        'check_in_date': ('DATE', ['2023-06-01', '2023-06-05', '2023-06-10', '2023-06-15', '2023-06-20', '2023-06-25', '2023-07-01', '2023-07-05', '2023-07-10', '2023-07-15', '2023-07-20', '2023-07-25', '2023-08-01', '2023-08-05', '2023-08-10'])
                    },
                    'primary_key': 'reservation_id',
                    'grouping_cols': [],
                    'metric_cols': ['nights', 'total_cost'],
                    'date_cols': ['check_in_date'],
                    'foreign_keys': {'room_id': 'rooms'}
                }
            }
        }
    
    def _get_social_media_domain(self) -> Dict:
        return {
            'description': 'Social media platform data including users, posts, engagement metrics, and content analytics.',
            'tables': {
                'users': {
                    'columns': {
                        'user_id': ('INT', list(range(1, 16))),
                        'username': ('VARCHAR(100)', ['techguru', 'travelbug', 'foodie_life', 'fitness_pro', 'art_lover', 'music_fan', 'book_worm', 'gamer_101', 'photo_wizard', 'cook_master', 'sports_enthusiast', 'fashion_icon', 'pet_parent', 'nature_explorer', 'film_critic']),
                        'followers': ('INT', [15000, 8500, 22000, 12000, 6500, 18000, 4200, 25000, 11000, 7800, 9500, 16000, 5500, 13000, 10500]),
                        'following': ('INT', [450, 320, 580, 410, 290, 520, 230, 650, 380, 310, 420, 490, 270, 460, 390]),
                        'account_type': ('VARCHAR(50)', ['Creator', 'Business', 'Creator', 'Business', 'Personal', 'Creator', 'Personal', 'Creator', 'Business', 'Creator', 'Personal', 'Business', 'Personal', 'Creator', 'Personal']),
                        'join_date': ('DATE', ['2019-03-15', '2020-06-20', '2018-09-10', '2020-01-25', '2021-11-30', '2019-05-12', '2022-02-18', '2018-07-22', '2020-10-05', '2021-08-14', '2022-04-18', '2019-11-22', '2023-01-15', '2020-09-08', '2021-03-30'])
                    },
                    'primary_key': 'user_id',
                    'grouping_cols': ['account_type'],
                    'metric_cols': ['followers', 'following'],
                    'date_cols': ['join_date']
                },
                'posts': {
                    'columns': {
                        'post_id': ('INT', list(range(1, 16))),
                        'user_id': ('INT', list(range(1, 16))),
                        'likes': ('INT', [1500, 850, 2200, 1200, 650, 1800, 420, 2500, 1100, 780, 950, 1600, 550, 1300, 1050]),
                        'comments': ('INT', [45, 28, 68, 42, 21, 55, 15, 72, 38, 26, 32, 51, 19, 44, 35]),
                        'shares': ('INT', [120, 75, 180, 95, 40, 140, 30, 200, 85, 60, 70, 130, 45, 110, 90]),
                        'content_type': ('VARCHAR(50)', ['Image', 'Video', 'Image', 'Video', 'Image', 'Video', 'Image', 'Video', 'Image', 'Video', 'Image', 'Video', 'Image', 'Video', 'Image']),
                        'post_date': ('DATE', ['2023-06-01', '2023-06-05', '2023-06-10', '2023-06-15', '2023-06-20', '2023-06-25', '2023-07-01', '2023-07-05', '2023-07-10', '2023-07-15', '2023-07-20', '2023-07-25', '2023-08-01', '2023-08-05', '2023-08-10'])
                    },
                    'primary_key': 'post_id',
                    'grouping_cols': ['content_type'],
                    'metric_cols': ['likes', 'comments', 'shares'],
                    'date_cols': ['post_date'],
                    'foreign_keys': {'user_id': 'users'}
                },
                'campaigns': {
                    'columns': {
                        'campaign_id': ('INT', list(range(1, 16))),
                        'user_id': ('INT', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
                        'campaign_name': ('VARCHAR(255)', ['Summer Sale', 'New Product Launch', 'Holiday Special', 'Flash Deal', 'Brand Awareness', 'Influencer Collab', 'Seasonal Promo', 'Product Review', 'Contest Giveaway', 'Tutorial Series', 'Customer Stories', 'Behind Scenes', 'FAQ Session', 'Live Event', 'Product Demo']),
                        'budget': ('DECIMAL(10,2)', [5000.00, 8000.00, 3500.00, 2000.00, 10000.00, 6500.00, 4000.00, 1500.00, 7500.00, 3000.00, 5500.00, 4500.00, 2500.00, 9000.00, 6000.00]),
                        'impressions': ('INT', [150000, 220000, 95000, 65000, 280000, 175000, 120000, 55000, 210000, 85000, 145000, 125000, 75000, 250000, 165000]),
                        'start_date': ('DATE', ['2023-06-01', '2023-06-15', '2023-07-01', '2023-07-15', '2023-08-01', '2023-08-15', '2023-09-01', '2023-09-15', '2023-10-01', '2023-10-15', '2023-11-01', '2023-11-15', '2023-12-01', '2023-12-15', '2024-01-01'])
                    },
                    'primary_key': 'campaign_id',
                    'grouping_cols': [],
                    'metric_cols': ['budget', 'impressions'],
                    'date_cols': ['start_date'],
                    'foreign_keys': {'user_id': 'users'}
                }
            }
        }
    
    def _get_gaming_domain(self) -> Dict:
        return {
            'description': 'Gaming platform data including players, matches, achievements, and performance statistics.',
            'tables': {
                'players': {
                    'columns': {
                        'player_id': ('INT', list(range(1, 16))),
                        'username': ('VARCHAR(100)', ['DragonSlayer', 'ShadowNinja', 'IceQueen', 'FireStorm', 'NightHawk', 'StormBreaker', 'MysticMage', 'IronWarrior', 'SwiftArcher', 'DarkKnight', 'LightBringer', 'ThunderBolt', 'CrimsonBlade', 'FrostWolf', 'SilverFox']),
                        'level': ('INT', [45, 62, 38, 71, 52, 48, 67, 55, 43, 69, 51, 58, 40, 65, 47]),
                        'experience_points': ('INT', [45000, 78000, 32000, 95000, 58000, 51000, 82000, 61000, 40000, 88000, 56000, 70000, 38000, 80000, 49000]),
                        'rank': ('VARCHAR(50)', ['Gold', 'Platinum', 'Silver', 'Diamond', 'Gold', 'Gold', 'Platinum', 'Gold', 'Silver', 'Diamond', 'Gold', 'Platinum', 'Silver', 'Platinum', 'Gold']),
                        'region': ('VARCHAR(50)', ['NA', 'EU', 'NA', 'ASIA', 'EU', 'NA', 'EU', 'ASIA', 'NA', 'EU', 'ASIA', 'NA', 'EU', 'ASIA', 'NA']),
                        'join_date': ('DATE', ['2021-03-15', '2020-06-20', '2022-09-10', '2020-01-25', '2021-11-30', '2022-05-12', '2020-08-18', '2021-07-22', '2022-10-05', '2020-03-14', '2021-04-18', '2020-11-22', '2022-06-15', '2020-09-08', '2021-12-30'])
                    },
                    'primary_key': 'player_id',
                    'grouping_cols': ['rank', 'region'],
                    'metric_cols': ['level', 'experience_points'],
                    'date_cols': ['join_date']
                },
                'matches': {
                    'columns': {
                        'match_id': ('INT', list(range(1, 16))),
                        'player_id': ('INT', list(range(1, 16))),
                        'game_mode': ('VARCHAR(50)', ['Ranked', 'Casual', 'Ranked', 'Tournament', 'Casual', 'Ranked', 'Tournament', 'Casual', 'Ranked', 'Tournament', 'Casual', 'Ranked', 'Tournament', 'Casual', 'Ranked']),
                        'score': ('INT', [2500, 1800, 3200, 4100, 1500, 2800, 3900, 1700, 2900, 4300, 1600, 3100, 4500, 1900, 2700]),
                        'duration_minutes': ('INT', [35, 28, 42, 55, 25, 38, 52, 30, 40, 58, 27, 44, 60, 32, 37]),
                        'result': ('VARCHAR(20)', ['Win', 'Loss', 'Win', 'Win', 'Loss', 'Win', 'Loss', 'Loss', 'Win', 'Win', 'Loss', 'Win', 'Win', 'Loss', 'Win']),
                        'match_date': ('DATE', ['2023-08-01', '2023-08-05', '2023-08-10', '2023-08-15', '2023-08-20', '2023-08-25', '2023-09-01', '2023-09-05', '2023-09-10', '2023-09-15', '2023-09-20', '2023-09-25', '2023-10-01', '2023-10-05', '2023-10-10'])
                    },
                    'primary_key': 'match_id',
                    'grouping_cols': ['game_mode', 'result'],
                    'metric_cols': ['score', 'duration_minutes'],
                    'date_cols': ['match_date'],
                    'foreign_keys': {'player_id': 'players'}
                },
                'achievements': {
                    'columns': {
                        'achievement_id': ('INT', list(range(1, 16))),
                        'player_id': ('INT', list(range(1, 16))),
                        'achievement_name': ('VARCHAR(255)', ['First Blood', 'Triple Kill', 'Legendary', 'Untouchable', 'Dominating', 'Rampage', 'Godlike', 'Master Strategist', 'Perfect Game', 'Comeback King', 'Speed Demon', 'Sharpshooter', 'Tank Buster', 'Support Hero', 'Solo Carry']),
                        'rarity': ('VARCHAR(50)', ['Common', 'Rare', 'Epic', 'Legendary', 'Rare', 'Epic', 'Legendary', 'Epic', 'Legendary', 'Rare', 'Common', 'Rare', 'Epic', 'Common', 'Legendary']),
                        'points': ('INT', [100, 250, 500, 1000, 250, 500, 1000, 500, 1000, 250, 100, 250, 500, 100, 1000]),
                        'unlock_date': ('DATE', ['2023-07-15', '2023-08-05', '2023-08-20', '2023-09-10', '2023-09-15', '2023-09-25', '2023-10-05', '2023-10-15', '2023-10-25', '2023-11-01', '2023-11-10', '2023-11-20', '2023-12-01', '2023-12-10', '2023-12-20'])
                    },
                    'primary_key': 'achievement_id',
                    'grouping_cols': ['rarity'],
                    'metric_cols': ['points'],
                    'date_cols': ['unlock_date'],
                    'foreign_keys': {'player_id': 'players'}
                }
            }
        }
    
    def _get_cte_templates(self) -> List[Dict]:
        """7 diverse CTE templates"""
        return [
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
            }
        ]
    
    def _get_set_operation_templates(self) -> List[Dict]:
        """6 diverse set operation templates"""
        return [
            {
                'sql': """SELECT {group_col1} as value FROM {table1} WHERE {group_col1} IS NOT NULL
UNION
SELECT {group_col2} as value FROM {table2} WHERE {group_col2} IS NOT NULL;""",
                'prompt_templates': [
                    "{verb} all unique values from {group_col1} in {table1} and {group_col2} in {table2}",
                    "{verb} combined distinct values of {group_col1} and {group_col2} from both tables",
                    "{verb} union of {group_col1} from {table1} with {group_col2} from {table2}",
                    "{verb} merged unique values from {table1}.{group_col1} and {table2}.{group_col2}",
                    "{verb} consolidated list of {group_col1} and {group_col2} across tables"
                ],
                'explanation': "Combines unique values from {group_col1} in {table1} and {group_col2} in {table2} using UNION, which automatically removes duplicates",
                'tables_needed': ['{table1}', '{table2}'],
                'requires': ['grouping_col'],
                'verb_style': 'find'
            },
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
            }
        ]
    
    def _get_multiple_join_templates(self) -> List[Dict]:
        """5 diverse multiple join templates"""
        return [
            {
                'sql': """SELECT t1.{group_col1}, t2.{group_col2}, SUM(t1.{metric_col}) as total_{metric_col}
FROM {table1} t1
INNER JOIN {table2} t2 ON t1.{fk1} = t2.{pk2}
WHERE t1.{date_col} >= '{year}-01-01'
GROUP BY t1.{group_col1}, t2.{group_col2}
ORDER BY total_{metric_col} DESC
LIMIT {top_n};""",
                'prompt_templates': [
                    "{verb} top {top_n} combinations of {group_col1} and {group_col2} by total {metric_col} since {year}",
                    "{verb} highest {top_n} {group_col1}-{group_col2} pairs by {metric_col} for {year}",
                    "{verb} leading {top_n} {group_col1} and {group_col2} combinations based on {metric_col}",
                    "{verb} best performing {top_n} {group_col1}-{group_col2} groups by {metric_col}",
                    "{verb} {top_n} strongest {group_col1} and {group_col2} pairings in {metric_col}"
                ],
                'explanation': "Joins {table1} and {table2}, aggregates {metric_col} by combinations of {group_col1} and {group_col2}, filters for {year} onwards, and returns top {top_n} results",
                'tables_needed': ['{table1}', '{table2}'],
                'requires': ['foreign_key_relationship', 'metric_col', 'grouping_col', 'date_col'],
                'verb_style': 'show'
            },
            {
                'sql': """SELECT t1.{group_col1}, t2.{group_col2}, AVG(t1.{metric_col}) as avg_{metric_col}, COUNT(*) as record_count
FROM {table1} t1
LEFT JOIN {table2} t2 ON t1.{fk1} = t2.{pk2}
WHERE t2.{pk2} IS NOT NULL
GROUP BY t1.{group_col1}, t2.{group_col2}
HAVING COUNT(*) >= 3
ORDER BY avg_{metric_col} DESC;""",
                'prompt_templates': [
                    "{verb} {group_col1} and {group_col2} pairs with average {metric_col} having at least 3 records",
                    "{verb} combinations of {group_col1}-{group_col2} where {metric_col} average is calculated from 3+ entries",
                    "{verb} grouped {group_col1} and {group_col2} with mean {metric_col} from sufficient data",
                    "{verb} {group_col1}-{group_col2} statistics filtered by minimum record count",
                    "{verb} {group_col1} and {group_col2} averages with meaningful sample sizes"
                ],
                'explanation': "Left joins {table1} with {table2}, calculates average {metric_col} for each {group_col1}-{group_col2} combination, filters to show only groups with 3 or more records",
                'tables_needed': ['{table1}', '{table2}'],
                'requires': ['foreign_key_relationship', 'metric_col', 'grouping_col'],
                'verb_style': 'analyze'
            },
            {
                'sql': """SELECT t1.{group_col1}, COUNT(DISTINCT t2.{pk2}) as related_count, MAX(t1.{metric_col}) as max_{metric_col}
FROM {table1} t1
INNER JOIN {table2} t2 ON t1.{fk1} = t2.{pk2}
WHERE t1.{date_col} >= '{year}-01-01'
GROUP BY t1.{group_col1}
HAVING MAX(t1.{metric_col}) > {threshold};""",
                'prompt_templates': [
                    "{verb} {group_col1} with maximum {metric_col} above {threshold} and their relationship counts",
                    "{verb} {group_col1} where max {metric_col} exceeds {threshold}, including distinct {table2} associations",
                    "{verb} high-performing {group_col1} based on {metric_col} threshold with relationship metrics",
                    "{verb} {group_col1} surpassing {threshold} in {metric_col} with related {table2} counts",
                    "{verb} top {group_col1} by {metric_col} threshold showing {table2} connection frequency"
                ],
                'explanation': "Joins tables to count distinct related records from {table2} for each {group_col1}, finds maximum {metric_col}, filters groups where max exceeds {threshold} for records since {year}",
                'tables_needed': ['{table1}', '{table2}'],
                'requires': ['foreign_key_relationship', 'metric_col', 'grouping_col', 'date_col'],
                'verb_style': 'find'
            },
            {
                'sql': """SELECT t1.{group_col1}, t2.{group_col2}, 
       SUM(t1.{metric_col}) as total, 
       ROUND(SUM(t1.{metric_col}) * 100.0 / SUM(SUM(t1.{metric_col})) OVER (), 2) as percentage
FROM {table1} t1
INNER JOIN {table2} t2 ON t1.{fk1} = t2.{pk2}
WHERE t1.{date_col} >= '{year}-01-01'
GROUP BY t1.{group_col1}, t2.{group_col2}
HAVING SUM(t1.{metric_col}) > {threshold}
ORDER BY percentage DESC;""",
                'prompt_templates': [
                    "{verb} {group_col1}-{group_col2} distribution by {metric_col} with percentages for {year}",
                    "{verb} relative contribution of each {group_col1}-{group_col2} pair to total {metric_col}",
                    "{verb} {group_col1} and {group_col2} breakdown with percentage shares exceeding threshold",
                    "{verb} proportional {metric_col} analysis by {group_col1} and {group_col2}",
                    "{verb} {group_col1}-{group_col2} composition showing {metric_col} percentages"
                ],
                'explanation': "Joins {table1} and {table2}, calculates both absolute totals and percentage contribution of each {group_col1}-{group_col2} combination to overall {metric_col}, filters for {year} and values above {threshold}",
                'tables_needed': ['{table1}', '{table2}'],
                'requires': ['foreign_key_relationship', 'metric_col', 'grouping_col', 'date_col'],
                'verb_style': 'calculate'
            },
            {
                'sql': """SELECT t1.{group_col1}, t2.{group_col2},
       AVG(t1.{metric_col}) as avg_{metric_col},
       MIN(t1.{metric_col}) as min_{metric_col},
       MAX(t1.{metric_col}) as max_{metric_col},
       STDDEV(t1.{metric_col}) as stddev_{metric_col}
FROM {table1} t1
RIGHT JOIN {table2} t2 ON t1.{fk1} = t2.{pk2}
WHERE t1.{date_col} IS NOT NULL
GROUP BY t1.{group_col1}, t2.{group_col2}
ORDER BY avg_{metric_col} DESC;""",
                'prompt_templates': [
                    "{verb} statistical summary of {metric_col} by {group_col1} and {group_col2}",
                    "{verb} comprehensive {metric_col} metrics across {group_col1}-{group_col2} combinations",
                    "{verb} complete {metric_col} statistics grouped by {group_col1} and {group_col2}",
                    "{verb} {group_col1} and {group_col2} with full {metric_col} distribution metrics",
                    "{verb} detailed {metric_col} analysis by {group_col1}-{group_col2} pairs"
                ],
                'explanation': "Right joins {table1} and {table2} to calculate comprehensive statistics (mean, minimum, maximum, standard deviation) of {metric_col} for each {group_col1}-{group_col2} combination, including all {table2} records",
                'tables_needed': ['{table1}', '{table2}'],
                'requires': ['foreign_key_relationship', 'metric_col', 'grouping_col', 'date_col'],
                'verb_style': 'summarize'
            }
        ]
    
    def _get_window_function_templates(self) -> List[Dict]:
        """8 diverse window function templates"""
        return [
            {
                'sql': """SELECT {group_col}, {metric_col},
    RANK() OVER (PARTITION BY {partition_col} ORDER BY {metric_col} DESC) as rank,
    AVG({metric_col}) OVER (PARTITION BY {partition_col}) as avg_{metric_col}
FROM {table1}
WHERE {date_col} >= '{year}-01-01'
HAVING rank <= {top_n};""",
                'prompt_templates': [
                    "{verb} top {top_n} records by {metric_col} within each {partition_col} since {year}",
                    "{verb} highest {top_n} {group_col} per {partition_col} showing {metric_col}",
                    "{verb} ranked {group_col} by {metric_col} in each {partition_col}, limited to top {top_n}",
                    "{verb} leading {top_n} entries per {partition_col} based on {metric_col}",
                    "{verb} best {top_n} performers by {metric_col} within {partition_col} groups"
                ],
                'explanation': "Uses RANK() to rank records by {metric_col} within each {partition_col}, calculates partition-level average {metric_col}, filters for {year} and shows top {top_n} per partition",
                'tables_needed': ['{table1}'],
                'requires': ['grouping_col', 'metric_col', 'date_col'],
                'verb_style': 'rank'
            },
            {
                'sql': """SELECT {group_col}, {date_col}, {metric_col},
    LAG({metric_col}, 1) OVER (PARTITION BY {group_col} ORDER BY {date_col}) as prev_{metric_col},
    LEAD({metric_col}, 1) OVER (PARTITION BY {group_col} ORDER BY {date_col}) as next_{metric_col},
    {metric_col} - LAG({metric_col}, 1) OVER (PARTITION BY {group_col} ORDER BY {date_col}) as change
FROM {table1}
WHERE {date_col} >= '{year}-01-01'
ORDER BY {group_col}, {date_col};""",
                'prompt_templates': [
                    "{verb} {metric_col} trends for each {group_col} with previous and next period values",
                    "{verb} period-over-period {metric_col} changes by {group_col} with adjacent comparisons",
                    "{verb} time-series analysis of {metric_col} per {group_col} including lag and lead",
                    "{verb} sequential {metric_col} progression for {group_col} showing transitions",
                    "{verb} temporal {metric_col} patterns by {group_col} with period comparisons"
                ],
                'explanation': "Uses LAG() and LEAD() window functions to access previous and next period {metric_col} values for each {group_col}, calculates period-over-period change, orders chronologically for records since {year}",
                'tables_needed': ['{table1}'],
                'requires': ['grouping_col', 'metric_col', 'date_col'],
                'verb_style': 'analyze'
            },
            {
                'sql': """SELECT {group_col}, {metric_col},
    ROW_NUMBER() OVER (ORDER BY {metric_col} DESC) as overall_rank,
    PERCENT_RANK() OVER (PARTITION BY {partition_col} ORDER BY {metric_col}) as percentile,
    CUME_DIST() OVER (PARTITION BY {partition_col} ORDER BY {metric_col}) as cumulative_dist
FROM {table1}
WHERE {date_col} >= '{year}-01-01';""",
                'prompt_templates': [
                    "{verb} {group_col} with overall ranking and percentile within {partition_col} for {year}",
                    "{verb} {group_col} by {metric_col} showing absolute rank and relative percentile",
                    "{verb} ranked {group_col} with percentile distribution per {partition_col}",
                    "{verb} {group_col} with comprehensive ranking metrics across {partition_col}",
                    "{verb} {group_col} performance showing global rank and local percentiles"
                ],
                'explanation': "Calculates overall ROW_NUMBER rank by {metric_col}, PERCENT_RANK percentile within each {partition_col}, and CUME_DIST cumulative distribution, filtered for records from {year}",
                'tables_needed': ['{table1}'],
                'requires': ['grouping_col', 'metric_col', 'date_col'],
                'verb_style': 'rank'
            },
            {
                'sql': """SELECT {group_col}, {metric_col}, {date_col},
    SUM({metric_col}) OVER (PARTITION BY {partition_col} ORDER BY {date_col} ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as running_total,
    AVG({metric_col}) OVER (PARTITION BY {partition_col} ORDER BY {date_col} ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) as moving_avg_3
FROM {table1}
WHERE {date_col} >= '{year}-01-01'
ORDER BY {partition_col}, {date_col};""",
                'prompt_templates': [
                    "{verb} running total and 3-period moving average of {metric_col} by {partition_col}",
                    "{verb} cumulative {metric_col} and rolling average for each {partition_col}",
                    "{verb} {group_col} with running sum and moving average of {metric_col}",
                    "{verb} progressive {metric_col} totals and trends per {partition_col}",
                    "{verb} cumulative and smoothed {metric_col} metrics by {partition_col}"
                ],
                'explanation': "Calculates running total of {metric_col} from start to current row, and 3-period moving average (current + 2 preceding rows) within each {partition_col}, ordered chronologically for records since {year}",
                'tables_needed': ['{table1}'],
                'requires': ['grouping_col', 'metric_col', 'date_col'],
                'verb_style': 'calculate'
            },
            {
                'sql': """SELECT {group_col}, {partition_col}, {metric_col},
    NTILE(4) OVER (PARTITION BY {partition_col} ORDER BY {metric_col}) as quartile,
    DENSE_RANK() OVER (PARTITION BY {partition_col} ORDER BY {metric_col} DESC) as dense_rank,
    RANK() OVER (PARTITION BY {partition_col} ORDER BY {metric_col} DESC) as rank
FROM {table1}
WHERE {date_col} >= '{year}-01-01';""",
                'prompt_templates': [
                    "{verb} {group_col} divided into quartiles by {metric_col} within each {partition_col}",
                    "{verb} quartile distribution and ranking of {group_col} per {partition_col}",
                    "{verb} {group_col} categorized into 4 groups based on {metric_col}",
                    "{verb} {group_col} with quartile assignment and multiple ranking schemes",
                    "{verb} {group_col} segmented by {metric_col} quartiles within {partition_col}"
                ],
                'explanation': "Uses NTILE(4) to divide records into quartiles based on {metric_col} within each {partition_col}, also provides DENSE_RANK and RANK for comprehensive ranking, filtered for records since {year}",
                'tables_needed': ['{table1}'],
                'requires': ['grouping_col', 'metric_col', 'date_col'],
                'verb_style': 'categorize'
            },
            {
                'sql': """SELECT {group_col}, {date_col}, {metric_col},
    FIRST_VALUE({metric_col}) OVER (PARTITION BY {partition_col} ORDER BY {date_col}) as period_start_value,
    LAST_VALUE({metric_col}) OVER (PARTITION BY {partition_col} ORDER BY {date_col} ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as period_end_value,
    {metric_col} - FIRST_VALUE({metric_col}) OVER (PARTITION BY {partition_col} ORDER BY {date_col}) as change_from_start
FROM {table1}
WHERE {date_col} >= '{year}-01-01';""",
                'prompt_templates': [
                    "{verb} {group_col} showing {metric_col} changes from period start",
                    "{verb} {metric_col} evolution per {group_col} relative to initial values",
                    "{verb} {group_col} with first-to-current {metric_col} comparisons",
                    "{verb} {metric_col} progression tracking for {group_col} from baseline",
                    "{verb} {group_col} performance versus period starting {metric_col}"
                ],
                'explanation': "Uses FIRST_VALUE and LAST_VALUE to capture period start and end {metric_col} values for each {partition_col}, calculates change from start for every record, filtered for {year} onwards",
                'tables_needed': ['{table1}'],
                'requires': ['grouping_col', 'metric_col', 'date_col'],
                'verb_style': 'track'
            },
            {
                'sql': """SELECT {group_col}, {partition_col}, {metric_col}, {date_col},
    {metric_col} - AVG({metric_col}) OVER (PARTITION BY {partition_col}) as deviation_from_avg,
    CASE 
        WHEN {metric_col} > AVG({metric_col}) OVER (PARTITION BY {partition_col}) + STDDEV({metric_col}) OVER (PARTITION BY {partition_col}) THEN 'Above Avg'
        WHEN {metric_col} < AVG({metric_col}) OVER (PARTITION BY {partition_col}) - STDDEV({metric_col}) OVER (PARTITION BY {partition_col}) THEN 'Below Avg'
        ELSE 'Within Avg'
    END as performance_category
FROM {table1}
WHERE {date_col} >= '{year}-01-01';""",
                'prompt_templates': [
                    "{verb} {group_col} categorized by {metric_col} deviation within {partition_col}",
                    "{verb} performance classification of {group_col} relative to {partition_col} average",
                    "{verb} {group_col} with statistical categorization by {metric_col}",
                    "{verb} {group_col} segmented into performance tiers within {partition_col}",
                    "{verb} {group_col} classified by {metric_col} relative to group statistics"
                ],
                'explanation': "Calculates deviation of each record's {metric_col} from its partition average, categorizes records as above/below/within average based on standard deviation thresholds, for records since {year}",
                'tables_needed': ['{table1}'],
                'requires': ['grouping_col', 'metric_col', 'date_col'],
                'verb_style': 'classify'
            },
            {
                'sql': """SELECT {group_col}, {date_col}, {metric_col},
    SUM({metric_col}) OVER (PARTITION BY {partition_col} ORDER BY {date_col}) / 
    SUM({metric_col}) OVER (PARTITION BY {partition_col}) * 100 as pct_of_total_to_date,
    ROW_NUMBER() OVER (PARTITION BY {partition_col} ORDER BY {date_col}) as period_number
FROM {table1}
WHERE {date_col} >= '{year}-01-01';""",
                'prompt_templates': [
                    "{verb} cumulative percentage of total {metric_col} by {group_col}",
                    "{verb} running {metric_col} contribution as percentage per {group_col}",
                    "{verb} progressive {metric_col} share analysis by {partition_col}",
                    "{verb} {group_col} showing cumulative {metric_col} as portion of total",
                    "{verb} period-by-period {metric_col} accumulation percentage for {group_col}"
                ],
                'explanation': "Calculates running sum of {metric_col} as percentage of partition total, assigns period numbers, showing how much of total {metric_col} has accumulated by each point in time within {partition_col}, for records since {year}",
                'tables_needed': ['{table1}'],
                'requires': ['grouping_col', 'metric_col', 'date_col'],
                'verb_style': 'monitor'
            }
        ]
    
    def _get_subquery_templates(self) -> List[Dict]:
        """7 diverse subquery templates"""
        return [
            {
                'sql': """SELECT {group_col}, {metric_col}
FROM {table1}
WHERE {metric_col} > (SELECT AVG({metric_col}) FROM {table1})
AND {date_col} >= '{year}-01-01'
ORDER BY {metric_col} DESC;""",
                'prompt_templates': [
                    "{verb} {group_col} where {metric_col} exceeds the overall average since {year}",
                    "{verb} above-average {group_col} by {metric_col} for {year}",
                    "{verb} {group_col} with {metric_col} higher than mean value",
                    "{verb} {group_col} surpassing average {metric_col} in {year}",
                    "{verb} {group_col} outperforming mean {metric_col} since {year}"
                ],
                'explanation': "Uses subquery to calculate overall average {metric_col}, filters main query to show only {group_col} with above-average values from {year} onwards",
                'tables_needed': ['{table1}'],
                'requires': ['grouping_col', 'metric_col', 'date_col'],
                'verb_style': 'find'
            },
            {
                'sql': """SELECT t1.{group_col}, t1.{metric_col},
    (SELECT COUNT(*) FROM {table2} t2 WHERE t2.{fk1} = t1.{pk1}) as related_count,
    (SELECT AVG({metric_col2}) FROM {table2} t2 WHERE t2.{fk1} = t1.{pk1}) as avg_related_{metric_col2}
FROM {table1} t1
WHERE t1.{date_col} >= '{year}-01-01'
AND (SELECT COUNT(*) FROM {table2} t2 WHERE t2.{fk1} = t1.{pk1}) > 0;""",
                'prompt_templates': [
                    "{verb} {group_col} with counts and averages from related {table2} records since {year}",
                    "{verb} {group_col} including relationship statistics from {table2}",
                    "{verb} {table1} {group_col} with aggregated metrics from associated {table2} entries",
                    "{verb} {group_col} showing {table2} relationships and average values",
                    "{verb} {group_col} with correlated {table2} counts and means"
                ],
                'explanation': "Uses correlated subqueries to count related records in {table2} and calculate their average {metric_col2} for each record in {table1}, filters for {year} and non-zero relationships",
                'tables_needed': ['{table1}', '{table2}'],
                'requires': ['foreign_key_relationship', 'grouping_col', 'metric_col', 'date_col'],
                'verb_style': 'show'
            },
            {
                'sql': """SELECT {group_col}, {metric_col}, {date_col}
FROM {table1} t1
WHERE {metric_col} >= (
    SELECT {metric_col}
    FROM {table1} t2
    WHERE t2.{group_col} = t1.{group_col}
    ORDER BY {date_col} DESC
    LIMIT 1
)
AND {date_col} >= '{year}-01-01'
ORDER BY {metric_col} DESC;""",
                'prompt_templates': [
                    "{verb} {group_col} where current {metric_col} matches or exceeds most recent value",
                    "{verb} {group_col} with {metric_col} at or above their latest recorded level",
                    "{verb} {group_col} maintaining or improving {metric_col} versus most recent",
                    "{verb} {group_col} with {metric_col} meeting or surpassing latest observation",
                    "{verb} {group_col} where {metric_col} equals or tops most recent entry"
                ],
                'explanation': "Correlated subquery finds most recent {metric_col} value for each {group_col}, main query filters to show only records where {metric_col} meets or exceeds this recent value, for records since {year}",
                'tables_needed': ['{table1}'],
                'requires': ['grouping_col', 'metric_col', 'date_col'],
                'verb_style': 'identify'
            },
            {
                'sql': """SELECT {group_col}, COUNT(*) as count, AVG({metric_col}) as avg_{metric_col}
FROM {table1}
WHERE {group_col} IN (
    SELECT {group_col}
    FROM {table1}
    WHERE {date_col} >= '{year}-01-01'
    GROUP BY {group_col}
    HAVING SUM({metric_col}) > {threshold}
)
AND {date_col} >= '{year}-01-01'
GROUP BY {group_col}
ORDER BY avg_{metric_col} DESC;""",
                'prompt_templates': [
                    "{verb} {group_col} statistics where total {metric_col} exceeds {threshold} for {year}",
                    "{verb} counts and averages for {group_col} with cumulative {metric_col} above {threshold}",
                    "{verb} aggregated {group_col} data filtered by total {metric_col} threshold",
                    "{verb} {group_col} metrics for groups surpassing {threshold} in {metric_col}",
                    "{verb} statistical breakdown of high-performing {group_col} by {metric_col}"
                ],
                'explanation': "Subquery identifies {group_col} values where sum of {metric_col} exceeds {threshold} in {year}, main query calculates count and average for those groups",
                'tables_needed': ['{table1}'],
                'requires': ['grouping_col', 'metric_col', 'date_col'],
                'verb_style': 'analyze'
            },
            {
                'sql': """SELECT {group_col}, {metric_col}, {date_col}
FROM {table1} t1
WHERE {metric_col} = (
    SELECT MAX({metric_col})
    FROM {table1} t2
    WHERE t2.{group_col} = t1.{group_col}
    AND t2.{date_col} >= '{year}-01-01'
)
AND {date_col} >= '{year}-01-01'
ORDER BY {metric_col} DESC;""",
                'prompt_templates': [
                    "{verb} records with maximum {metric_col} for each {group_col} in {year}",
                    "{verb} highest {metric_col} entry per {group_col} during {year}",
                    "{verb} peak {metric_col} values across all {group_col} for {year}",
                    "{verb} {group_col} showing their maximum {metric_col} achievement in {year}",
                    "{verb} top {metric_col} records by {group_col} for {year}"
                ],
                'explanation': "Correlated subquery finds maximum {metric_col} for each {group_col} in {year}, main query returns only records matching these maximum values",
                'tables_needed': ['{table1}'],
                'requires': ['grouping_col', 'metric_col', 'date_col'],
                'verb_style': 'find'
            },
            {
                'sql': """SELECT {group_col}, {metric_col}
FROM {table1} t1
WHERE {metric_col} > ALL (
    SELECT {metric_col}
    FROM {table1} t2
    WHERE t2.{group_col} = t1.{group_col}
    AND t2.{date_col} < t1.{date_col}
    AND t2.{date_col} >= DATE_SUB(t1.{date_col}, INTERVAL 90 DAY)
)
AND {date_col} >= '{year}-01-01'
ORDER BY {date_col}, {metric_col} DESC;""",
                'prompt_templates': [
                    "{verb} {group_col} where {metric_col} exceeds all values in previous 90 days",
                    "{verb} {group_col} reaching new 90-day highs in {metric_col}",
                    "{verb} {group_col} with {metric_col} surpassing recent 3-month peak",
                    "{verb} {group_col} breaking 90-day {metric_col} records",
                    "{verb} {group_col} achieving quarterly {metric_col} peaks"
                ],
                'explanation': "Uses ALL comparison in subquery to find records where {metric_col} exceeds all values from the previous 90 days for that {group_col}, identifying new quarterly peaks, filtered for {year}",
                'tables_needed': ['{table1}'],
                'requires': ['grouping_col', 'metric_col', 'date_col'],
                'verb_style': 'detect'
            },
            {
                'sql': """SELECT {group_col}, {metric_col},
    (SELECT COUNT(*) FROM {table1} t2 WHERE t2.{metric_col} > t1.{metric_col}) + 1 as overall_rank
FROM {table1} t1
WHERE {date_col} >= '{year}-01-01'
ORDER BY overall_rank
LIMIT {top_n};""",
                'prompt_templates': [
                    "{verb} top {top_n} {group_col} by {metric_col} with calculated ranks for {year}",
                    "{verb} highest-ranked {group_col} based on {metric_col} in {year}",
                    "{verb} leading {top_n} {group_col} ordered by {metric_col} rank",
                    "{verb} best {top_n} {group_col} with position rankings by {metric_col}",
                    "{verb} {top_n} highest {group_col} showing their {metric_col} ranks"
                ],
                'explanation': "Subquery counts how many records have higher {metric_col} to calculate rank for each record, main query returns top {top_n} by this calculated rank for records since {year}",
                'tables_needed': ['{table1}'],
                'requires': ['grouping_col', 'metric_col', 'date_col'],
                'verb_style': 'rank'
            }
        ]
    
    def calculate_diversity_stats(self) -> Dict:
        """Calculate expected unique combinations and duplicate rates"""    
        # Parameter variation counts (ENHANCED)
        variations = {
            'templates_per_complexity': 7,  # average
            'domains': 11,
            'prompt_variations': 5,
            'tables_per_domain': 4,
            'column_combinations': 4,
            'years': 10,  # 2015-2024 (was 5)
            'months': 12,
            'days': 28,
            'threshold': 100000,  # 1-100,000 (was 9,900)
            'top_n': 50,  # 1-50 (was 8)
            'aggregation_functions': 5,  # SUM, AVG, MAX, MIN, COUNT
            'comparison_operators': 5,  # >, >=, <, <=, !=
            'where_clause_styles': 5
        }
        
        # Calculate total combinations
        total_combinations = 1
        for param, count in variations.items():
            total_combinations *= count
        
        return {
            'variations': variations,
            'total_combinations': total_combinations
        }
    
    def calculate_expected_duplicates(self, sample_size: int) -> Dict:
        """Calculate expected duplicate rate using birthday paradox"""
        stats = self.calculate_diversity_stats()
        total_combos = stats['total_combinations']
        
        # Birthday paradox: probability of collision
        # P(duplicate) ≈ n²/(2N) for small n²/N
        pairs = (sample_size * (sample_size - 1)) / 2
        expected_duplicate_pairs = pairs / total_combos
        expected_duplicate_rows = expected_duplicate_pairs * 2
        duplicate_percentage = (expected_duplicate_rows / sample_size * 100) if sample_size > 0 else 0
        
        return {
            'sample_size': sample_size,
            'total_combinations': total_combos,
            'expected_duplicate_rows': int(expected_duplicate_rows),
            'duplicate_percentage': duplicate_percentage,
            'unique_rows': sample_size - int(expected_duplicate_rows),
            'unique_percentage': 100 - duplicate_percentage
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
