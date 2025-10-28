from typing import Dict

def _get_healthcare_domain() -> Dict:
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
