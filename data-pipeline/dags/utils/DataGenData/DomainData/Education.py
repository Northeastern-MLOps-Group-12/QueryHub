from typing import Dict

def _get_education_domain() -> Dict:
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
