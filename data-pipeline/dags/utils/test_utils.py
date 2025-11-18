import pytest
from pathlib import Path

def run_unit_tests():
    """
    Runs pytest on model_training_test.py.
    DAG should fail if any test fails.
    """

    # Path to your test file
    test_file = Path(__file__).parent.parent.parent / "tests" / "model_training_tests.py"

    print(f"Running tests: {test_file}")

    # Run pytest programmatically
    exit_code = pytest.main([str(test_file)])

    if exit_code != 0:
        raise Exception(f"Unit tests failed with exit code {exit_code}")

    print("All tests passed successfully.")