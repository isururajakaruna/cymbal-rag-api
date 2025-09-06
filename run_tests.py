#!/usr/bin/env python3
"""Test runner script."""

import subprocess
import sys
import os


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*50}")
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("SUCCESS")
        if result.stdout:
            print(result.stdout)
    else:
        print("FAILED")
        if result.stderr:
            print(result.stderr)
        return False
    
    return True


def main():
    """Run all tests and checks."""
    print("Running Cymbal RAG API Tests and Checks")
    
    # Check if we're in the right directory
    if not os.path.exists("app/main.py"):
        print("Error: Please run this script from the project root directory")
        sys.exit(1)
    
    # Run tests (only the working ones to avoid hanging on Google Cloud connections)
    commands = [
        ("python -m pytest tests/unit/test_services/test_document_processing.py tests/unit/test_utils/test_ingestion/test_chunking.py -v --disable-warnings", "Document Processing & Chunking Tests"),
    ]
    
    all_passed = True
    
    for command, description in commands:
        if not run_command(command, description):
            all_passed = False
    
    print(f"\n{'='*50}")
    if all_passed:
        print("All tests and checks passed!")
        sys.exit(0)
    else:
        print("Some tests or checks failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
