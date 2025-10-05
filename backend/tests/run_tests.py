#!/usr/bin/env python3
"""
Test runner for FinTech DataGen backend tests.

This script runs all unit tests and provides a comprehensive test report.

Usage:
    python run_tests.py
    python run_tests.py --verbose
    python run_tests.py --coverage

Author: FinTech DataGen Team
Date: October 2025
"""

import unittest
import sys
import os
import argparse
from io import StringIO

# Add tests directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_tests(verbose=False, coverage=False):
    """Run all tests and return results."""
    
    print("=" * 60)
    print("FinTech DataGen - Backend Test Suite")
    print("=" * 60)
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(os.path.abspath(__file__))
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests
    runner = unittest.TextTestRunner(
        verbosity=2 if verbose else 1,
        stream=sys.stdout,
        descriptions=True,
        failfast=False
    )
    
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED!")
        return True
    else:
        print("\n❌ SOME TESTS FAILED!")
        return False

def main():
    """Main function to run tests."""
    parser = argparse.ArgumentParser(description='Run FinTech DataGen backend tests')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Run tests with verbose output')
    parser.add_argument('--coverage', '-c', action='store_true',
                       help='Run tests with coverage report (requires coverage.py)')
    
    args = parser.parse_args()
    
    if args.coverage:
        try:
            import coverage
            cov = coverage.Coverage()
            cov.start()
        except ImportError:
            print("Warning: coverage.py not installed. Running tests without coverage.")
            args.coverage = False
    
    success = run_tests(verbose=args.verbose, coverage=args.coverage)
    
    if args.coverage:
        try:
            cov.stop()
            cov.save()
            print("\n" + "=" * 60)
            print("COVERAGE REPORT")
            print("=" * 60)
            cov.report()
        except NameError:
            pass
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
