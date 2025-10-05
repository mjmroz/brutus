#!/usr/bin/env python
"""
Standard coverage runner for brutus project.

This script handles the coverage instrumentation conflicts by running
test suites separately and combining results. Always use this script
for accurate coverage measurement instead of running pytest directly.

Usage:
    python run_coverage.py              # Run all test suites
    python run_coverage.py --utils      # Only utils comprehensive tests
    python run_coverage.py --core       # Only core module tests
    python run_coverage.py --analysis   # Only analysis module tests
    python run_coverage.py --data       # Only data module tests
    python run_coverage.py --plotting   # Only plotting module tests
    python run_coverage.py --priors     # Only priors module tests
    python run_coverage.py --dust       # Only dust module tests
    python run_coverage.py --report-only # Generate reports from existing data
    python run_coverage.py --help       # Show help
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description, timeout=120):
    """Run a command and return success status with timing and verbose output."""
    import time

    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        # Run with real-time output for better visibility
        print("📊 Starting test execution...")

        result = subprocess.run(
            cmd,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent,
            # Show output in real-time but also capture it
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        elapsed_time = time.time() - start_time

        if result.returncode == 0:
            print(f"✅ SUCCESS (took {elapsed_time:.2f}s)")

            # Show test results summary
            lines = result.stdout.split("\n")

            # Show test count and any failures
            for line in lines:
                if "passed" in line or "failed" in line or "error" in line:
                    if any(x in line for x in ["==", "passed", "failed", "error"]):
                        print(f"   📋 {line.strip()}")

            # Show coverage summary
            for line in lines:
                if "TOTAL" in line and "%" in line:
                    print(f"   📈 Coverage: {line}")

            # Warn about slow tests
            if elapsed_time > 60:
                print(f"   ⚠️  SLOW TEST: {description} took {elapsed_time:.1f}s")
            elif elapsed_time > 30:
                print(f"   🐌 MODERATE: {description} took {elapsed_time:.1f}s")

            return True
        else:
            print(f"❌ FAILED (took {elapsed_time:.2f}s)")

            # Show stdout for context
            if result.stdout:
                print("\n📤 STDOUT (last 800 chars):")
                print(result.stdout[-800:])

            # Show stderr
            if result.stderr:
                print("\n📥 STDERR (last 500 chars):")
                print(result.stderr[-500:])

            if "ImportError" in result.stderr:
                print("\n⚠️  Coverage instrumentation conflict detected")
                print("   This is expected and handled by running tests separately")

            return False

    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        print(f"⏰ TIMEOUT after {elapsed_time:.1f}s (limit: {timeout}s)")
        print(f"   Consider breaking down {description} into smaller test suites")
        return False
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"💥 ERROR after {elapsed_time:.1f}s: {e}")
        return False


def main():
    """Run coverage analysis with proper test separation."""

    parser = argparse.ArgumentParser(
        description="Run brutus coverage analysis avoiding instrumentation conflicts"
    )
    parser.add_argument(
        "--utils", action="store_true", help="Only run utils comprehensive tests"
    )
    parser.add_argument(
        "--core", action="store_true", help="Only run core module tests"
    )
    parser.add_argument(
        "--analysis", action="store_true", help="Only run analysis module tests"
    )
    parser.add_argument(
        "--data", action="store_true", help="Only run data module tests"
    )
    parser.add_argument(
        "--plotting", action="store_true", help="Only run plotting module tests"
    )
    parser.add_argument(
        "--priors", action="store_true", help="Only run priors module tests"
    )
    parser.add_argument(
        "--dust", action="store_true", help="Only run dust module tests"
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only generate reports from existing coverage data",
    )

    args = parser.parse_args()

    print("🧪 BRUTUS COVERAGE ANALYSIS")
    print("=" * 60)
    print("Using test separation to avoid instrumentation conflicts")

    if args.report_only:
        print("\n📊 Generating reports from existing coverage data...")
        run_command(
            ["python", "-m", "coverage", "report", "--show-missing"],
            "Coverage Report",
            30,
        )
        run_command(["python", "-m", "coverage", "html"], "HTML Report", 30)
        return True

    # Remove existing coverage data for clean slate
    for f in [".coverage", "coverage.xml"]:
        if os.path.exists(f):
            os.remove(f)
            print(f"Removed existing {f}")

    success_count = 0
    total_count = 0

    # Define test suites that must run separately
    test_suites = []

    if args.utils:
        # Only utils tests
        test_suites = [
            (
                [
                    "python",
                    "-m",
                    "pytest",
                    "tests/test_utils/test_math_comprehensive.py",
                    "tests/test_utils/test_photometry_comprehensive.py",
                    "tests/test_utils/test_sampling_comprehensive.py",
                    "--cov=src.brutus",
                    "--cov-append",
                    "--cov-report=term-missing",
                    "-v",
                ],
                "Utils Comprehensive Tests",
            )
        ]
    elif args.core:
        # Only core tests
        test_suites = [
            (
                [
                    "python",
                    "-m",
                    "pytest",
                    "tests/test_core/test_core_integration.py",
                    "--cov=src.brutus",
                    "--cov-append",
                    "--cov-report=term-missing",
                    "-v",
                ],
                "Core Integration Tests",
            )
        ]
    elif args.analysis:
        # Only analysis tests
        test_suites = [
            (
                [
                    "python",
                    "-m",
                    "pytest",
                    "tests/test_analysis/",
                    "--cov=src.brutus",
                    "--cov-append",
                    "--cov-report=term-missing",
                    "-v",
                ],
                "Analysis Tests",
            )
        ]
    elif args.data:
        # Only data tests
        test_suites = [
            (
                [
                    "python",
                    "-m",
                    "pytest",
                    "tests/test_data/",
                    "--cov=src.brutus",
                    "--cov-append",
                    "--cov-report=term-missing",
                    "-v",
                ],
                "Data Tests",
            )
        ]
    elif args.plotting:
        # Only plotting tests
        test_suites = [
            (
                [
                    "python",
                    "-m",
                    "pytest",
                    "tests/test_plotting/",
                    "--cov=src.brutus",
                    "--cov-append",
                    "--cov-report=term-missing",
                    "-v",
                ],
                "Plotting Tests",
            )
        ]
    elif args.priors:
        # Only priors tests
        test_suites = [
            (
                [
                    "python",
                    "-m",
                    "pytest",
                    "tests/test_priors/",
                    "--cov=src.brutus",
                    "--cov-append",
                    "--cov-report=term-missing",
                    "-v",
                ],
                "Priors Tests",
            )
        ]
    elif args.dust:
        # Only dust tests
        test_suites = [
            (
                [
                    "python",
                    "-m",
                    "pytest",
                    "tests/test_dust/",
                    "--cov=src.brutus",
                    "--cov-append",
                    "--cov-report=term-missing",
                    "-v",
                ],
                "Dust Tests",
            )
        ]
    else:
        # Full test suite - all tests should be fast enough to run
        test_suites = [
            (
                [
                    "python",
                    "-m",
                    "pytest",
                    "tests/test_utils/test_math_comprehensive.py",
                    "tests/test_utils/test_photometry_comprehensive.py",
                    "tests/test_utils/test_sampling_comprehensive.py",
                    "--cov=src.brutus",
                    "--cov-append",
                    "--cov-report=term-missing",
                    "-v",
                ],
                "Utils Comprehensive Tests",
            ),
            (
                [
                    "python",
                    "-m",
                    "pytest",
                    "tests/test_utils/test_photometry.py",
                    "tests/test_utils/test_utils_init.py",
                    "--cov=src.brutus",
                    "--cov-append",
                    "--cov-report=term-missing",
                    "-v",
                ],
                "Utils Basic Tests",
            ),
            (
                [
                    "python",
                    "-m",
                    "pytest",
                    "tests/test_core/test_import_structure.py",
                    "--cov=src.brutus",
                    "--cov-append",
                    "--cov-report=term-missing",
                    "-v",
                ],
                "Import Structure Tests",
            ),
            (
                [
                    "python",
                    "-m",
                    "pytest",
                    "tests/test_core/test_core_integration.py",
                    "--cov=src.brutus",
                    "--cov-append",
                    "--cov-report=term-missing",
                    "-v",
                ],
                "Core Integration Tests",
            ),
            (
                [
                    "python",
                    "-m",
                    "pytest",
                    "tests/test_analysis/",
                    "--cov=src.brutus",
                    "--cov-append",
                    "--cov-report=term-missing",
                    "-v",
                ],
                "Analysis Tests",
            ),
            (
                [
                    "python",
                    "-m",
                    "pytest",
                    "tests/test_data/",
                    "--cov=src.brutus",
                    "--cov-append",
                    "--cov-report=term-missing",
                    "-v",
                ],
                "Data Tests",
            ),
            (
                [
                    "python",
                    "-m",
                    "pytest",
                    "tests/test_plotting/",
                    "--cov=src.brutus",
                    "--cov-append",
                    "--cov-report=term-missing",
                    "-v",
                ],
                "Plotting Tests",
            ),
            (
                [
                    "python",
                    "-m",
                    "pytest",
                    "tests/test_priors/",
                    "--cov=src.brutus",
                    "--cov-append",
                    "--cov-report=term-missing",
                    "-v",
                ],
                "Priors Tests",
            ),
            (
                [
                    "python",
                    "-m",
                    "pytest",
                    "tests/test_dust/",
                    "--cov=src.brutus",
                    "--cov-append",
                    "--cov-report=term-missing",
                    "-v",
                ],
                "Dust Tests",
            ),
        ]

    # Run each test suite separately
    all_times = []

    for cmd, description in test_suites:
        total_count += 1
        import time

        suite_start = time.time()

        success = run_command(cmd, description)

        suite_elapsed = time.time() - suite_start
        all_times.append((description, suite_elapsed, success))

        if success:
            success_count += 1

    # Generate final reports
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS & TIMING ANALYSIS")
    print(f"{'='*60}")
    print(f"Successful test suites: {success_count}/{total_count}")

    # Show timing summary
    print(f"\n⏱️  TIMING SUMMARY:")
    total_time = sum(elapsed for _, elapsed, _ in all_times)
    print(f"   Total execution time: {total_time:.1f}s")

    for description, elapsed, success in sorted(
        all_times, key=lambda x: x[1], reverse=True
    ):
        status = "✅" if success else "❌"
        if elapsed > 60:
            speed = "🐌 SLOW"
        elif elapsed > 30:
            speed = "🟡 MODERATE"
        else:
            speed = "🟢 FAST"
        print(f"   {status} {speed} {description}: {elapsed:.1f}s")

    # Identify slow tests for follow-up
    slow_tests = [(desc, time) for desc, time, success in all_times if time > 30]
    if slow_tests:
        print(f"\n⚠️  TESTS TO OPTIMIZE (>30s):")
        for desc, elapsed in slow_tests:
            print(f"   • {desc}: {elapsed:.1f}s")

    if success_count > 0:
        print("\n📊 Generating combined coverage report...")
        run_command(
            ["python", "-m", "coverage", "report", "--show-missing"],
            "Final Coverage Report",
            30,
        )
        print("\n💾 Saving coverage data...")
        run_command(["python", "-m", "coverage", "html"], "HTML Report", 30)
        run_command(["python", "-m", "coverage", "xml"], "XML Report", 30)

        print(f"\n✅ Coverage analysis complete!")
        print(f"📂 View detailed report: open htmlcov/index.html")

        return True
    else:
        print("❌ No successful test suites - cannot generate coverage report")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
