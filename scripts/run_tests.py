#!/usr/bin/env python3
"""Test runner script for GenOps AI test suite."""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🧪 {description}...")
    try:
        result = subprocess.run(
            cmd, cwd=Path(__file__).parent, capture_output=True, text=True
        )

        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print(f"❌ {description} - FAILED")
            if result.stderr:
                print(result.stderr)
            if result.stdout:
                print(result.stdout)
            return False

    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False


def main():
    """Run comprehensive test suite."""
    print("🚀 GenOps AI Test Suite")
    print("=" * 50)

    # Ensure we're in the right directory
    os.chdir(Path(__file__).parent)

    success = True

    # 1. Run unit tests with coverage
    success &= run_command(
        [
            "python",
            "-m",
            "pytest",
            "tests/",
            "-v",
            "--cov=src/genops",
            "--cov-report=term-missing",
            "--cov-report=html",
        ],
        "Unit tests with coverage",
    )

    # 2. Run integration tests separately
    success &= run_command(
        ["python", "-m", "pytest", "tests/integration/", "-v", "-m", "integration"],
        "Integration tests",
    )

    # 3. Run linting
    success &= run_command(
        ["ruff", "check", "src/", "tests/"], "Code linting (ruff check)"
    )

    # 4. Run formatting check
    success &= run_command(
        ["ruff", "format", "--check", "src/", "tests/"],
        "Code formatting check (ruff format)",
    )

    # 5. Run type checking (if mypy is available)
    try:
        subprocess.run(["mypy", "--version"], check=True, capture_output=True)
        success &= run_command(["mypy", "src/genops"], "Type checking (mypy)")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️ mypy not available, skipping type checking")

    # 6. Test package import
    success &= run_command(
        [
            "python",
            "-c",
            "import sys; sys.path.insert(0, 'src'); import genops; print(f'✅ GenOps v{genops.__version__} imports successfully')",
        ],
        "Package import test",
    )

    # 7. Test CLI entry point
    success &= run_command(
        ["python", "-m", "genops.cli.main", "version"], "CLI entry point test"
    )

    print("\n" + "=" * 50)

    if success:
        print("🎉 ALL TESTS PASSED!")
        print("\nTest Summary:")
        print("• Unit tests: ✅")
        print("• Integration tests: ✅")
        print("• Code quality: ✅")
        print("• Package integrity: ✅")
        print("\n📊 Check htmlcov/index.html for detailed coverage report")
        return 0
    else:
        print("❌ SOME TESTS FAILED!")
        print("\nPlease fix failing tests before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
