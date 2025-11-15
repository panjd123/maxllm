"""
Run all MaxLLM tests
"""
import asyncio
import sys
import os
import subprocess

# Add parent directory to path for importing maxllm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()


def run_test(test_file):
    """Run a single test file"""
    print("\n" + "=" * 80)
    print(f"Running: {test_file}")
    print("=" * 80)

    try:
        result = subprocess.run(
            [sys.executable, test_file],
            cwd=os.path.dirname(__file__),
            capture_output=False,
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error running {test_file}: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("MaxLLM Test Suite")
    print("=" * 80)

    # Check if .env file exists
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if not os.path.exists(env_path):
        print("\n⚠️  WARNING: .env file not found!")
        print(f"   Please create .env file at: {env_path}")
        print("   Copy from .env.example and fill in your API keys")
        response = input("\nContinue anyway? (y/N): ")
        if response.lower() != 'y':
            return

    # List of test files
    test_files = [
        "test_basic.py",
        "test_json.py",
        "test_batch.py",
        "test_embedding.py",
    ]

    # Run each test
    results = {}
    for test_file in test_files:
        test_path = os.path.join(os.path.dirname(__file__), test_file)
        if os.path.exists(test_path):
            results[test_file] = run_test(test_path)
        else:
            print(f"⚠️  Test file not found: {test_file}")
            results[test_file] = False

    # Print summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    for test_file, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_file}")

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\nTotal: {passed}/{total} tests passed")
    print()


if __name__ == "__main__":
    main()
