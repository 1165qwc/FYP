#!/usr/bin/env python3
"""
Simple setup script for Facial Expression Expressiveness Recognition System
Run this to set up the environment and test the installation.
"""

import subprocess
import sys
import os

def run_command(cmd, description=""):
    """Run a command and return success status"""
    try:
        print(f"[SETUP] {description}")
        result = subprocess.run(cmd, shell=True, check=True,
                              capture_output=True, text=True)
        print(f"[OK] {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAILED] {description}")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("Facial Expression Expressiveness Recognition - Setup")
    print("=" * 60)

    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"Python version: {python_version}")

    if sys.version_info < (3, 8):
        print("[ERROR] Python 3.8 or higher is required")
        return

    print("[OK] Python version is compatible")
    print()

    # Install requirements
    print("Installing Python packages...")
    if not run_command(f"pip install -r requirements.txt",
                      "Installing dependencies"):
        print("[ERROR] Failed to install dependencies")
        return

    print()
    print("[SUCCESS] Setup completed!")
    print()
    print("Next steps:")
    print("1. Test the system: python test_system.py")
    print("2. Train model: jupyter notebook Facial_Expression_Expressiveness_Recognition.ipynb")
    print("3. Run demo: python real_time_demo.py")
    print()
    print("For detailed usage, see README.md and USAGE_GUIDE.md")

if __name__ == "__main__":
    main()