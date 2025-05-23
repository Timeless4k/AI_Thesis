# setup.py - Installation helper
"""
Setup script to prepare the environment for thesis experiments
"""

import subprocess
import sys

def setup_environment():
    """Set up the Python environment"""
    
    print("Setting up thesis experiment environment...")
    
    # Install required packages
    print("\nInstalling required packages...")
    import os
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found. Please make sure it's in the root directory.")
        sys.exit(1)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    except subprocess.CalledProcessError as e:
        print(f"❌ pip install failed: {e}")
        print("Please check your requirements.txt and internet connection.")
        sys.exit(1)
    
    # Create necessary directories
    directories = ['./results', './thesis_results', './logs', './models']
    for dir in directories:
        os.makedirs(dir, exist_ok=True)
        print(f"Created directory: {dir}")
    
    print("\nEnvironment setup complete!")
    print("You can now run the experiments using: python main.py")

if __name__ == "__main__":
    setup_environment()