import subprocess
import sys

def main():
    print("=" * 40)
    print("   Welcome to OmniRec 🎯")
    print("   Multi-Domain Recommender")
    print("=" * 40)
    print("\nLaunching app...")
    subprocess.run([
        sys.executable, "-m", "streamlit",
        "run", "app/streamlit_app.py"
    ])

if __name__ == "__main__":
    main()
