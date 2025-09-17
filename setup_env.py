# FILE: setup_env.py

import os, subprocess, sys, venv

def main():
    print("Setting up environment...")

    # Get rid of existing virtual environment
    if os.path.exists("venv"):
        print("Existing virtual environment found. Removing...")
        import shutil
        shutil.rmtree("venv")
    
    # Creating new virtual environment
    print("🐍 Creating new virtual environment...")
    venv.create("venv", with_pip=True)

    # Activate virtual environment
    if os.name != "nt":
        python_bin = os.path.join("venv", "bin", "python")
    else:
        python_bin = os.path.join("venv", "Scripts", "python.exe")
    subprocess.run([python_bin, "-m", "pip", "install", "--upgrade", "pip"], check=True)

    # Install requirements
    if os.path.exists("requirements.txt"):
        print("Installing dependencies...")
        subprocess.run([python_bin, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    else:
        print("requirements.txt not found. Skipping dependencies installation.")

    print_usage_instructions()


def print_usage_instructions():
    print("\n" + "="*60)
    print("🎉  Environment Setup Complete!")
    print("="*60)
    print("\n👉 To **activate** the virtual environment, run:\n")
    print("   source venv/bin/activate\n")
    print("After this, you will see '(venv)' at the start of your terminal prompt.")
    print("This means you are inside the virtual environment and can run project scripts.\n")
    print("❌ To **exit** the virtual environment, run:\n")
    print("   deactivate\n")
    print("💡 Tip: You must activate the environment every time you open a new terminal session.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()