
import os
import sys
import subprocess
from pathlib import Path


current_dir = Path(__file__).parent.absolute()
scripts_directory = str(current_dir)
scripts_directory = os.path.abspath(scripts_directory)

scripts_to_run = [
    "main_gudhi_persistence.py",
    "main_plotting_gudhi_representations.py",
]

def run_script(script_name):
    script_path = os.path.join(scripts_directory, script_name)
    print(f"Running script: {script_path}")

    result = subprocess.run(
        [sys.executable, script_path],
        check=True,
        cwd=scripts_directory
    )
    print(f"Script {script_name} executed successfully.")

if __name__ == "__main__":
    print(f"Current working directory: {os.getcwd()}")
    print(f"Scripts directory: {scripts_directory}")

    for script in scripts_to_run:
        run_script(script)