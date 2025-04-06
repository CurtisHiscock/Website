import subprocess
import sys

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "torch==2.0.1",
    "--extra-index-url", "https://download.pytorch.org/whl/cpu"
])