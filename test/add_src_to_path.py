from pathlib import Path
import sys


def append_src():
    sys.path.append(str(Path(__file__).parent.parent))