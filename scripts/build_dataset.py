import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_prep import build, check
if __name__=="__main__": build(); check()
