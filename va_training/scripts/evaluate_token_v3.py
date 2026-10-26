import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from v3.training import eval_main
if __name__ == '__main__': eval_main()
