"""LLZO: updated readout, with current amplifier."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edl_updated_entry import main

if __name__ == "__main__":
    raise SystemExit(main('LLZO', True))
