"""Checkout entry point; installed users can run triton-msl-export."""
from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triton_msl.tools.export_source_program import export, main

if __name__ == "__main__":
    main()
