#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from network_prep import main  # noqa: E402


if __name__ == "__main__":
    main()