#!/usr/bin/env python3
"""
Single-command entrypoint for Cantill.ai Spend Analytics & Zero-Shot Classification Pipeline.
"""

import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from spend_pipeline.cli import main

if __name__ == "__main__":
    main()

