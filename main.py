"""Hydruxiom - 3D Tag Space Explorer.

Standalone app: projects a Hydrus tag collection into navigable 3D space.

Usage:
    python main.py
"""

import os
import sys

# Ensure project root is on sys.path so `src` is importable
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    from src.app import run
    return run()


if __name__ == "__main__":
    sys.exit(main())
