"""Convenient launcher for normal and embedded Python installations."""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT.parents[1]))
if __name__ == "__main__":
    runpy.run_module("prototype.selective_feedback.pilot", run_name="__main__")

