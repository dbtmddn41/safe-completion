import sys, os, runpy
# Ensure the project root (where safe_rlhf/ lives) is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
runpy.run_module("safe_rlhf.finetune", run_name="__main__", alter_sys=True)
