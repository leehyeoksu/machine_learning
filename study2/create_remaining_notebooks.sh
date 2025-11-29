#!/usr/bin/env python3
# Quick script to verify all 7 notebooks exist
import os
import glob

notebooks = sorted(glob.glob("study2/Day*.ipynb"))
print(f"Found {len(notebooks)} notebooks:")
for nb in notebooks:
    size = os.path.getsize(nb)
    print(f"  {os.path.basename(nb): <40} {size:>8,} bytes")
