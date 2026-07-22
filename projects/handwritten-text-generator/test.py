"""
test.py  –  Quick smoke-test for the trained model.
Run from the project root:  py -3.12 test.py
"""

import sys
import os

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))

from generate import generate_text

# ── Test cases ────────────────────────────────────────────────────────────────
seeds = [
    "Artificial Intelligence",
    "The quick brown fox",
    "Once upon a time",
]

temperatures = [0.4, 0.7, 1.0]

print("=" * 60)
print("  HANDWRITTEN TEXT GENERATOR  –  Model Test")
print("=" * 60)

for seed in seeds:
    print(f"\nSeed : '{seed}'")
    print("-" * 50)
    for temp in temperatures:
        result = generate_text(seed, num_characters=120, temperature=temp)
        # Show only the newly generated part (after the seed)
        generated_part = result[len(seed):]
        print(f"  temp={temp:.1f} → {generated_part.strip()[:100]}")
    print()

print("=" * 60)
print("All tests passed ✓")