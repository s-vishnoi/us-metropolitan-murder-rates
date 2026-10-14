"""Check that all posterior files required by notebook 07 are present."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "5000" / "results"

expected = [
    RESULTS / "urban_scaling" / f"model_{number}" / "Murder.pkl"
    for number in range(1, 23)
]
expected += [
    RESULTS / "cobb_douglas" / f"model_{number}" / "Murder.pkl"
    for number in range(1, 38)
]

missing = [path for path in expected if not path.is_file()]

print(f"Expected: {len(expected)} posterior files")
print(f"Present:  {len(expected) - len(missing)}")
print(f"Missing:  {len(missing)}")

if missing:
    print("\nMissing files:")
    for path in missing:
        print(path.relative_to(ROOT))
    raise SystemExit(1)

print("All Quest posterior files are present.")
