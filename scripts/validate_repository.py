"""Run lightweight integrity checks before publishing the repository."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript" / "Nature_Cities_Manuscript.pdf"
EXPECTED_MANUSCRIPT_SHA256 = (
    "aabd2f26fc5d0c6d99479935348986a148873b00786855f1f3e99079182c296f"
)
MAX_GITHUB_FILE_SIZE = 100 * 1024 * 1024

EXPECTED_FIGURES = {
    "fig1A.pdf",
    "fig1B.pdf",
    "fig2A.pdf",
    "fig2B.pdf",
    "fig3A.pdf",
    "fig3B.pdf",
    "fig3CD.pdf",
    "fig4.pdf",
    "homicide-rates.pdf",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    problems: list[str] = []

    if not MANUSCRIPT.exists() or sha256(MANUSCRIPT) != EXPECTED_MANUSCRIPT_SHA256:
        problems.append("The manuscript is missing or differs from the authoritative PDF.")

    analysis_path = ROOT / "data" / "log_df.csv"
    analysis_data = None
    if analysis_path.exists():
        analysis_data = pd.read_csv(analysis_path, index_col=0)
        if analysis_data["MSA"].nunique() != 344:
            problems.append("The analysis dataset does not contain the manuscript's 344 MSAs.")
        if set(analysis_data["Year"].unique()) != set(range(1960, 2021)):
            problems.append("The analysis dataset does not cover every year from 1960–2020.")
        if analysis_data.duplicated(["MSA", "Year"]).any():
            problems.append("The analysis dataset contains duplicate MSA-year rows.")
        if analysis_data.isna().any().any():
            problems.append("The analysis dataset contains missing values.")

    helper_text = (ROOT / "src" / "pystan_cobb_douglas_mod.py").read_text()
    for setting in ("NITER     = 7000", "WARMUP    = 6000", "NCHAINS   = 2", "THIN      = 1"):
        if setting not in helper_text:
            problems.append(f"Bayesian helper setting is missing: {setting}")

    manifest = [
        line for line in (ROOT / "quest_download_manifest.txt").read_text().splitlines()
        if line.strip()
    ]
    if len(manifest) != 59 or len(set(manifest)) != 59:
        problems.append("The Quest manifest must contain 59 unique posterior paths.")

    present_figures = {path.name for path in (ROOT / "figures").glob("*.pdf")}
    missing_figures = EXPECTED_FIGURES - present_figures
    if missing_figures:
        problems.append(f"Missing reference figures: {sorted(missing_figures)}")

    for notebook in sorted((ROOT / "notebooks").glob("*.ipynb")):
        try:
            parsed = json.loads(notebook.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            problems.append(f"Invalid notebook {notebook.name}: {error}")
            continue
        if parsed.get("nbformat") != 4:
            problems.append(f"Unexpected notebook format in {notebook.name}.")
        for cell in parsed.get("cells", []):
            for output in cell.get("outputs", []):
                if output.get("output_type") == "error":
                    problems.append(f"Stored error output in {notebook.name}.")

    for path in ROOT.rglob("*"):
        if path.is_file() and path.stat().st_size >= MAX_GITHUB_FILE_SIZE:
            problems.append(f"File exceeds GitHub's 100 MB limit: {path.relative_to(ROOT)}")

    forbidden_names = {"Cover-Letter.pdf", "Nature_COMM_reviewers_rejections.pdf"}
    leaked = [path for path in ROOT.rglob("*") if path.name in forbidden_names]
    if leaked:
        problems.append(f"Private submission files found: {leaked}")

    if problems:
        print("Repository validation failed:")
        for problem in problems:
            print(f"- {problem}")
        raise SystemExit(1)

    print("Repository validation passed.")
    print(f"Manuscript SHA-256: {EXPECTED_MANUSCRIPT_SHA256}")
    print(f"Notebooks: {len(list((ROOT / 'notebooks').glob('*.ipynb')))}")
    print(f"Reference figures: {len(EXPECTED_FIGURES)}")
    if analysis_data is not None:
        print(f"Local analysis MSAs: {analysis_data['MSA'].nunique()}")
        print(f"Local analysis years: {analysis_data['Year'].min()}–{analysis_data['Year'].max()}")
    else:
        print("Processed analysis data: not present (code-only public release)")
    print(f"Quest posterior paths: {len(manifest)}")


if __name__ == "__main__":
    main()
