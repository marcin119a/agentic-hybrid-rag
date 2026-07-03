"""Pobiera plik danych z Google Drive na podstawie ID pliku."""

import argparse
import sys
from pathlib import Path

import gdown

FILE_ID = "1dAsj-ZI0Uz9mNR4ydlCMEElQXfjFFrT3"
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "data" / "szkolenia.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Ścieżka docelowa pliku (domyślnie: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    result = gdown.download(url, str(args.output), quiet=False)

    if result is None:
        print("Pobieranie nie powiodło się.", file=sys.stderr)
        sys.exit(1)

    print(f"Zapisano: {args.output}")


if __name__ == "__main__":
    main()
