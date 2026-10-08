"""Fetch the actual released incumbent without falling back on transport/integrity failures."""
import argparse
from pathlib import Path
from bsdraft.models.releases import download_incumbent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = download_incumbent(args.repository, args.output)
    print("downloaded " + (manifest["release_id"] if manifest else "legacy incumbent; explicit bootstrap required"))
