"""Materialize the frozen review candidate in a new directory without a Git checkout/rollback."""
from pathlib import Path
import argparse, hashlib, json, shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("destination", nargs="?", help="New, non-existing output directory")
    p.add_argument("--verify", action="store_true", help="Only verify the available sources")
    p.add_argument("--source-root", type=Path, default=ROOT, help="Base checkout with original LFS inputs")
    a = p.parse_args()
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    if not a.verify and not a.destination:
        p.error("destination required unless --verify is selected")
    base = a.source_root.resolve()
    dest = Path(a.destination).resolve() if a.destination else None
    if dest is not None and dest.exists():
        p.error("destination already exists; choose a new directory")
    sources = {}
    errors = []
    for name, expected in manifest["files"].items():
        path = HERE / "overlay" / name if name in manifest["overlay"] else base / name
        if not path.is_file():
            errors.append(f"missing: {name}")
            continue
        with path.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        if digest != expected:
            errors.append(f"hash mismatch: {name}")
        sources[name] = path
    if errors:
        raise SystemExit("Frozen source verification failed. Fetch existing LFS objects or provide the correct base checkout.\n" + "\n".join(errors))
    print(f"Verified {len(sources)} frozen candidate files")
    if dest is not None and not a.verify:
        dest.mkdir(parents=True, exist_ok=False)
        for name, source in sources.items():
            path = dest / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, path)
        print(f"Materialized {manifest['candidate']} at {dest}")

if __name__ == "__main__":
    main()
