import gzip
import shutil
import sys
import os

if len(sys.argv) != 3:
    raise SystemExit(f"Usage: {sys.argv[0]} <src.gz> <dest>")

src, dest = sys.argv[1], sys.argv[2]
os.makedirs(os.path.dirname(dest), exist_ok=True)

with open(src, 'rb') as raw:
    magic = raw.read(2)
    raw.seek(0)
    with (gzip.open(raw, 'rb') if magic == b'\x1f\x8b' else raw) as f_in, open(dest, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
