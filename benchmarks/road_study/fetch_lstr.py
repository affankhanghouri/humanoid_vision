"""Fetch ~3.3 MB of pinned official LSTR weights/source, without a training install."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('models/lstr_source'))
    args=parser.parse_args()
    manifest=json.loads(Path(__file__).with_name('lstr_sources.json').read_text())
    base=f"https://raw.githubusercontent.com/{manifest['repository']}/{manifest['revision']}/"
    for name,digest in manifest['files'].items():
        target=args.out/name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==digest:
            print('Cached',name);continue
        with urllib.request.urlopen(base+name,timeout=45) as response:
            data=response.read(5_000_001)
        if len(data)>5_000_000 or hashlib.sha256(data).hexdigest()!=digest:
            raise RuntimeError(f'Size/hash mismatch for {name}; no file written')
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(data)
        print('Saved',name,len(data),flush=True)

if __name__=='__main__':main()
