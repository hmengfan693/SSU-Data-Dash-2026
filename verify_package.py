"""Verify the release file sizes and SHA-256 hashes without external dependencies."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    records = json.loads((root / 'PACKAGE_MANIFEST.json').read_text())['files']
    for record in records:
        path = root / record['path']
        path.resolve().relative_to(root)
        if not path.is_file() or path.stat().st_size != record['bytes']:
            raise ValueError('Missing or resized release file: ' + record['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Changed release file: ' + record['path'])
    print(f'Package verified: {len(records)} release files match their recorded SHA-256 hashes.')


if __name__ == '__main__':
    main()
