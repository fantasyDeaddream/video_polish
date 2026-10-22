"""Build and hash-check a source-only or model-inclusive ZIP from an allowlist."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = ('README.md', 'RELEASE_NOTES.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md',
              '.gitignore', 'requirements.txt', 'requirements-acceleration.txt',
              'start.bat', 'stop.bat', 'start.sh', 'stop.sh')
SOURCE_DIRS = {'app': {'.py'}, 'web': {'.html', '.css', '.js', '.svg'},
               'scripts': {'.py', '.ps1', '.sh'}, 'tests': {'.py'},
               'docs': {'.md', '.txt', '.png'}}
EXCLUDED_DIRS = {'__pycache__', 'artifacts', '.pytest_cache', '.git', '.agents', '.codex'}


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def version():
    tree = ast.parse((ROOT / 'app/config.py').read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'VERSION' for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError('Version not found')


def git_value(*args):
    try:
        result = subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True)
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def source_files():
    files = {name: ROOT / name for name in ROOT_FILES}
    for folder, extensions in SOURCE_DIRS.items():
        for parent, directories, names in os.walk(ROOT / folder, followlinks=False):
            directories[:] = sorted(d for d in directories if d not in EXCLUDED_DIRS and not (Path(parent) / d).is_symlink())
            for name in sorted(names):
                path = Path(parent) / name
                if path.suffix.lower() in extensions and not path.is_symlink():
                    files[path.relative_to(ROOT).as_posix()] = path
    for name in ('models/catalog.json', 'models/lightweight-manifest.json', 'models/rife/manifest.json'):
        files[name] = ROOT / name
    return files


def models():
    from scripts.prepare_models import existing_model
    files = {}
    catalog = json.loads((ROOT / 'models/catalog.json').read_text(encoding='utf-8'))
    for record in catalog:
        relative = PurePosixPath(record['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise RuntimeError('Invalid catalog path')
        matches = [p for p in existing_model(str(relative))
                   if p.stat().st_size == record['bytes'] and digest(p) == record['sha256']]
        if not matches:
            raise RuntimeError(f'Missing/corrupt model: {relative}. Run scripts/prepare_models.py first.')
        files['models/' + str(relative)] = matches[0]
    references = json.loads((ROOT / 'models/rife/manifest.json').read_text(encoding='utf-8'))
    for record in references:
        for name, info in record['members'].items():
            relative = f'models/rife/{record["model"]}/{name}'
            path = ROOT / relative
            if digest(path) != info['sha256']:
                raise RuntimeError('RIFE reference file hash mismatch: ' + relative)
            files[relative] = path
    return files


def verify(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError('Duplicate ZIP members')
        roots = {PurePosixPath(name).parts[0] for name in names}
        if len(roots) != 1:
            raise RuntimeError('Archive must have one top-level directory')
        prefix = next(iter(roots)) + '/'
        for name in names:
            parts = PurePosixPath(name).parts
            if PurePosixPath(name).is_absolute() or '..' in parts:
                raise RuntimeError('Unsafe ZIP member')
            if any(p in {'.runtime', '.deps', 'cache', 'logs', '.git', '__pycache__', '.env'} for p in parts):
                raise RuntimeError('Local runtime data in archive: ' + name)
        manifest = json.loads(archive.read(prefix + 'RELEASE-MANIFEST.json'))
        expected = {prefix + row['path'] for row in manifest['files']} | {prefix + 'RELEASE-MANIFEST.json'}
        if set(names) != expected:
            raise RuntimeError('ZIP members do not match release manifest')
        for row in manifest['files']:
            with archive.open(prefix + row['path']) as handle:
                checksum = hashlib.sha256()
                size = 0
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    checksum.update(chunk)
                    size += len(chunk)
            if size != row['bytes'] or checksum.hexdigest() != row['sha256']:
                raise RuntimeError('Archive checksum mismatch: ' + row['path'])
    print(f'Verified {len(manifest["files"])} files; all CRCs/sizes/SHA256 match.', flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-only', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'dist')
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.verify:
        verify(args.verify)
        return
    release_version = version()
    git_root = git_value('rev-parse', '--show-toplevel')
    own_git = git_root is not None and Path(git_root).resolve() == ROOT
    commit = git_value('rev-parse', 'HEAD') if own_git else None
    if commit and git_value('status', '--porcelain', '--untracked-files=normal'):
        raise RuntimeError('Commit source changes before packaging')
    epoch = git_value('show', '-s', '--format=%ct', 'HEAD') if commit else None
    timestamp = datetime.fromtimestamp(int(epoch), timezone.utc) if epoch else datetime(2026, 10, 22, tzinfo=timezone.utc)
    zip_time = (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute, timestamp.second // 2 * 2)
    files = source_files()
    if not args.source_only:
        files.update(models())
    rows = [{'path': name, 'bytes': path.stat().st_size, 'sha256': digest(path)} for name, path in sorted(files.items())]
    manifest = {'project': 'VideoPolish', 'version': release_version, 'source_commit': commit,
                'contains_models': not args.source_only, 'files': rows,
                'runtime': 'Existing Python environment required; TensorRT optional and not bundled.'}
    args.output.mkdir(parents=True, exist_ok=True)
    suffix = 'source' if args.source_only else 'with-models'
    filename = f'VideoPolish-v{release_version}-{suffix}.zip'
    destination = args.output / filename
    partial = destination.with_suffix('.partial.zip')
    prefix = f'VideoPolish-v{release_version}/'
    with zipfile.ZipFile(partial, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for name, path in sorted(files.items()):
            info = zipfile.ZipInfo(prefix + name, date_time=zip_time)
            info.create_system = 3
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100755 if name.endswith('.sh') else 0o100644) << 16
            with path.open('rb') as source, archive.open(info, 'w') as target:
                for chunk in iter(lambda: source.read(1024 * 1024), b''):
                    target.write(chunk)
        info = zipfile.ZipInfo(prefix + 'RELEASE-MANIFEST.json', date_time=zip_time)
        info.create_system = 3
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(info, json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2).encode('utf-8'))
    verify(partial)
    partial.replace(destination)
    checksum = digest(destination)
    destination.with_suffix('.zip.sha256').write_text(f'{checksum}  {filename}\n', encoding='utf-8')
    print(json.dumps({'zip': str(destination), 'bytes': destination.stat().st_size, 'sha256': checksum,
                      'source_commit': commit, 'files': len(rows)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
