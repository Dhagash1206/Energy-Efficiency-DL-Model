"""Download LayerSkip weights using bounded HTTP ranges; verify SHA-256."""
import hashlib
import os
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parent
try:
    from .paths import CACHE, HUB, TOKEN, ensure
except ImportError:  # executed directly as a script, not via -m
    import sys
    sys.path.insert(0, str(ROOT.parent.parent))
    from prototype.selective_feedback.paths import CACHE, HUB, TOKEN, ensure
os.environ.pop('HF_TOKEN', None)
ensure()
os.environ['HF_HOME'] = str(CACHE)
os.environ['HF_HUB_CACHE'] = str(HUB)
os.environ['HF_TOKEN_PATH'] = str(TOKEN)
from huggingface_hub import get_hf_file_metadata, hf_hub_url
import requests


def main():
    repo = 'facebook/layerskip-llama3.2-1B'
    meta = get_hf_file_metadata(hf_hub_url(repo, 'model.safetensors'), token=True, timeout=30)
    if not re.fullmatch(r'[0-9a-f]{64}', meta.etag or ''):
        raise RuntimeError('Expected SHA-256 metadata for weights')
    directory = HUB / 'models--facebook--layerskip-llama3.2-1B' / 'snapshots' / meta.commit_hash
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / 'model.safetensors'
    partial = directory / 'model.safetensors.part'
    if final.exists():
        with final.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != meta.etag:
            raise RuntimeError('Existing weights hash mismatch')
        print('Existing weights verified.', flush=True)
        return
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > meta.size:
        raise RuntimeError('Partial file exceeds expected size')
    step = 8 * 1024 * 1024
    print(f'Downloading {meta.size} bytes at revision {meta.commit_hash}; resuming at {offset}', flush=True)
    with requests.Session() as session, partial.open('ab') as stream:
        while offset < meta.size:
            end = min(offset + step, meta.size) - 1
            for attempt in range(3):
                try:
                    with session.get(meta.location, headers={'Range': f'bytes={offset}-{end}'}, timeout=(20, 30), stream=True) as response:
                        expected = f'bytes {offset}-{end}/{meta.size}'
                        if response.status_code != 206 or response.headers.get('Content-Range') != expected:
                            raise RuntimeError(f'Unexpected range response: HTTP {response.status_code}')
                        data = bytearray()
                        for chunk in response.iter_content(1024 * 1024):
                            data.extend(chunk)
                            if len(data) > end - offset + 1:
                                raise RuntimeError('Response exceeds requested range')
                        if len(data) != end - offset + 1:
                            raise RuntimeError('Incomplete range')
                    break
                except Exception as exc:
                    print(f'Range retry {attempt + 1}: {type(exc).__name__}', flush=True)
                    if attempt == 2:
                        raise SystemExit('Download interrupted; rerun to resume.') from None
                    time.sleep(2)
            stream.write(data)
            stream.flush()
            offset = end + 1
            if offset % (step * 8) == 0 or offset == meta.size:
                print(f'Weights: {offset / meta.size:.1%} ({offset // (1024 * 1024)} MiB)', flush=True)
    with partial.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != meta.etag:
        raise RuntimeError('Downloaded weights failed SHA-256 verification')
    partial.replace(final)
    print('Weights downloaded and SHA-256 verified.', flush=True)


if __name__ == '__main__':
    main()
