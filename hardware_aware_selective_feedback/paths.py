"""Single source of truth for where Hugging Face downloads live.

Weights, tokenizer files and the Hugging Face token are stored under the
repository-root ``layerskip/cache`` folder: a dedicated top-level directory
for the checkpoint, not nested inside any existing folder.
None of it reaches source control: the root ``.gitignore`` excludes the
``layerskip/`` directory.

Results stay under this package (``results/pilot/``) because they are small,
reviewable artifacts that belong with the code that produced them.
"""
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parent
MODELS = REPO / "layerskip"
CACHE = MODELS / "cache"
HUB = CACHE / "hub"
TOKEN = CACHE / "token"


def ensure():
    """Create the cache directories if missing and return them."""
    CACHE.mkdir(parents=True, exist_ok=True)
    HUB.mkdir(parents=True, exist_ok=True)
    return CACHE, HUB
