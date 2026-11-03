# Defines the cache folder paths used by the pilot and weight downloader.
# REPO, CACHE, HUB and TOKEN point inside the repository-root layerskip/ folder.
# ensure() creates the cache folders when they do not exist yet.
# Central path settings for downloaded models and data.
# Keeps multi-gigabyte weights out of the source folders and out of git.
# Import CACHE, HUB or TOKEN instead of hard-coding locations.
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
