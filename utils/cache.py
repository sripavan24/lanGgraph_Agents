from pathlib import Path

def get_cache_dir():
    cache_dir = Path("resume_cache")
    cache_dir.mkdir(exist_ok=True)
    return cache_dir