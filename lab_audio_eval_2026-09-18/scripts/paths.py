"""Local paths for the report experiments; each input can be overridden."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]

def env_path(name, default):
    return Path(os.environ.get(name, str(default))).expanduser().resolve()

REPO = env_path('CONTACT_AUDIO_PACKAGE', ROOT / 'src/contact_audio_perception')
sys.path.insert(0, str(REPO))
BAGS = env_path('BAGS_DIR', ROOT / 'data/home_bags')
MODELS = env_path('MODELS_DIR', REPO / 'models')
OUT = env_path('OUT_DIR', ROOT / 'output/evaluation')
FIG = env_path('FIG_DIR', ROOT / 'output/evaluation/figures')
DER = env_path('DERIVED_DIR', ROOT / 'data/maniwav/derived')
ZIP = env_path('MANIWAV_ZIP', ROOT / 'data/maniwav/wipe_replay_buffer.zarr.zip')
