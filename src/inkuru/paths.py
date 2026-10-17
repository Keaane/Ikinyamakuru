from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Everything lives inside this repository. The KinNews CSV drop (data/source/)
# and the transformer checkpoints (models/runs/) are large and stay out of git;
# the deployed linear model, its cutoff, the scores, and the figures are tracked.
DATA = ROOT / "data"
RAW = DATA / "raw"
SOURCE = DATA / "source" / "KINNEWS" / "raw"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
MODELS = ROOT / "models"
RUNS = MODELS / "runs"
BASELINES = MODELS / "baselines"

MODEL_NAME = "Davlan/afro-xlmr-mini"
SEED = 42
MAX_LENGTH = 128
DEV_FRACTION = 0.10
