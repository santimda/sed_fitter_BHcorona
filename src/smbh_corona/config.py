from pathlib import Path


PACKAGE_PATH = Path(__file__).resolve().parent
PROJECT_PATH = PACKAGE_PATH.parents[1]
DATA_PATH = PROJECT_PATH / "data"
PARAMETERS_PATH = DATA_PATH / "parameters"
FLUX_PATH = DATA_PATH / "fluxes"
