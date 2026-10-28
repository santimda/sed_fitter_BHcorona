import os
from pathlib import Path


PACKAGE_PATH = Path(__file__).resolve().parent
PROJECT_PATH = PACKAGE_PATH.parents[1]
DEFAULT_DATA_PATH = PROJECT_PATH / "data"
DATA_PATH = Path(os.environ.get("SMBH_CORONA_DATA_PATH", DEFAULT_DATA_PATH)).expanduser().resolve()
PARAMETERS_PATH = DATA_PATH / "parameters"
FLUX_PATH = DATA_PATH / "fluxes"
