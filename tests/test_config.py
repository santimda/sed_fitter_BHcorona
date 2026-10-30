import os
import subprocess
import sys
from pathlib import Path


def test_external_data_path_override(tmp_path):
    environment = os.environ.copy()
    environment["SMBH_CORONA_DATA_PATH"] = str(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from smbh_corona.config import CONFIGS_PATH, DATA_PATH; "
            "print(DATA_PATH); print(CONFIGS_PATH)",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    data_path, configs_path = result.stdout.splitlines()
    assert Path(data_path) == tmp_path.resolve()
    assert Path(configs_path) == tmp_path.resolve() / "configs"
