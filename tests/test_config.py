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
            "from smbh_corona.config import DATA_PATH; print(DATA_PATH)",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert Path(result.stdout.strip()) == tmp_path.resolve()
