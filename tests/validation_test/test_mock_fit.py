from pathlib import Path
from types import SimpleNamespace

from fit_mock_galaxy import fit_mock_galaxy


def test_mock_fit_recovers_truth_quickly(tmp_path):
    args = SimpleNamespace(
        nsteps=100,
        nwalkers=32,
        seed=12345,
        output_dir=tmp_path,
    )
    summary = fit_mock_galaxy(args)

    assert summary["all_within_one_sigma"]
    assert Path(summary["mock_data"]).is_file()
    assert Path(summary["mock_data"]).read_text().splitlines()[1].startswith("0.8 ")
    assert "log_M" not in summary["fit_parameters"]
    assert summary["fixed_parameters"]["log_M"] == 7.5
    assert (tmp_path / "test_galaxy_summary.json").is_file()
    assert (tmp_path / "test_galaxy_fit.pdf").is_file()
    assert (tmp_path / "posterior_test_galaxy.pdf").is_file()
