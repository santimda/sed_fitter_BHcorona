import json
from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from galaxy_sed import (
    _limit_kwargs,
    _prepare_output,
    _validate_config,
    build_parser,
    load_config,
    parse_bool,
)


DEFAULT_CONFIG_PATH = Path(__file__).parents[1] / "data" / "configs" / "default_config.json"


def read_default_config():
    return json.loads(DEFAULT_CONFIG_PATH.read_text())


def test_parser_uses_positional_galaxy_and_defaults_to_fit():
    args = build_parser().parse_args(["IC4329A"])

    assert args.galaxy_name == "IC4329A"
    assert args.fit is True


def test_parser_supports_plot_only_mode():
    args = build_parser().parse_args(["NGC985", "--fit=False"])

    assert args.fit is False
    assert parse_bool("True") is True
    assert parse_bool("false") is False


def test_missing_galaxy_config_is_created_from_default(tmp_path, capsys):
    default_config = read_default_config()
    (tmp_path / "default_config.json").write_text(json.dumps(default_config))

    config, config_path = load_config("IC4329A", configs_path=tmp_path)

    assert config == default_config
    assert config_path == tmp_path / "IC4329A_config.json"
    assert config_path.exists()
    assert "Created" in capsys.readouterr().out


def test_prepare_output_cleans_only_selected_galaxy(tmp_path):
    config = read_default_config()
    config["output"]["root"] = str(tmp_path)
    other_output = tmp_path / "IC4329A"
    other_output.mkdir(parents=True)
    (other_output / "keep.txt").write_text("keep")
    selected_output = tmp_path / "NGC985"
    (selected_output / "emcee_fit").mkdir(parents=True)
    (selected_output / "emcee_fit" / "old.dat").write_text("old")
    (selected_output / "fit_summary.json").write_text("old")

    output_dir = _prepare_output("NGC985", config)

    assert output_dir == selected_output
    assert not (selected_output / "emcee_fit").exists()
    assert not (selected_output / "fit_summary.json").exists()
    assert (other_output / "keep.txt").read_text() == "keep"


def test_default_config_preserves_current_ngc985_settings():
    config = read_default_config()

    assert config["fit"]["w"] == 3.0
    assert config["fit"]["nsteps"] == 4000
    assert config["fit"]["nwalkers"] == 50
    assert config["data"]["min_ang_scale_arcsec"] == 0.05
    assert config["data"]["max_ang_scale_arcsec"] == 0.4
    assert config["plot"]["xmin_ghz"] == 1.01
    assert config["plot"]["xmax_ghz"] == 2000.0


def test_invalid_config_reports_missing_section(tmp_path):
    config_path = tmp_path / "broken.json"

    try:
        _validate_config({}, config_path)
    except ValueError as error:
        assert "model" in str(error)
    else:
        raise AssertionError("Expected invalid config to raise ValueError")


def test_limit_kwargs_includes_limit_widths():
    class GalaxyLimits:
        data_frequencies_UL = np.array([100.0])
        data_fluxes_UL = np.array([2.0])
        data_fluxes_UL_error = np.array([0.2])
        data_frequencies_LL = np.array([200.0])
        data_fluxes_LL = np.array([1.0])
        data_fluxes_LL_error = np.array([0.1])

    kwargs = _limit_kwargs(GalaxyLimits())

    assert kwargs["x_ul"].tolist() == [100e9]
    assert kwargs["sigma_ul"].tolist() == [0.2]
    assert kwargs["x_ll"].tolist() == [200e9]
    assert kwargs["sigma_ll"].tolist() == [0.1]
