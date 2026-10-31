import json
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from galaxy_sed import (
    _prepare_output,
    _round_for_json,
    _validate_config,
    build_parser,
    load_config,
    parse_bool,
)


DEFAULT_CONFIG_PATH = Path(__file__).parents[1] / "data" / "configs" / "default_config.json"


def read_default_config():
    return json.loads(DEFAULT_CONFIG_PATH.read_text())


def test_parser_supports_fit_and_plot_only_modes():
    args = build_parser().parse_args(["IC4329A"])

    assert args.galaxy_name == "IC4329A"
    assert args.fit is True
    plot_args = build_parser().parse_args(["NGC985", "--fit=False"])
    assert plot_args.galaxy_name == "NGC985"
    assert plot_args.fit is False
    assert parse_bool("True") is True


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
    stale_artifacts = (
        "fit_result.json",
        "fit_summary.json",
        "run_metadata.json",
        "derived_summary.json",
        "posterior_NGC985.pdf",
        "SED_NGC985.pdf",
    )
    for filename in stale_artifacts:
        (selected_output / filename).write_text("old")

    output_dir = _prepare_output("NGC985", config)

    assert output_dir == selected_output
    assert not (selected_output / "emcee_fit").exists()
    for filename in stale_artifacts:
        assert not (selected_output / filename).exists()
    assert (other_output / "keep.txt").read_text() == "keep"


def test_fit_summary_values_are_rounded_recursively():
    rounded = _round_for_json({"outer": {"value": 1.23456}, "values": [2.34567]})

    assert rounded == {"outer": {"value": 1.235}, "values": [2.346]}


def test_invalid_config_reports_missing_section(tmp_path):
    config_path = tmp_path / "broken.json"

    with pytest.raises(ValueError, match="model"):
        _validate_config({}, config_path)
