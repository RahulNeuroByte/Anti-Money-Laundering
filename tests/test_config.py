from aml.config import load_config


def test_config_loads_and_target_is_correct_case():
    cfg = load_config()
    assert cfg["data"]["target"] == "Is_laundering"
    assert "Laundering_type" in cfg["data"]["leakage_columns"]


def test_config_missing_file_raises():
    import pytest

    with pytest.raises(FileNotFoundError):
        load_config("configs/does_not_exist.yaml")
