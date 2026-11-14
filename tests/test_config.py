import pytest

from alacv.config import ConfigError, load_config


def test_default_config_loads():
    cfg = load_config("configs/default.yaml")
    assert cfg.get("model.id") == "facebook/dinov2-small"
    assert cfg.get("pyramid.scales") == [1, 2, 4, 8]
    assert cfg.get("model.input_size") % cfg.get("model.patch_size") == 0


def test_config_hash_stable():
    a = load_config("configs/default.yaml")
    b = load_config("configs/default.yaml")
    assert a.sha256 == b.sha256


def test_stable_hash_ignores_paths(tmp_path):
    a = load_config("configs/default.yaml")
    b = load_config(
        "configs/default.yaml",
        overrides={"paths": {"raw_dir": str(tmp_path / "x")}},
    )
    assert a.sha256 != b.sha256
    assert a.stable_sha256 == b.stable_sha256


def test_invalid_scales_rejected():
    with pytest.raises(ConfigError):
        load_config("configs/default.yaml", overrides={"pyramid": {"scales": [4, 2]}})


def test_input_size_must_be_multiple_of_patch():
    with pytest.raises(ConfigError):
        load_config("configs/default.yaml", overrides={"model": {"input_size": 500}})


def test_unknown_source_rejected():
    with pytest.raises(ConfigError):
        load_config("configs/default.yaml", overrides={"encoding": {"variants": {"X": ["bogus"]}}})


def test_snapshot_written(tmp_path):
    cfg = load_config("configs/default.yaml")
    p = cfg.snapshot(tmp_path)
    assert p.exists()
    assert (tmp_path / "config.sha256").read_text().strip() == cfg.sha256
