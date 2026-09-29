from __future__ import annotations

import pytest

from osa.utils.config import ConfigError, EnvironmentConfig


def test_environment_config_reads_text_and_defaults() -> None:
    config = EnvironmentConfig({"OSA_TEST_TEXT": "  hello  "})

    assert config.get("OSA_TEST_TEXT") == "hello"
    assert config.get("OSA_MISSING", "fallback") == "fallback"


def test_environment_config_parses_boolean_integer_and_float() -> None:
    config = EnvironmentConfig(
        {
            "OSA_TEST_BOOL": "yes",
            "OSA_TEST_INT": "42",
            "OSA_TEST_FLOAT": "1.5",
        }
    )

    assert config.boolean("OSA_TEST_BOOL") is True
    assert config.integer("OSA_TEST_INT", 0) == 42
    assert config.floating("OSA_TEST_FLOAT", 0.0) == 1.5


def test_environment_config_rejects_invalid_typed_values() -> None:
    config = EnvironmentConfig(
        {
            "OSA_TEST_BOOL": "maybe",
            "OSA_TEST_INT": "abc",
            "OSA_TEST_FLOAT": "xyz",
        }
    )

    with pytest.raises(ConfigError):
        config.boolean("OSA_TEST_BOOL")

    with pytest.raises(ConfigError):
        config.integer("OSA_TEST_INT", 0)

    with pytest.raises(ConfigError):
        config.floating("OSA_TEST_FLOAT", 0.0)


def test_environment_config_requires_non_empty_value() -> None:
    config = EnvironmentConfig({"OSA_EMPTY": "  "})

    with pytest.raises(ConfigError):
        config.require("OSA_EMPTY")
