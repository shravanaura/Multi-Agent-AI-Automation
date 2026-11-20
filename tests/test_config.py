import os
from unittest.mock import patch

import pytest


def test_load_config_with_all_keys() -> None:
    from src.config.env import load_config

    env = {
        "ANTHROPIC_API_KEY": "sk-ant-test",
        "TAVILY_API_KEY": "tvly-test",
    }
    with patch.dict(os.environ, env, clear=False):
        config = load_config()

    assert config.anthropic_api_key == "sk-ant-test"
    assert config.tavily_api_key == "tvly-test"
    assert config.openai_api_key is None
    assert config.port == 8005
    assert config.log_level == "info"


def test_load_config_with_optional_keys() -> None:
    from src.config.env import load_config

    env = {
        "ANTHROPIC_API_KEY": "sk-ant-test",
        "TAVILY_API_KEY": "tvly-test",
        "OPENAI_API_KEY": "sk-openai-test",
        "PORT": "9000",
        "LOG_LEVEL": "debug",
    }
    with patch.dict(os.environ, env, clear=False):
        config = load_config()

    assert config.openai_api_key == "sk-openai-test"
    assert config.port == 9000
    assert config.log_level == "debug"


def test_load_config_missing_anthropic_key() -> None:
    from src.config.env import load_config

    env = {"TAVILY_API_KEY": "tvly-test"}
    with patch.dict(os.environ, env, clear=False):
        os.environ.pop("ANTHROPIC_API_KEY", None)
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            load_config()


def test_load_config_missing_tavily_key() -> None:
    from src.config.env import load_config

    env = {"ANTHROPIC_API_KEY": "sk-ant-test"}
    with patch.dict(os.environ, env, clear=False):
        os.environ.pop("TAVILY_API_KEY", None)
        with pytest.raises(ValueError, match="TAVILY_API_KEY"):
            load_config()
