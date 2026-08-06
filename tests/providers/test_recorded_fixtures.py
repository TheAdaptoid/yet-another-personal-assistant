"""Tests that provider parsers handle real recorded API responses."""

from types import SimpleNamespace
from unittest.mock import patch

from yapa.models import EmbedModel, LanguageModel
from yapa.providers.lmstudio import LMStudioIP
from yapa.providers.openai import OpenAIIP
from yapa.providers.openai.openai_compat import OpenAICompatibleProvider
from yapa.providers.openrouter import OpenRouterProvider
from yapa.services.config import Config, ProviderConfig

from .conftest import load_fixture


class _P(OpenAICompatibleProvider):
    def __init__(self):
        super().__init__("x", "X", api_key="k", base_url="http://x/v1")


def _openai_provider():
    with patch("yapa.providers.openai._noauth.AsyncOpenAI"):
        return OpenAIIP(
            Config(provider_configs={"openai": ProviderConfig(api_key="sk-t")})
        )


def _openrouter_provider():
    with patch("yapa.providers.openai._noauth.AsyncOpenAI"):
        return OpenRouterProvider(
            Config(
                provider_configs={"openrouter": ProviderConfig(api_key="sk-or-test")}
            )
        )


def _to_obj(d: dict) -> SimpleNamespace:
    """Convert a dict to an object with attributes (mimics SDK parsing)."""
    return SimpleNamespace(**d)


def test_openrouter_models_parsing() -> None:
    p = _openrouter_provider()
    raw = load_fixture("openrouter_models.json")
    formatted = [p._format_model_from_openrouter(m) for m in raw["data"]]
    assert type(formatted[0]) is LanguageModel
    assert formatted[0].pricing.input == 0.002  # 0.000002 * 1000 (per-1M)
    assert formatted[0].pricing.output == 0.006
    assert formatted[0].supports_tools is True
    assert type(formatted[1]) is EmbedModel


def test_openrouter_reasoning_extracted() -> None:
    p = _openrouter_provider()
    raw = load_fixture("openrouter_chat_response.json")
    msg = _to_obj(raw["choices"][0]["message"])
    assert p._extract_reasoning(msg) == 'The user said "Say'


def test_lmstudio_native_models_parsing() -> None:
    with patch("yapa.providers.openai._noauth.AsyncOpenAI"):
        p = LMStudioIP(Config(provider_configs={"lmstudio": ProviderConfig()}))
    raw = load_fixture("lmstudio_models_native.json")
    formatted = [p._format_model_from_native(m) for m in raw["models"]]
    assert type(formatted[0]) is LanguageModel
    assert formatted[0].supports_tools is True
    assert formatted[0].supports_vision is True
    assert type(formatted[1]) is EmbedModel


def test_lmstudio_reasoning_content() -> None:
    p = _openai_provider()
    raw = load_fixture("lmstudio_chat_response.json")
    msg = _to_obj(raw["choices"][0]["message"])
    assert p._extract_reasoning(msg) == "The user is asking me"


def test_reasoning_fallback_fields() -> None:
    p = _P()
    # reasoning_content takes precedence
    obj = SimpleNamespace(reasoning="r", reasoning_content="rc")
    assert p._extract_reasoning(obj) == "rc"
    # falls back to reasoning when reasoning_content is None
    obj2 = SimpleNamespace(reasoning="r", reasoning_content=None)
    assert p._extract_reasoning(obj2) == "r"
    # whitespace-only becomes None
    obj3 = SimpleNamespace(reasoning=None, reasoning_content="   ")
    assert p._extract_reasoning(obj3) is None
