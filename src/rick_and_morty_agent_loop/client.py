"""Which provider and model the loop talks to.

Three providers are supported: OpenRouter, OpenAI and Claude. Pydantic AI gives
each one a client that speaks its own protocol behind a single interface, so the
loop in `main.py` never learns which it got: https://ai.pydantic.dev/models/

Pydantic AI can't work out the provider on its own -- `infer_model` needs the
`provider:` prefix to know which client class to build, and only then does that
class go looking for its own key. So the sniffing happens here: whichever key is
in the environment picks the provider, and that provider's default model runs
unless MODEL names another.
"""

import os

from pydantic_ai.models import Model, infer_model

PROVIDERS = {
    # our name      pydantic-ai prefix, default model, API key
    "openrouter": ("openrouter", "inclusionai/ling-3.0-flash-vl:free", "OPENROUTER_API_KEY"),
    "openai": ("openai", "gpt-5.2", "OPENAI_API_KEY"),
    "claude": ("anthropic", "claude-opus-5", "ANTHROPIC_API_KEY"),
}


def detect() -> str:
    """The provider whose key is set. Ties break in PROVIDERS order."""
    # A blank value doesn't count -- .env.example ships every key empty, so
    # `in os.environ` would match all three on a fresh copy.
    found = [name for name, (*_, key) in PROVIDERS.items() if os.environ.get(key)]
    if not found:
        keys = ", ".join(key for *_, key in PROVIDERS.values())
        raise SystemExit(f"no API key set -- put one of {keys} in .env")
    return found[0]


def resolve(provider: str = "", model: str = "") -> Model:
    """Blank provider means "whichever key is set"; blank model means its default."""
    provider = provider or detect()
    if provider not in PROVIDERS:
        raise SystemExit(
            f"unknown provider {provider!r} -- pick one of: {', '.join(PROVIDERS)}"
        )

    prefix, default_model, key = PROVIDERS[provider]
    # Naming a provider skips detect(), so its key still has to be checked.
    if not os.environ.get(key):
        raise SystemExit(f"provider {provider!r} needs {key} -- set it in .env")

    return infer_model(f"{prefix}:{model or default_model}")
