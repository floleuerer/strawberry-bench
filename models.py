"""
OpenRouter model registry for StrawberryBench.

All models are called via the OpenRouter API using the OpenAI-compatible interface.
Set OPENROUTER_API_KEY in your environment before running.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    id: str            # OpenRouter model ID
    name: str          # Short display name for leaderboard
    provider: str      # Organization name
    max_tokens: int = 10000
    temperature: float = 0.0


MODELS: list[ModelConfig] = [
    ModelConfig("anthropic/claude-opus-4.6",      "Claude Opus 4.6",       "Anthropic"),
    ModelConfig("anthropic/claude-sonnet-4.6",    "Claude Sonnet 4.6",     "Anthropic"),
    ModelConfig("openai/gpt-5.2",                 "GPT-5.2",               "OpenAI"),
    ModelConfig("z-ai/glm-5",                     "GLM-5",                 "Zhipu AI"),
    ModelConfig("openai/gpt-5.2-codex",           "GPT-5.2 Codex",         "OpenAI"),
    ModelConfig("google/gemini-3-pro-preview",    "Gemini 3 Pro",          "Google"),
    ModelConfig("moonshotai/kimi-k2.5",           "Kimi K2.5",             "Moonshot AI"),
    ModelConfig("google/gemini-3-flash-preview",  "Gemini 3 Flash",        "Google"),
    ModelConfig("qwen/qwen3.5-397b-a17b",         "Qwen3.5 397B",          "Alibaba"),
    ModelConfig("minimax/minimax-m2.5",           "MiniMax M2.5",          "MiniMax"),
    ModelConfig("deepseek/deepseek-v3.2",         "DeepSeek V3.2",         "DeepSeek"),
    ModelConfig("x-ai/grok-4",                    "Grok 4",                "xAI"),
    ModelConfig("xiaomi/mimo-v2-flash",           "MiMo-V2-Flash",         "Xiaomi"),
    ModelConfig("nvidia/nemotron-3-nano-30b-a3b", "Nemotron 3 Nano 30B",   "NVIDIA"),
    #ModelConfig("mistralai/mistral-large-2512",   "Mistral Large 3",       "Mistral"),
]

MODEL_BY_ID: dict[str, ModelConfig] = {m.id: m for m in MODELS}


def get_model(model_id: str) -> ModelConfig:
    if model_id not in MODEL_BY_ID:
        # Allow ad-hoc model IDs not in the registry
        return ModelConfig(id=model_id, name=model_id, provider="Unknown", tier="unknown")
    return MODEL_BY_ID[model_id]
