"""Contador de tokens y estimador de costo.

Estima tokens usando ~3-4 caracteres por token (regla general
para modelos que usan BPE). Muestra contadores en tiempo real
y permite al usuario saber cuanto gasta en cada interaccion.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


# Precios aproximados por 1M tokens (USD) - actualizar periodicamente
# Los precios se usan solo como estimacion
MODEL_PRICES: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    # Anthropic
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    # Google
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    # DeepSeek
    "deepseek-chat": {"input": 0.14, "output": 0.28},
    "deepseek-coder": {"input": 0.14, "output": 0.28},
    # Qwen
    "qwen2.5-coder": {"input": 0.30, "output": 1.20},
}


@dataclass
class TokenStats:
    """Estadisticas de token usage para una interaccion."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    response_time_ms: int = 0
    model: str = ""


@dataclass
class SessionStats:
    """Estadisticas acumuladas de la sesion."""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_api_calls: int = 0
    total_tool_calls: int = 0
    session_start: float = field(default_factory=time.time)

    @property
    def session_duration(self) -> str:
        elapsed = int(time.time() - self.session_start)
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        if hours:
            return f"{hours}h {minutes}m {seconds}s"
        if minutes:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"


class TokenCounter:
    """Contador de tokens y estimador de costo.

    Usa la API response cuando esta disponible (tokens reales),
    o estima basandose en caracteres cuando no.
    """

    # Caracteres por token (estimacion conservadora)
    CHARS_PER_TOKEN = 3.5

    def __init__(self, model_name: str = "") -> None:
        self.model_name = model_name
        self.session = SessionStats()

    def estimate_tokens(self, text: str) -> int:
        """Estima el numero de tokens en un texto."""
        if not text:
            return 0
        return max(1, int(len(text) / self.CHARS_PER_TOKEN))

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estima el costo en USD basado en precios del modelo."""
        model_lower = self.model_name.lower()

        # Buscar precio exacto
        if model_lower in MODEL_PRICES:
            prices = MODEL_PRICES[model_lower]
            return (
                (input_tokens / 1_000_000) * prices["input"]
                + (output_tokens / 1_000_000) * prices["output"]
            )

        # Buscar parcial (ej: "gpt-4o" dentro de "gpt-4o-2024-...")
        for key, prices in MODEL_PRICES.items():
            if key in model_lower or model_lower in key:
                return (
                    (input_tokens / 1_000_000) * prices["input"]
                    + (output_tokens / 1_000_000) * prices["output"]
                )

        # Default: estimacion generica
        return (input_tokens / 1_000_000) * 1.0 + (output_tokens / 1_000_000) * 3.0

    def record_api_call(
        self,
        input_text: str,
        output_text: str,
        response_time_ms: int = 0,
        api_usage: dict[str, Any] | None = None,
    ) -> TokenStats:
        """Registra una llamada a la API y retorna estadisticas.

        Args:
            input_text: Texto enviado a la API.
            output_text: Texto recibido de la API.
            response_time_ms: Tiempo de respuesta en milisegundos.
            api_usage: Uso real de la API (si disponible) con claves
                       'prompt_tokens', 'completion_tokens', 'total_tokens'.
        """
        if api_usage and "total_tokens" in api_usage:
            # Usar datos reales de la API
            input_t = api_usage.get("prompt_tokens", 0)
            output_t = api_usage.get("completion_tokens", 0)
            total_t = api_usage["total_tokens"]
        else:
            # Estimar
            input_t = self.estimate_tokens(input_text)
            output_t = self.estimate_tokens(output_text)
            total_t = input_t + output_t

        cost = self.estimate_cost(input_t, output_t)

        stats = TokenStats(
            input_tokens=input_t,
            output_tokens=output_t,
            total_tokens=total_t,
            estimated_cost_usd=cost,
            response_time_ms=response_time_ms,
            model=self.model_name,
        )

        # Actualizar sesion
        self.session.total_input_tokens += input_t
        self.session.total_output_tokens += output_t
        self.session.total_tokens += total_t
        self.session.total_cost_usd += cost
        self.session.total_api_calls += 1

        return stats

    def record_tool_call(self) -> None:
        """Registra una llamada a herramienta."""
        self.session.total_tool_calls += 1

    def set_model(self, model_name: str) -> None:
        """Actualiza el nombre del modelo."""
        self.model_name = model_name

    def get_session_summary(self) -> str:
        """Retorna un resumen legible de la sesion."""
        s = self.session
        lines = [
            f"  Llamadas API: {s.total_api_calls}",
            f"  Llamadas herramientas: {s.total_tool_calls}",
            f"  Tokens entrada: {s.total_input_tokens:,}",
            f"  Tokens salida: {s.total_output_tokens:,}",
            f"  Tokens total: {s.total_tokens:,}",
            f"  Costo estimado: ${s.total_cost_usd:.4f}",
            f"  Duracion sesion: {s.session_duration}",
        ]
        return "\n".join(lines)

    def format_stats(self, stats: TokenStats) -> str:
        """Formatea estadisticas de una llamada individual."""
        parts = [
            f"{stats.total_tokens:,} tokens",
            f"~${stats.estimated_cost_usd:.4f}",
        ]
        if stats.response_time_ms:
            parts.append(f"{stats.response_time_ms}ms")
        return " | ".join(parts)
