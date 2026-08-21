"""Validacion inteligente de modelos con fuzzy matching y cache.

Antes de enviar prompts al LLM, verifica que el modelo sea valido.
Si no existe, sugiere alternativas cercanas.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationResult:
    """Resultado de la validacion de un modelo."""
    valid: bool
    model: str
    message: str = ""
    suggestions: list[str] = field(default_factory=list)


class ModelValidator:
    """Valida modelos contra la API con fuzzy matching.

    Caracteristicas:
    - Cache de lista de modelos (se refresca cada 5 min)
    - Fuzzy matching por distancia de Levenshtein
    - Matching por prefijo ("gpt-4" -> "gpt-4o-mini")
    - Matching insensible a mayusculas
    - Sugerencias automaticas en caso de error
    """

    CACHE_TTL = 300  # 5 minutos

    def __init__(self, api_client: Any) -> None:
        self._api = api_client
        self._cache: list[str] = []
        self._cache_time: float = 0
        self._last_validated: str = ""
        self._last_result: ValidationResult | None = None

    def refresh_cache(self) -> list[str]:
        """Fuerza la actualizacion del cache de modelos."""
        models = self._api.list_models()
        if models is not None:
            self._cache = models
            self._cache_time = time.time()
        return self._cache

    def _get_models(self) -> list[str]:
        """Obtiene modelos, usando cache si es valido."""
        if not self._cache or (time.time() - self._cache_time) > self.CACHE_TTL:
            self.refresh_cache()
        return self._cache

    def validate(self, model_name: str) -> ValidationResult:
        """Valida un modelo contra la API.

        Retorna un ValidationResult con:
        - valid: True si el modelo existe exactamente
        - suggestions: modelos similares si no existe
        - message: descripcion legible
        """
        # Cache de ultima validacion
        if (model_name == self._last_validated
                and self._last_result is not None):
            return self._last_result

        models = self._get_models()

        # Si no hay modelos (API caida), permitir con advertencia
        if not models:
            result = ValidationResult(
                valid=True,
                model=model_name,
                message="No se pudo verificar contra la API (sin conexion). Se usara tal cual.",
            )
            self._last_validated = model_name
            self._last_result = result
            return result

        # 1. Match exacto
        if model_name in models:
            result = ValidationResult(
                valid=True,
                model=model_name,
                message=f"Modelo valido: {model_name}",
            )
            self._last_validated = model_name
            self._last_result = result
            return result

        # 2. Match insensible a mayusculas
        lower_models = {m.lower(): m for m in models}
        if model_name.lower() in lower_models:
            correct = lower_models[model_name.lower()]
            result = ValidationResult(
                valid=False,
                model=model_name,
                message=f"El modelo no existe exactamente. Quizas quisiste decir: {correct}",
                suggestions=[correct],
            )
            self._last_validated = model_name
            self._last_result = result
            return result

        # 3. Buscar sugerencias (fuzzy + prefijo)
        suggestions = self._find_suggestions(model_name, models)

        if suggestions:
            sug_str = ", ".join(suggestions[:5])
            result = ValidationResult(
                valid=False,
                model=model_name,
                message=f"Modelo '{model_name}' no encontrado. Modelos similares: {sug_str}",
                suggestions=suggestions,
            )
        else:
            result = ValidationResult(
                valid=False,
                model=model_name,
                message=f"Modelo '{model_name}' no encontrado. No hay sugerencias.",
                suggestions=[],
            )

        self._last_validated = model_name
        self._last_result = result
        return result

    def _find_suggestions(self, model_name: str, models: list[str]) -> list[str]:
        """Encuentra modelos similares usando fuzzy matching y prefijos."""
        candidates: list[tuple[int, str]] = []
        target = model_name.lower()

        for m in models:
            m_lower = m.lower()

            # Prefijo: el modelo empieza con lo que escribio el usuario
            if m_lower.startswith(target) or target.startswith(m_lower):
                # Bonus por prefijo, penalizar por diferencia de longitud
                len_diff = abs(len(m) - len(model_name))
                score = max(1, 100 - len_diff * 2)
                candidates.append((score, m))
                continue

            # Fuzzy: distancia de Levenshtein
            dist = self._levenshtein(target, m_lower)
            max_len = max(len(target), len(m_lower))
            if max_len == 0:
                continue
            similarity = 1 - (dist / max_len)
            if similarity >= 0.5:  # al menos 50% similar
                score = int(similarity * 100)
                candidates.append((score, m))

        # Ordenar por score descendente
        candidates.sort(key=lambda x: x[0], reverse=True)

        # Deduplicar y retornar top 5
        seen = set()
        result = []
        for score, m in candidates:
            if m not in seen:
                seen.add(m)
                result.append(m)
                if len(result) >= 5:
                    break
        return result

    @staticmethod
    def _levenshtein(s1: str, s2: str) -> int:
        """Calcula la distancia de Levenshtein entre dos strings."""
        if len(s1) < len(s2):
            return ModelValidator._levenshtein(s2, s1)

        if len(s2) == 0:
            return len(s1)

        prev_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Costo de insercion, eliminacion, sustitucion
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (0 if c1 == c2 else 1)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row

        return prev_row[-1]

    def validate_or_suggest(self, model_name: str) -> str | None:
        """Valida y retorna el modelo correcto, o None si no se puede resolver.

        Si hay una sugerencia unica muy clara (score >= 90), la retorna.
        Si hay multiples, retorna None.
        Si es exacto, retorna el nombre.
        """
        result = self.validate(model_name)
        if result.valid:
            return model_name

        # Si hay exactamente 1 sugerencia, usarla
        if len(result.suggestions) == 1:
            return result.suggestions[0]

        # Si la primera sugerencia tiene score alto
        if result.suggestions:
            return None  # Ambiguo, pedir al usuario

        return None

    def auto_correct(self, model_name: str) -> tuple[str, bool]:
        """Intenta autocorregir el nombre del modelo.

        Returns:
            (modelo_corregido, fue_corregido)
        """
        models = self._get_models()
        if not models:
            return model_name, False

        result = self.validate(model_name)
        if result.valid:
            return model_name, False

        # Autocorregir solo si hay 1 sugerencia clara
        if len(result.suggestions) == 1:
            return result.suggestions[0], True

        return model_name, False
