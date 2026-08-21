"""Memoria a Corto Plazo — LRU de interacciones con compresion inteligente.

Gestiona las ultimas N interacciones de la sesion actual con politicas
de eviccion LRU y compresion por capas cuando se acerca al limite de tokens.

Tareas: T-19
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..api_client import Message


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CompressedBlock:
    """Representa un bloque de mensajes comprimidos en un resumen."""
    original_count: int
    summary: str
    turn_range: tuple[int, int]  # (first_turn, last_turn)
    hash_digest: str  # para detectar si el bloque ya fue comprimido
    created_at: float = field(default_factory=time.time)


@dataclass
class ShortTermConfig:
    """Configuracion de la memoria a corto plazo."""
    max_turns: int = 20  # ultimas N interacciones completas
    max_tokens: int = 80_000  # budget aproximado de ventana
    compress_ratio_threshold: float = 0.70  # comprimir al 70% de uso
    keep_recent_turns: int = 3  # nunca comprimir los ultimos N turnos
    summary_max_length: int = 600  # caracteres maximos por resumen de bloque
    tool_output_max_lines: int = 50  # lineas antes de truncar tool outputs antiguos


# ---------------------------------------------------------------------------
# ShortTermMemory
# ---------------------------------------------------------------------------


class ShortTermMemory:
    """Memoria a corto plazo con politica LRU y compresion por capas.

    Responsabilidades:
    1. Mantener las ultimas N interacciones accesibles
    2. Comprimir bloques antiguos cuando se acerca al limite
    3. Truncar tool outputs grandes de turnos antiguos
    4. Proveer el contexto optimo para cada llamada al LLM

    Uso tipico::
        stm = ShortTermMemory(config=ShortTermConfig())
        stm.add_turn(user_msg, assistant_msg, tool_messages)
        messages = stm.get_context_for_llm()
    """

    def __init__(
        self,
        config: ShortTermConfig | None = None,
    ) -> None:
        self.config = config or ShortTermConfig()

        # Turnos completos almacenados: OrderedDict[turn_id, TurnData]
        # OrderedDict mantiene orden de insercion = orden cronologico
        self._turns: OrderedDict[int, _TurnData] = OrderedDict()

        # Bloques comprimidos: lista ordenada por posicion
        self._compressed_blocks: list[CompressedBlock] = []

        # Contador de turnos (monotonico)
        self._next_turn_id: int = 1

        # Estimacion de tokens (aproximacion: 1 token ~ 4 chars)
        self._estimated_tokens: int = 0

        # Estadisticas
        self._total_turns_added: int = 0
        self._total_compressions: int = 0
        self._total_tokens_saved: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_turn(
        self,
        user_msg: Message | None = None,
        assistant_msg: Message | None = None,
        tool_messages: list[Message] | None = None,
    ) -> int:
        """Agrega un turno completo (user + assistant + tools) a la memoria.

        Args:
            user_msg: Mensaje del usuario (puede ser None si el turno
                     inicio por tool call interno).
            assistant_msg: Respuesta del asistente.
            tool_messages: Lista de resultados de herramientas.

        Returns:
            El turn_id asignado.
        """
        turn_id = self._next_turn_id
        self._next_turn_id += 1

        turn_data = _TurnData(
            turn_id=turn_id,
            user_msg=user_msg,
            assistant_msg=assistant_msg,
            tool_messages=tool_messages or [],
            added_at=time.time(),
        )

        self._turns[turn_id] = turn_data
        self._total_turns_added += 1
        self._update_token_estimate()

        # Eviccion LRU si excedemos max_turns
        self._maybe_evict()

        # Compresion si nos acercamos al limite de tokens
        self._maybe_compress()

        return turn_id

    def get_context_for_llm(
        self,
        system_prompt: str = "",
        include_compressed: bool = True,
    ) -> list[dict[str, Any]]:
        """Construye la lista de mensajes optimizada para enviar al LLM.

        Estrategia:
        1. System prompt
        2. Bloques comprimidos (resumen de historial antiguo)
        3. Turnos recientes sin comprimir

        Args:
            system_prompt: El system prompt del agente.
            include_compressed: Si incluir bloques comprimidos como resumen.

        Returns:
            Lista de mensajes en formato API de OpenAI.
        """
        result: list[dict[str, Any]] = []

        # 1. System prompt
        if system_prompt:
            result.append({"role": "system", "content": system_prompt})

        # 2. Bloques comprimidos como un solo mensaje resumen
        if include_compressed and self._compressed_blocks:
            summaries = []
            for block in self._compressed_blocks:
                summaries.append(
                    f"[Turnos {block.turn_range[0]}-{block.turn_range[1]} "
                    f"({block.original_count} mensajes)]: {block.summary}"
                )
            combined = "\n".join(summaries)
            result.append({
                "role": "assistant",
                "content": f"[Resumen de historial anterior]\n{combined}",
            })

        # 3. Turnos recientes completos
        for turn_id, turn in self._turns.items():
            if turn.user_msg:
                result.append(turn.user_msg.to_api_dict())
            if turn.assistant_msg:
                result.append(turn.assistant_msg.to_api_dict())
            for tool_msg in turn.tool_messages:
                result.append(tool_msg.to_api_dict())

        return result

    def get_recent_turns(self, n: int = 5) -> list[_TurnData]:
        """Retorna los ultimos N turnos sin comprimir."""
        turns = list(self._turns.values())
        return turns[-n:] if n > 0 else []

    def clear(self) -> None:
        """Limpia toda la memoria a corto plazo."""
        self._turns.clear()
        self._compressed_blocks.clear()
        self._next_turn_id = 1
        self._estimated_tokens = 0

    def force_compress(self) -> int:
        """Fuerza una compresion manual. Retorna tokens ahorrados."""
        return self._compress_oldest_turns()

    @property
    def stats(self) -> dict[str, Any]:
        """Estadisticas de la memoria a corto plazo."""
        return {
            "active_turns": len(self._turns),
            "compressed_blocks": len(self._compressed_blocks),
            "compressed_messages": sum(b.original_count for b in self._compressed_blocks),
            "estimated_tokens": self._estimated_tokens,
            "token_budget": self.config.max_tokens,
            "usage_ratio": round(
                self._estimated_tokens / self.config.max_tokens, 2
            ) if self.config.max_tokens > 0 else 0,
            "total_turns_added": self._total_turns_added,
            "total_compressions": self._total_compressions,
            "total_tokens_saved": self._total_tokens_saved,
        }

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    @property
    def is_near_limit(self) -> bool:
        """True si el uso estimado esta por encima del umbral de compresion."""
        return (
            self._estimated_tokens / self.config.max_tokens
            >= self.config.compress_ratio_threshold
        )

    # ------------------------------------------------------------------
    # Internal: Token estimation
    # ------------------------------------------------------------------

    def _estimate_tokens_for_message(self, msg: Message) -> int:
        """Estima tokens para un mensaje (~4 chars = 1 token para ingles,
        ~2 chars = 1 token para CJK/espanol con multibyte)."""
        total_chars = 0
        if msg.content:
            total_chars += len(msg.content)
        if msg.tool_calls:
            for tc in msg.tool_calls:
                total_chars += len(str(tc))
        # Estimacion conservadora: 3 chars por token
        return max(1, total_chars // 3)

    def _update_token_estimate(self) -> None:
        """Recalcula la estimacion total de tokens."""
        total = 0
        for turn in self._turns.values():
            if turn.user_msg:
                total += self._estimate_tokens_for_message(turn.user_msg)
            if turn.assistant_msg:
                total += self._estimate_tokens_for_message(turn.assistant_msg)
            for tm in turn.tool_messages:
                total += self._estimate_tokens_for_message(tm)
        self._estimated_tokens = total

    # ------------------------------------------------------------------
    # Internal: LRU Eviction
    # ------------------------------------------------------------------

    def _maybe_evict(self) -> None:
        """Evicta los turnos mas antiguos si excedemos max_turns.

        Los turnos evicted se comprimen antes de ser eliminados.
        """
        keep = self.config.keep_recent_turns
        while len(self._turns) > self.config.max_turns:
            # Evictar el turno mas antiguo (primero en OrderedDict)
            oldest_id = next(iter(self._turns))
            oldest = self._turns.pop(oldest_id)

            # Comprimir antes de perderlo
            summary = self._summarize_turn(oldest)
            if summary:
                self._add_compressed_block(oldest, summary)

    # ------------------------------------------------------------------
    # Internal: Compression
    # ------------------------------------------------------------------

    def _maybe_compress(self) -> None:
        """Ejecuta compresion por capas si el uso esta cerca del limite."""
        if not self.is_near_limit:
            return

        # Capa 1: Truncar tool outputs antiguos grandes
        self._truncate_old_tool_outputs()

        # Re-check despues de capa 1
        if not self.is_near_limit:
            return

        # Capa 2: Comprimir los turnos mas antiguos (excepto recientes)
        self._compress_oldest_turns()

    def _truncate_old_tool_outputs(self) -> None:
        """Trunca tool outputs de turnos antiguos que excedan el limite.

        Solo afecta turnos que NO estan entre los keep_recent_turns.
        """
        keep = self.config.keep_recent_turns
        turn_ids = list(self._turns.keys())
        evictable = turn_ids[:max(0, len(turn_ids) - keep)]

        max_lines = self.config.tool_output_max_lines

        for tid in evictable:
            turn = self._turns[tid]
            for tm in turn.tool_messages:
                if not tm.content:
                    continue
                lines = tm.content.split("\n")
                if len(lines) > max_lines:
                    kept = lines[:max_lines // 2]
                    removed = len(lines) - max_lines
                    tail = lines[-(max_lines // 2):]
                    tm.content = (
                        "\n".join(kept)
                        + f"\n\n[{removed} lineas omitidas]\n"
                        + "\n".join(tail)
                    )

        self._update_token_estimate()

    def _compress_oldest_turns(self) -> int:
        """Comprime los turnos mas antiguos en un bloque resumen.

        Returns:
            Tokens estimados ahorrados.
        """
        keep = self.config.keep_recent_turns
        turn_ids = list(self._turns.keys())
        evictable = turn_ids[:max(0, len(turn_ids) - keep)]

        if not evictable:
            return 0

        # Tomar hasta 5 turnos para comprimir en un bloque
        to_compress = evictable[:5]

        # Calcular tokens antes
        tokens_before = 0
        turns_data = []
        for tid in to_compress:
            turn = self._turns[tid]
            turns_data.append(turn)
            if turn.user_msg:
                tokens_before += self._estimate_tokens_for_message(turn.user_msg)
            if turn.assistant_msg:
                tokens_before += self._estimate_tokens_for_message(turn.assistant_msg)
            for tm in turn.tool_messages:
                tokens_before += self._estimate_tokens_for_message(tm)

        # Generar resumen combinado
        summary_parts = []
        for turn in turns_data:
            s = self._summarize_turn(turn)
            if s:
                summary_parts.append(s)

        if not summary_parts:
            return 0

        combined_summary = " | ".join(summary_parts)
        # Truncar si es muy largo
        if len(combined_summary) > self.config.summary_max_length:
            combined_summary = combined_summary[: self.config.summary_max_length - 3] + "..."

        # Remover turnos comprimidos de la memoria activa
        for tid in to_compress:
            self._turns.pop(tid, None)

        # Crear bloque comprimido
        first_id = to_compress[0]
        last_id = to_compress[-1]
        block = CompressedBlock(
            original_count=len(to_compress),
            summary=combined_summary,
            turn_range=(first_id, last_id),
            hash_digest=self._hash_turns(turns_data),
        )
        self._compressed_blocks.append(block)

        # Recalcular tokens
        self._update_token_estimate()
        tokens_after = 0
        for b in self._compressed_blocks:
            tokens_after += len(b.summary) // 3

        tokens_saved = max(0, tokens_before - tokens_after)
        self._total_compressions += 1
        self._total_tokens_saved += tokens_saved

        return tokens_saved

    def _summarize_turn(self, turn: _TurnData) -> str:
        """Genera un resumen de una linea de un turno."""
        parts = []

        if turn.user_msg and turn.user_msg.content:
            content = turn.user_msg.content.replace("\n", " ").strip()
            parts.append(f"User: {content[:120]}")

        if turn.assistant_msg:
            if turn.assistant_msg.content:
                content = turn.assistant_msg.content.replace("\n", " ").strip()
                parts.append(f"AI: {content[:120]}")
            if turn.assistant_msg.tool_calls:
                tools = [tc.get("function", {}).get("name", "?") for tc in turn.assistant_msg.tool_calls]
                parts.append(f"Tools: {', '.join(tools)}")

        if turn.tool_messages:
            success_count = sum(
                1 for tm in turn.tool_messages
                if tm.content and '"success": true' in tm.content
            )
            fail_count = len(turn.tool_messages) - success_count
            if success_count or fail_count:
                parts.append(f"Tools OK: {success_count}, Fail: {fail_count}")

        return " — ".join(parts) if parts else ""

    def _hash_turns(self, turns: list[_TurnData]) -> str:
        """Genera hash de un grupo de turnos para deduplicacion."""
        h = hashlib.md5()
        for turn in turns:
            if turn.user_msg:
                h.update(turn.user_msg.content.encode())
            if turn.assistant_msg:
                h.update((turn.assistant_msg.content or "").encode())
        return h.hexdigest()[:12]

    def _add_compressed_block(self, turn: _TurnData, summary: str) -> None:
        """Agrega un solo turno como bloque comprimido (usado por eviction)."""
        block = CompressedBlock(
            original_count=1,
            summary=summary,
            turn_range=(turn.turn_id, turn.turn_id),
            hash_digest=self._hash_turns([turn]),
        )
        self._compressed_blocks.append(block)

    def __len__(self) -> int:
        return len(self._turns)

    def __repr__(self) -> str:
        return (
            f"ShortTermMemory(turns={len(self._turns)}, "
            f"compressed={len(self._compressed_blocks)}, "
            f"tokens~{self._estimated_tokens})"
        )


# ---------------------------------------------------------------------------
# Internal data class
# ---------------------------------------------------------------------------


@dataclass
class _TurnData:
    """Datos de un turno completo de conversacion."""
    turn_id: int
    user_msg: Message | None = None
    assistant_msg: Message | None = None
    tool_messages: list[Message] = field(default_factory=list)
    added_at: float = field(default_factory=time.time)
