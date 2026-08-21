"""Cliente API para comunicacion con el modelo de IA.

Maneja el formato de historial de mensajes, streaming, y llamadas a la API.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx

from .config import ModelConfig


@dataclass
class Message:
    """Un mensaje en el historial del chat."""
    role: str  # 'system', 'user', 'assistant', 'tool'
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: str | None = None  # para mensajes de tipo 'tool'
    name: str | None = None  # nombre de la herramienta (para mensajes tool)
    timestamp: float = field(default_factory=time.time)

    def to_api_dict(self) -> dict[str, Any]:
        """Convierte al formato esperado por la API de OpenAI."""
        msg: dict[str, Any] = {"role": self.role, "content": self.content or None}
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.name:
            msg["name"] = self.name
        return msg

    def to_display(self) -> str:
        """Formato legible para mostrar en la interfaz."""
        if self.role == "system":
            return f"[Sistema] {self.content[:100]}..."
        elif self.role == "user":
            return f"[Tu] {self.content}"
        elif self.role == "assistant":
            parts = []
            if self.content:
                parts.append(self.content)
            if self.tool_calls:
                for tc in self.tool_calls:
                    fn = tc.get("function", {})
                    args = fn.get("arguments", "{}")
                    try:
                        args_parsed = json.loads(args)
                        args_str = json.dumps(args_parsed, ensure_ascii=False)[:200]
                    except (json.JSONDecodeError, TypeError):
                        args_str = args[:200]
                    parts.append(f"  [Herramienta] {fn.get('name', '?')}({args_str})")
            return "[IA] " + "\n".join(parts)
        elif self.role == "tool":
            content_preview = (self.content or "")[:300]
            return f"  [Resultado {self.name}] {content_preview}"
        return str(self.content)


class ChatHistory:
    """Gestiona el historial de mensajes del chat con formato y compresion."""

    def __init__(self, system_prompt: str = "", max_messages: int = 50) -> None:
        self.system_prompt = system_prompt
        self.max_messages = max_messages
        self._messages: list[Message] = []

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    def add(self, message: Message) -> None:
        """Agrega un mensaje al historial."""
        self._messages.append(message)

    def add_user(self, content: str) -> Message:
        """Conveniencia: agrega un mensaje de usuario."""
        msg = Message(role="user", content=content)
        self._messages.append(msg)
        return msg

    def add_assistant(self, content: str = "", tool_calls: list[dict] | None = None) -> Message:
        """Conveniencia: agrega una respuesta del asistente."""
        msg = Message(role="assistant", content=content, tool_calls=tool_calls or [])
        self._messages.append(msg)
        return msg

    def add_tool_result(self, tool_call_id: str, name: str, content: str) -> Message:
        """Conveniencia: agrega un resultado de herramienta."""
        msg = Message(
            role="tool",
            tool_call_id=tool_call_id,
            name=name,
            content=content,
        )
        self._messages.append(msg)
        return msg

    def get_api_messages(self) -> list[dict[str, Any]]:
        """Retorna todos los mensajes en formato API, incluyendo el system prompt."""
        result = []
        if self.system_prompt:
            result.append({"role": "system", "content": self.system_prompt})
        for msg in self._messages:
            result.append(msg.to_api_dict())
        return result

    def get_last_tool_call_messages(self) -> list[dict[str, Any]]:
        """Retorna los mensajes desde la ultima llamada a herramienta."""
        # Find the last assistant message with tool calls
        start_idx = len(self._messages)
        for i in range(len(self._messages) - 1, -1, -1):
            if self._messages[i].role == "assistant" and self._messages[i].tool_calls:
                start_idx = i
                break
            elif self._messages[i].role == "user":
                start_idx = i
                break

        result = []
        if self.system_prompt:
            result.append({"role": "system", "content": self.system_prompt})
        for msg in self._messages[start_idx:]:
            result.append(msg.to_api_dict())
        return result

    def compress(self) -> int:
        """Comprime el historial manteniendo contexto clave.
        
        Estrategia: mantiene los primeros 4 mensajes (contexto inicial),
        los ultimos N mensajes, y resumen del medio.
        Retorna el numero de mensajes eliminados.
        """
        if len(self._messages) <= self.max_messages:
            return 0

        keep_start = 4  # primeros mensajes (contexto inicial)
        keep_end = self.max_messages - keep_start - 2  # ultimos mensajes

        if keep_end < 10:
            keep_end = 10

        removed = len(self._messages) - keep_start - keep_end
        if removed <= 0:
            return 0

        # Create a summary message of removed messages
        removed_msgs = self._messages[keep_start:keep_start + removed]
        summary_parts = [f"[Historial comprimido: {removed} mensajes omitidos]"]
        for msg in removed_msgs:
            if msg.role == "user" and msg.content:
                summary_parts.append(f"- Usuario: {msg.content[:100]}")
            elif msg.role == "assistant" and msg.content:
                summary_parts.append(f"- IA: {msg.content[:100]}")
            elif msg.tool_calls:
                for tc in msg.tool_calls:
                    fn = tc.get("function", {})
                    summary_parts.append(f"- Herramienta: {fn.get('name', '?')}")

        summary_msg = Message(
            role="assistant",
            content="\n".join(summary_parts),
        )

        self._messages = (
            self._messages[:keep_start]
            + [summary_msg]
            + self._messages[keep_start + removed:]
        )
        return removed

    def clear(self) -> None:
        """Limpia todo el historial."""
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        return f"ChatHistory({len(self._messages)} mensajes)"


class APIClient:
    """Cliente HTTP para la API de OpenAI-compatible."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        base_url = config.api_url.rstrip("/")
        self.chat_url = f"{base_url}/chat/completions"
        self.models_url = f"{base_url}/models"
        self._client = httpx.Client(
            timeout=120.0,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
        )

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Envia una peticion de chat completion a la API.
        
        Args:
            messages: Lista de mensajes en formato OpenAI.
            tools: Lista de herramientas disponibles (schema).
            temperature: Temperatura de generacion.
            max_tokens: Maximo de tokens a generar.
            stream: Si se usa streaming.
        
        Returns:
            Diccionario con la respuesta de la API.
        """
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            "top_p": self.config.top_p,
            "frequency_penalty": self.config.frequency_penalty,
            "presence_penalty": self.config.presence_penalty,
            "stream": stream,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        try:
            response = self._client.post(self.chat_url, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_detail = e.response.text
            except Exception:
                pass
            raise APIError(
                f"Error HTTP {e.response.status_code}: {error_detail}",
                status_code=e.response.status_code,
            ) from e
        except httpx.RequestError as e:
            raise APIError(f"Error de conexion: {e}") from e

    def chat_completion_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        """Envia una peticion de chat completion con streaming.
        
        Yield chunks de la respuesta.
        """
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            "stream": True,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        try:
            with self._client.stream("POST", self.chat_url, json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data.strip() == "[DONE]":
                            break
                        try:
                            yield json.loads(data)
                        except json.JSONDecodeError:
                            continue
        except httpx.HTTPStatusError as e:
            raise APIError(f"Error HTTP {e.response.status_code}") from e
        except httpx.RequestError as e:
            raise APIError(f"Error de conexion: {e}") from e

    def list_models(self) -> list[str]:
        """Lista los modelos disponibles en la API."""
        try:
            response = self._client.get(self.models_url)
            response.raise_for_status()
            data = response.json()
            models = []
            for m in data.get("data", []):
                models.append(m.get("id", ""))
            return sorted(models)
        except Exception:
            return []

    def test_connection(self) -> tuple[bool, str]:
        """Prueba la conexion con la API."""
        try:
            models = self.list_models()
            if models:
                return True, f"Conexion exitosa. {len(models)} modelos disponibles."
            return True, "Conexion exitosa pero no se encontraron modelos."
        except APIError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Error inesperado: {e}"

    def close(self) -> None:
        """Cierra el cliente HTTP."""
        self._client.close()


class APIError(Exception):
    """Error de la API."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message

    def __str__(self) -> str:
        if self.status_code:
            return f"APIError [{self.status_code}]: {self.message}"
        return f"APIError: {self.message}"
