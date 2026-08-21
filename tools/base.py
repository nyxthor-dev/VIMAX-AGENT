"""Base tool definitions for the Cline agent."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """Resultado de la ejecucion de una herramienta."""

    success: bool
    output: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"success": self.success}
        if self.output:
            data["output"] = self.output
        if self.error:
            data["error"] = self.error
        return data

    def __str__(self) -> str:
        if self.success:
            return self.output or "(sin salida)"
        return f"ERROR: {self.error}"


@dataclass
class ToolParameter:
    """Definicion de un parametro de herramienta."""

    name: str
    type: str = "string"
    description: str = ""
    required: bool = True
    default: Any = None
    enum: list[str] | None = None


class BaseTool(ABC):
    """Clase base abstracta para todas las herramientas."""

    name: str = ""
    description: str = ""
    parameters: list[ToolParameter] = field(default_factory=list)

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Ejecuta la herramienta con los parametros dados."""
        ...

    def get_schema(self) -> dict[str, Any]:
        """Retorna el esquema JSON de la herramienta para la API."""
        properties = {}
        required = []
        for param in self.parameters:
            prop: dict[str, Any] = {"type": param.type, "description": param.description}
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        schema: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                },
            },
        }
        if required:
            schema["function"]["parameters"]["required"] = required
        return schema

    def validate(self, **kwargs: Any) -> list[str]:
        """Valida los parametros de entrada. Retorna lista de errores."""
        errors = []
        param_names = {p.name for p in self.parameters}

        # Check for unexpected parameters
        for key in kwargs:
            if key not in param_names:
                errors.append(f"Parametro desconocido: '{key}'")

        # Check required parameters
        for param in self.parameters:
            if param.required and param.name not in kwargs:
                if param.default is None:
                    errors.append(f"Parametro requerido faltante: '{param.name}'")

        # Check enum constraints
        for param in self.parameters:
            if param.enum and param.name in kwargs:
                value = kwargs[param.name]
                if value not in param.enum:
                    errors.append(
                        f"Valor '{value}' no valido para '{param.name}'. "
                        f"Valores permitidos: {param.enum}"
                    )

        return errors


class ToolRegistry:
    """Registro centralizado de herramientas disponibles."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Registra una herramienta."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        """Obtiene una herramienta por nombre."""
        return self._tools.get(name)

    def list_tools(self) -> list[BaseTool]:
        """Retorna todas las herramientas registradas."""
        return list(self._tools.values())

    def get_schemas(self) -> list[dict[str, Any]]:
        """Retorna los esquemas JSON de todas las herramientas."""
        return [tool.get_schema() for tool in self._tools.values()]

    def get_tool_names(self) -> list[str]:
        """Retorna los nombres de todas las herramientas."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
