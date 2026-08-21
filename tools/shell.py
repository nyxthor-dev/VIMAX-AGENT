"""Herramienta de ejecucion de comandos de shell."""

from __future__ import annotations

import subprocess
from typing import Any

from .base import BaseTool, ToolParameter, ToolResult


class RunCommandTool(BaseTool):
    """Ejecuta un comando de shell/terminal."""

    name = "run_command"
    description = (
        "Ejecuta un comando de shell (bash) en el directorio de trabajo actual del agente. "
        "Retorna stdout y stderr. Ideal para: compilacion, tests, git, pip, npm, y otros comandos CLI. "
        "NO uses esto para operaciones de archivos (usa las herramientas de archivos en su lugar). "
        "El comando tiene un timeout de 120 segundos por defecto."
    )
    parameters = [
        ToolParameter(
            name="command",
            type="string",
            description="Comando a ejecutar (ej: 'python main.py', 'git status', 'npm install').",
            required=True,
        ),
        ToolParameter(
            name="timeout",
            type="integer",
            description="Timeout en segundos. Defecto: 120.",
            required=False,
            default=120,
        ),
        ToolParameter(
            name="working_dir",
            type="string",
            description="Directorio de trabajo. Si se omite, usa el directorio actual del agente.",
            required=False,
        ),
    ]

    def __init__(self, cwd_getter: callable) -> None:
        self._get_cwd = cwd_getter

    def execute(self, **kwargs: Any) -> ToolResult:
        command = kwargs.get("command", "")
        timeout = kwargs.get("timeout", 120)
        working_dir = kwargs.get("working_dir")

        if not command.strip():
            return ToolResult(success=False, error="Comando vacio")

        if not working_dir:
            working_dir = self._get_cwd()

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=working_dir,
                env={**__import__("os").environ},
            )

            output_parts = []
            if result.stdout:
                output_parts.append(result.stdout)
            if result.stderr:
                output_parts.append(f"[STDERR]\n{result.stderr}")

            combined_output = "\n".join(output_parts) if output_parts else "(sin salida)"

            # Truncate if too long
            max_output = 10000
            truncated = False
            if len(combined_output) > max_output:
                combined_output = combined_output[:max_output]
                truncated = True
                combined_output += f"\n\n... (salida truncada a {max_output} caracteres)"

            success = result.returncode == 0
            if success:
                return ToolResult(success=True, output=combined_output)
            else:
                error_msg = f"Codigo de salida: {result.returncode}"
                return ToolResult(success=False, error=error_msg, output=combined_output)

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error=f"Timeout: el comando excedio {timeout} segundos",
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))
