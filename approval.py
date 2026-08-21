"""Sistema de Auto-Approve y Clasificacion de Seguridad.

Inspirado en Cline:
- Clasifica cada accion del agente (lectura, escritura, comando, navegador)
- Permite auto-approve por categoria
- Comandos tienen flag requires_approval
- YOLO mode desactiva TODOS los checks
- Notificaciones cuando se requiere aprobacion manual
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# Patrones de comandos que SIEMPRE requieren aprobacion (peligrosos)
DANGEROUS_PATTERNS: list[tuple[str, str]] = [
    (r"\brm\s+(-[rfRF]+\s+)?/", "Borrado recursivo desde raiz"),
    (r"\brm\s+(-[rfRF]+\s+.*)?\*", "Borrado con wildcard"),
    (r"\bmkfs\b", "Formateo de filesystem"),
    (r"\bdd\s+if=", "Escritura directa a disco (dd)"),
    (r"\bchmod\s+777\b", "Permisos 777"),
    (r"\b(chgrp|chown)\s+.*-R", "Cambio de propietario recursivo"),
    (r"\b(git\s+push\s+--force|git\s+push\s+-f)(?!\s+with-lease)", "Force push sin lease"),
    (r"\bgit\s+reset\s+--hard", "Git reset hard"),
    (r"\bDROP\s+(TABLE|DATABASE|SCHEMA)", "DROP SQL"),
    (r"\bDELETE\s+FROM\s+\w+\s*;\s*$", "DELETE SQL sin WHERE"),
    (r"\bTRUNCATE\s+TABLE", "TRUNCATE SQL"),
    (r"\bkubectl\s+delete", "Kubernetes delete"),
    (r"\bdocker\s+rm\s+-f", "Docker force remove"),
    (r"\bdocker\s+system\s+prune", "Docker system prune"),
    (r"\bsudo\s+", "Comando con sudo"),
    (r":\(\)\s*\{.*\}\s*;\s*:", "Fork bomb"),
    (r"\b(shutdown|reboot|halt|poweroff)\b", "Apagado del sistema"),
    (r">\s*/dev/(sd|nvme|vd)", "Escritura directa a dispositivo de bloque"),
    (r"\bcurl.*\|\s*(ba)?sh", "Pipe de curl a shell"),
    (r"\bwget.*\|\s*(ba)?sh", "Pipe de wget a shell"),
    (r"\b(npm|yarn|pnpm)\s+publish", "Publicacion de paquete"),
    (r"\bcargo\s+publish", "Publicacion de crate"),
    (r"\bpip\s+install.*--user", "Instalacion pip global"),
]

# Patrones de comandos que son seguros (solo lectura o no destructivos)
SAFE_PATTERNS: list[str] = [
    r"^\s*(ls|dir|ll|la)\b",
    r"^\s*(cat|head|tail|less|more|wc)\b",
    r"^\s*(echo|print|printf)\b",
    r"^\s*(pwd|whoami|hostname|uname|date|cal)\b",
    r"^\s*(which|whereis|type|command)\b",
    r"^\s*(git\s+(status|log|diff|branch|remote|tag|stash\s+list|show))\b",
    r"^\s*(python|python3)\s+(-c|m|\S+\.py)",
    r"^\s*(node|deno|bun)\s+(--version|-e)",
    r"^\s*(npm|yarn|pnpm)\s+(run\s+(test|lint|build|check|typecheck)|test|lint|build|check)\b",
    r"^\s*(pip|pip3)\s+(list|show|check|freeze)\b",
    r"^\s*(go\s+)?(go\s+test|go\s+vet|go\s+build)\b",
    r"^\s*(cargo\s+)?(cargo\s+test|cargo\s+check|cargo\s+clippy)\b",
    r"^\s*find\s+.+-name\b",
    r"^\s*grep\b",
    r"^\s*git\s+rev-parse\b",
    r"^\s*file\s+",
    r"^\s*env\b",
    r"^\s*printenv\b",
]


@dataclass
class ApprovalResult:
    """Resultado de la clasificacion de una accion."""
    action_type: str  # "read", "write", "command", "browser", "mcp"
    auto_approved: bool
    requires_approval: bool
    reason: str = ""
    danger_level: str = "safe"  # "safe", "caution", "dangerous"


@dataclass
class ApprovalConfig:
    """Configuracion de auto-approve por categoria."""
    auto_approve_reads: bool = True  # Lecturas de archivos
    auto_approve_writes: bool = False  # Escrituras de archivos
    auto_approve_commands: bool = False  # Comandos de shell
    auto_approve_browser: bool = False  # Acciones de navegador
    auto_approve_mcp: bool = False  # Herramientas MCP
    yolo_mode: bool = False  # Desactiva TODOS los checks
    always_allow_commands: list[str] = field(default_factory=lambda: [
        "npm test", "npm run test", "npm run lint", "npm run build",
        "pytest", "python -m pytest", "python3 -m pytest",
        "cargo test", "cargo check", "go test", "go vet",
        "git status", "git diff", "git log",
        "pip list", "node --version", "python --version",
    ])


class ApprovalManager:
    """Gestor de aprobaciones para las acciones del agente.

    Clasifica cada accion del agente y determina si necesita aprobacion manual.
    """

    # Herramientas de solo lectura
    READ_TOOLS = {"read_file", "cat", "ls", "pwd", "glob", "find", "grep",
                   "search_files", "list_code_definition_names"}
    # Herramientas de escritura
    WRITE_TOOLS = {"write_file", "edit_file", "mkdir", "touch", "rm"}
    # Herramienta de shell
    COMMAND_TOOL = "run_command"
    # Herramientas interactivas
    AGENT_TOOLS = {"ask_followup_question", "attempt_completion"}

    def __init__(self, config: ApprovalConfig | None = None) -> None:
        self.config = config or ApprovalConfig()

    def classify_tool_call(
        self, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> ApprovalResult:
        """Clasifica una llamada a herramienta.

        Args:
            tool_name: Nombre de la herramienta.
            arguments: Argumentos de la llamada (para comandos).

        Returns:
            ApprovalResult con la clasificacion.
        """
        # YOLO mode aprueba todo
        if self.config.yolo_mode:
            return ApprovalResult(
                action_type=self._get_action_type(tool_name),
                auto_approved=True,
                requires_approval=False,
                reason="YOLO mode activo",
                danger_level="safe",
            )

        # Herramientas de lectura
        if tool_name in self.READ_TOOLS:
            approved = self.config.auto_approve_reads
            return ApprovalResult(
                action_type="read",
                auto_approved=approved,
                requires_approval=not approved,
                reason="Lectura de archivo" if approved else "Lectura requiere aprobacion",
                danger_level="safe",
            )

        # Herramientas de escritura
        if tool_name in self.WRITE_TOOLS:
            approved = self.config.auto_approve_writes
            danger = self._assess_write_danger(tool_name, arguments)
            if danger == "dangerous":
                approved = False  # Sobreescribir auto-approve para operaciones peligrosas
            return ApprovalResult(
                action_type="write",
                auto_approved=approved,
                requires_approval=not approved,
                reason=self._write_reason(tool_name, danger, approved),
                danger_level=danger,
            )

        # Comando de shell - requiere clasificacion especial
        if tool_name == self.COMMAND_TOOL:
            return self._classify_command(arguments or {})

        # Herramientas interactivas (siempre auto-aprobadas)
        if tool_name in self.AGENT_TOOLS:
            return ApprovalResult(
                action_type="agent",
                auto_approved=True,
                requires_approval=False,
                reason="Herramienta interactiva del agente",
                danger_level="safe",
            )

        # Navegador
        if tool_name == "browser_action":
            approved = self.config.auto_approve_browser
            return ApprovalResult(
                action_type="browser",
                auto_approved=approved,
                requires_approval=not approved,
                reason="Accion de navegador",
                danger_level="caution",
            )

        # MCP tools
        if tool_name.startswith("mcp_"):
            approved = self.config.auto_approve_mcp
            return ApprovalResult(
                action_type="mcp",
                auto_approved=approved,
                requires_approval=not approved,
                reason="Herramienta MCP",
                danger_level="caution",
            )

        # CD tool (cambio de directorio)
        if tool_name == "cd":
            return ApprovalResult(
                action_type="read",
                auto_approved=True,
                requires_approval=False,
                reason="Cambio de directorio",
                danger_level="safe",
            )

        # Default: requiere aprobacion
        return ApprovalResult(
            action_type="unknown",
            auto_approved=False,
            requires_approval=True,
            reason=f"Herramienta desconocida: {tool_name}",
            danger_level="caution",
        )

    def _classify_command(self, arguments: dict[str, Any]) -> ApprovalResult:
        """Clasifica un comando de shell."""
        command = arguments.get("command", "")
        stripped = command.strip()

        # 1. Verificar si esta en la lista de siempre permitidos
        if stripped in self.config.always_allow_commands:
            return ApprovalResult(
                action_type="command",
                auto_approved=True,
                requires_approval=False,
                reason=f"Comando en lista de permitidos: {stripped[:50]}",
                danger_level="safe",
            )

        # 2. Verificar patrones seguros
        for pattern in SAFE_PATTERNS:
            if re.match(pattern, stripped):
                if self.config.auto_approve_commands:
                    return ApprovalResult(
                        action_type="command",
                        auto_approved=True,
                        requires_approval=False,
                        reason=f"Comando seguro (auto-approve): {stripped[:50]}",
                        danger_level="safe",
                    )
                return ApprovalResult(
                    action_type="command",
                    auto_approved=False,
                    requires_approval=True,
                    reason=f"Comando seguro pero auto-approve desactivado: {stripped[:50]}",
                    danger_level="safe",
                )

        # 3. Verificar patrones peligrosos
        for pattern, desc in DANGEROUS_PATTERNS:
            if re.search(pattern, stripped):
                return ApprovalResult(
                    action_type="command",
                    auto_approved=False,
                    requires_approval=True,
                    reason=f"PELIGROSO - {desc}: {stripped[:80]}",
                    danger_level="dangerous",
                )

        # 4. Comando no clasificado - requiere aprobacion
        return ApprovalResult(
            action_type="command",
            auto_approved=self.config.auto_approve_commands,
            requires_approval=not self.config.auto_approve_commands,
            reason=f"Comando no clasificado: {stripped[:80]}",
            danger_level="caution",
        )

    def _assess_write_danger(
        self, tool_name: str, arguments: dict[str, Any] | None
    ) -> str:
        """Evalua el nivel de peligro de una operacion de escritura."""
        if tool_name == "rm":
            path = arguments.get("path", "") if arguments else ""
            if path == "/" or path == "~":
                return "dangerous"
            if "*" in path:
                return "dangerous"
            return "caution"

        if tool_name == "write_file":
            file_path = arguments.get("file_path", "") if arguments else ""
            # Archivos de configuracion del sistema
            dangerous_paths = ["/etc/", "/usr/", "/bin/", "/sbin/", "/boot/"]
            for dp in dangerous_paths:
                if file_path.startswith(dp):
                    return "dangerous"
            return "safe"

        if tool_name == "edit_file":
            return "safe"  # Ediciones parciales son inherentemente mas seguras

        return "safe"

    def _write_reason(
        self, tool_name: str, danger: str, approved: bool
    ) -> str:
        """Genera razon legible para operaciones de escritura."""
        action_map = {
            "write_file": "Escritura de archivo",
            "edit_file": "Edicion de archivo",
            "mkdir": "Creacion de directorio",
            "touch": "Creacion de archivo",
            "rm": "Eliminacion de archivo/directorio",
        }
        action = action_map.get(tool_name, tool_name)

        if danger == "dangerous":
            return f"PELIGROSO - {action} requiere aprobacion obligatoria"
        if approved:
            return f"{action} (auto-aprobada)"
        return f"{action} requiere aprobacion"

    def _get_action_type(self, tool_name: str) -> str:
        if tool_name in self.READ_TOOLS:
            return "read"
        if tool_name in self.WRITE_TOOLS:
            return "write"
        if tool_name == self.COMMAND_TOOL:
            return "command"
        if tool_name == "browser_action":
            return "browser"
        if tool_name.startswith("mcp_"):
            return "mcp"
        return "unknown"
