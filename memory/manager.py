"""MemoryManager — Orquestacion de carga/guardado automatico de memoria.

Tareas: T-21 (carga automatica al inicio) + T-22 (guardado automatico al final).

Integra ShortTermMemory y LongTermMemory en una interfaz unificada que el
agente puede usar directamente. Se encarga de:

1. Cargar memoria del proyecto al iniciar sesion
2. Detectar convenciones del proyecto automaticamente
3. Generar resumen al finalizar la sesion
4. Detectar patrones error->fix durante la ejecucion
5. Inyectar contexto de memoria en el system prompt

Uso tipico (integrado en ClineAgent)::

    manager = MemoryManager(working_dir="/home/user/project")
    manager.start_session()
    # ... durante la ejecucion ...
    manager.record_tool_call("edit_file", success=True)
    manager.record_error_recovery(error, fix)
    # ... al finalizar ...
    manager.end_session()
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..api_client import Message

from .short_term import ShortTermMemory, ShortTermConfig
from .long_term import LongTermMemory, ProjectConventions


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class MemoryManagerConfig:
    """Configuracion del MemoryManager."""
    # Short term
    stm_max_turns: int = 20
    stm_max_tokens: int = 80_000
    stm_compress_threshold: float = 0.70
    # Long term
    auto_detect_conventions: bool = True
    max_patterns: int = 100
    auto_save_interval: int = 300  # segundos entre auto-saves


# ---------------------------------------------------------------------------
# Convention Detection
# ---------------------------------------------------------------------------

_LANGUAGE_PATTERNS: list[tuple[str, list[str]]] = [
    ("python", ["*.py", "requirements.txt", "setup.py", "pyproject.toml"]),
    ("typescript", ["*.ts", "tsconfig.json", "package.json"]),
    ("javascript", ["*.js", "package.json"]),
    ("go", ["*.go", "go.mod"]),
    ("rust", ["*.rs", "Cargo.toml"]),
    ("java", ["*.java", "pom.xml", "build.gradle"]),
]

_NAMING_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("snake_case", re.compile(r'^[a-z][a-z0-9_]*$')),
    ("camelCase", re.compile(r'^[a-z][a-zA-Z0-9]*$')),
    ("PascalCase", re.compile(r'^[A-Z][a-zA-Z0-9]*$')),
    ("kebab-case", re.compile(r'^[a-z][a-z0-9\-]*$')),
    ("UPPER_SNAKE", re.compile(r'^[A-Z][A-Z0-9_]*$')),
]

_TEST_FRAMEWORK_PATTERNS: list[tuple[str, list[str]]] = [
    ("pytest", ["conftest.py", "test_", "_test.py", "pytest.ini"]),
    ("unittest", ["unittest", "import unittest"]),
    ("jest", ["jest.config", "*.test.ts", "*.test.js", "*.spec.ts"]),
    ("vitest", ["vitest.config", "vite.config"]),
    ("go test", ["*_test.go"]),
]

_PACKAGE_MANAGER_PATTERNS: list[tuple[str, list[str]]] = [
    ("pip", ["requirements.txt", "setup.py", "pyproject.toml"]),
    ("poetry", ["poetry.lock", "pyproject.toml"]),
    ("npm", ["package-lock.json"]),
    ("pnpm", ["pnpm-lock.yaml"]),
    ("yarn", ["yarn.lock"]),
    ("go modules", ["go.mod"]),
    ("cargo", ["Cargo.toml", "Cargo.lock"]),
]


# ---------------------------------------------------------------------------
# MemoryManager
# ---------------------------------------------------------------------------


class MemoryManager:
    """Orquestador unificado de memoria del agente.

    Gestiona la carga automatica al inicio de sesion (T-21)
    y el guardado automatico al finalizar (T-22), ademas de
    deteccion de convenciones y patrones.
    """

    def __init__(
        self,
        working_dir: str = "",
        config: MemoryManagerConfig | None = None,
    ) -> None:
        self.config = config or MemoryManagerConfig()
        self.working_dir = working_dir or os.getcwd()

        # Sub-sistemas
        self.short_term = ShortTermMemory(
            config=ShortTermConfig(
                max_turns=self.config.stm_max_turns,
                max_tokens=self.config.stm_max_tokens,
                compress_ratio_threshold=self.config.stm_compress_threshold,
            )
        )
        self.long_term = LongTermMemory()

        # Tracking de la sesion actual
        self._session_active: bool = False
        self._session_start: float = 0
        self._session_actions: list[str] = []
        self._error_fix_pairs: list[tuple[str, str, str]] = []
        self._last_auto_save: float = 0

        # Cache del contexto inyectado
        self._cached_context: str = ""
        self._context_dirty: bool = True

    # ------------------------------------------------------------------
    # T-21: Carga automatica al inicio de sesion
    # ------------------------------------------------------------------

    def start_session(self) -> str:
        """Inicia una nueva sesion: carga memoria + detecta convenciones.

        Esto implementa T-21 completamente:
        1. Carga projectMemory del directorio actual
        2. Carga userPreferences globales
        3. Detecta convenciones del proyecto si es necesario
        4. Genera bloque de contexto para inyectar al LLM

        Returns:
            El bloque de contexto para inyectar en el system prompt.
        """
        self._session_active = True
        self._session_start = time.time()
        self._session_actions = ["Sesion iniciada"]
        self._error_fix_pairs = []

        # 1. Cargar memoria del proyecto
        self.long_term.load_project(self.working_dir)
        self.long_term.start_session()

        # 2. Detectar convenciones si es primera vez o se pide
        if self.config.auto_detect_conventions:
            pm = self.long_term.project_memory
            if pm and pm.conventions.language == "unknown":
                self._detect_conventions()

        # 3. Generar contexto
        self._context_dirty = True
        context = self.get_context_for_prompt()

        return context

    # ------------------------------------------------------------------
    # T-22: Guardado automatico al finalizar sesion
    # ------------------------------------------------------------------

    def end_session(self, final_response: str = "") -> None:
        """Finaliza la sesion: genera resumen y guarda todo.

        Esto implementa T-22 completamente:
        1. Genera resumen de la sesion a partir de las acciones
        2. Guarda patrones error->fix aprendidos
        3. Persiste toda la memoria a disco

        Args:
            final_response: Ultima respuesta del agente (opcional).
        """
        if not self._session_active:
            return

        # 1. Generar resumen de sesion
        summary = self._generate_session_summary(final_response)

        # 2. Guardar patrones aprendidos
        for error, fix, tool in self._error_fix_pairs:
            self.long_term.add_pattern(
                error_pattern=error,
                fix_description=fix,
                tool_name=tool,
            )

        # 3. Guardar todo
        self.long_term.end_session(summary=summary)

        self._session_active = False

    def auto_save_if_needed(self) -> None:
        """Guarda periodicamente si ha pasado suficiente tiempo."""
        if not self._session_active:
            return

        now = time.time()
        if now - self._last_auto_save >= self.config.auto_save_interval:
            self.long_term.save()
            self._last_auto_save = now

    # ------------------------------------------------------------------
    # API para el Agente: Registro de eventos
    # ------------------------------------------------------------------

    def add_turn(
        self,
        user_msg: Message | None = None,
        assistant_msg: Message | None = None,
        tool_messages: list[Message] | None = None,
    ) -> int:
        """Registra un turno completo en la memoria a corto plazo.

        Wrapper que ademas registra estadisticas de herramientas en LTM.
        """
        turn_id = self.short_term.add_turn(user_msg, assistant_msg, tool_messages)

        # Registrar tool calls en long_term stats
        if tool_messages:
            for tm in tool_messages:
                success = tm.content and '"success": true' in tm.content
                self.long_term.record_tool_call(
                    tool_name=tm.name or "unknown",
                    success=success,
                )

        return turn_id

    def get_context_for_llm(
        self,
        system_prompt: str = "",
    ) -> list[dict[str, Any]]:
        """Obtiene los mensajes optimizados para enviar al LLM.

        Inyecta el contexto de memoria a largo plazo como parte del
        system prompt.
        """
        enriched_prompt = system_prompt
        memory_context = self.get_context_for_prompt()
        if memory_context:
            separator = "\n\n" if system_prompt else ""
            enriched_prompt = f"{system_prompt}{separator}{memory_context}"

        return self.short_term.get_context_for_llm(
            system_prompt=enriched_prompt,
        )

    def get_context_for_prompt(self) -> str:
        """Genera el bloque de contexto de memoria para inyectar al prompt.

        Usa cache para evitar regenerar innecesariamente.
        """
        if not self._context_dirty and self._cached_context:
            return self._cached_context

        self._cached_context = self.long_term.get_context_injection()
        self._context_dirty = False
        return self._cached_context

    def record_action(self, action: str) -> None:
        """Registra una accion importante de la sesion."""
        self._session_actions.append(action)
        if len(self._session_actions) > 50:
            self._session_actions = self._session_actions[-50:]
        self._context_dirty = True

    def record_error_recovery(
        self,
        error_msg: str,
        fix_description: str,
        tool_name: str = "",
    ) -> None:
        """Registra un par error->fix para aprendizaje automatico."""
        self._error_fix_pairs.append((error_msg, fix_description, tool_name))
        self.long_term.record_error(error_msg, fix_description)

    def find_known_fix(self, error_msg: str, tool_name: str = "") -> str | None:
        """Busca un fix conocido para un error.

        Consulta primero patrones aprendidos, luego errores comunes.
        """
        # 1. Buscar en patrones aprendidos
        pattern = self.long_term.find_pattern(error_msg, tool_name)
        if pattern:
            return pattern.fix_description

        # 2. Buscar en errores comunes
        common = self.long_term.find_known_error(error_msg)
        if common:
            return common.solution

        return None

    def update_project_file_map(self, path: str, description: str) -> None:
        """Actualiza el mapa de archivos del proyecto."""
        self.long_term.add_file_to_map(path, description)
        self._context_dirty = True

    def mark_context_dirty(self) -> None:
        """Marca que el contexto necesita regenerarse."""
        self._context_dirty = True

    # ------------------------------------------------------------------
    # Convention Detection (Internal)
    # ------------------------------------------------------------------

    def _detect_conventions(self) -> None:
        """Detecta convenciones del proyecto escaneando archivos."""
        if not self.long_term.project_memory:
            return

        pm = self.long_term.project_memory
        cwd = Path(self.working_dir) if self.working_dir else Path.cwd()

        # Detectar lenguaje
        pm.conventions.language = self._detect_language(cwd)

        # Detectar package manager
        pm.conventions.package_manager = self._detect_package_manager(cwd)

        # Detectar test framework
        pm.conventions.test_framework = self._detect_test_framework(cwd)

        # Detectar naming convention
        pm.conventions.naming = self._detect_naming_convention(cwd)

        # Detectar entry points
        self._detect_entry_points(cwd, pm)

        # Detectar dependencias
        self._detect_dependencies(cwd, pm)

        # Marcar como actualizado
        pm.last_updated = time.time()

    def _detect_language(self, cwd: Path) -> str:
        """Detecta el lenguaje principal del proyecto."""
        from pathlib import Path

        extensions: dict[str, int] = {}

        try:
            for root, _dirs, files in os.walk(cwd, topdown=True):
                rel = os.path.relpath(root, cwd)
                skip_dirs = {
                    "node_modules", ".git", "__pycache__", ".venv",
                    "venv", "dist", "build", ".next", ".cache",
                }
                if any(part in skip_dirs for part in Path(rel).parts):
                    continue

                for f in files:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in (".py", ".ts", ".tsx", ".js", ".jsx",
                               ".go", ".rs", ".java", ".rb", ".php",
                               ".cpp", ".c", ".h", ".cs"):
                        extensions[ext] = extensions.get(ext, 0) + 1
        except PermissionError:
            pass

        if not extensions:
            return "unknown"

        ext_to_lang = {
            ".py": "python",
            ".ts": "typescript", ".tsx": "typescript",
            ".js": "javascript", ".jsx": "javascript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".rb": "ruby",
            ".php": "php",
            ".cpp": "c++", ".c": "c", ".h": "c/c++",
            ".cs": "c#",
        }

        most_common_ext = max(extensions, key=extensions.get)
        return ext_to_lang.get(most_common_ext, "unknown")

    def _detect_package_manager(self, cwd: Path) -> str:
        """Detecta el gestor de paquetes del proyecto."""
        for name, patterns in _PACKAGE_MANAGER_PATTERNS:
            for pattern in patterns:
                if pattern.startswith("*"):
                    continue
                if (cwd / pattern).exists():
                    return name
        return "unknown"

    def _detect_test_framework(self, cwd: Path) -> str:
        """Detecta el framework de tests del proyecto."""
        for name, patterns in _TEST_FRAMEWORK_PATTERNS:
            for pattern in patterns:
                if pattern.startswith("*"):
                    import glob as glob_mod
                    matches = glob_mod.glob(str(cwd / "**" / pattern), recursive=True)
                    if matches:
                        return name
                elif (cwd / pattern).exists():
                    return name
        return "unknown"

    def _detect_naming_convention(self, cwd: Path) -> str:
        """Detecta la convencion de nombres leyendo muestras de codigo."""
        from pathlib import Path

        code_files = []
        try:
            for root, _dirs, files in os.walk(cwd, topdown=True):
                rel = os.path.relpath(root, cwd)
                skip = {
                    "node_modules", ".git", "__pycache__", ".venv",
                    "venv", "dist", "build", ".next",
                }
                if any(part in skip for part in Path(rel).parts):
                    continue
                for f in files:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in (".py", ".ts", ".js", ".go", ".rs", ".java"):
                        code_files.append(os.path.join(root, f))
                if len(code_files) >= 10:
                    break
        except PermissionError:
            pass

        if not code_files:
            return "unknown"

        # Extraer identifiers
        identifiers: list[str] = []
        for fpath in code_files[:5]:
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                for pattern in [
                    r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)',
                    r'function\s+([a-zA-Z_][a-zA-Z0-9_]*)',
                    r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)',
                    r'const\s+([a-zA-Z_][a-zA-Z0-9_]*)',
                    r'let\s+([a-zA-Z_][a-zA-Z0-9_]*)',
                    r'var\s+([a-zA-Z_][a-zA-Z0-9_]*)',
                ]:
                    matches = re.findall(pattern, content)
                    identifiers.extend(matches)
            except Exception:
                continue

        if not identifiers:
            return "unknown"

        # Clasificar cada identifier
        scores: dict[str, int] = {}
        for ident in identifiers:
            if len(ident) < 2:
                continue
            for name, pat in _NAMING_PATTERNS:
                if pat.match(ident):
                    scores[name] = scores.get(name, 0) + 1
                    break

        if not scores:
            return "unknown"

        return max(scores, key=scores.get)

    def _detect_entry_points(self, cwd: Path, pm: Any) -> None:
        """Detecta puntos de entrada del proyecto."""
        candidates = [
            "main.py", "app.py", "index.py", "manage.py",
            "index.ts", "index.tsx", "main.ts", "server.ts",
            "index.js", "main.js", "server.js", "app.js",
            "main.go",
            "main.rs", "lib.rs",
            "Main.java",
        ]

        for candidate in candidates:
            if (cwd / candidate).exists():
                pm.entry_points.append(candidate)

        pkg = cwd / "package.json"
        if pkg.exists():
            try:
                import json as json_mod
                with open(pkg) as f:
                    data = json_mod.load(f)
                if "main" in data:
                    pm.entry_points.append(f"package.json#main={data['main']}")
            except Exception:
                pass

    def _detect_dependencies(self, cwd: Path, pm: Any) -> None:
        """Detecta dependencias principales del proyecto."""
        deps: list[str] = []

        # Python requirements.txt
        req = cwd / "requirements.txt"
        if req.exists():
            try:
                with open(req) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            dep_name = re.split(r'[><=!~\[]', line)[0].strip()
                            if dep_name:
                                deps.append(dep_name)
            except Exception:
                pass

        # package.json
        pkg = cwd / "package.json"
        if pkg.exists():
            try:
                import json as json_mod
                with open(pkg) as f:
                    data = json_mod.load(f)
                for key in ("dependencies", "devDependencies"):
                    for dep in data.get(key, {}):
                        deps.append(dep)
            except Exception:
                pass

        # Cargo.toml
        cargo = cwd / "Cargo.toml"
        if cargo.exists():
            try:
                with open(cargo) as f:
                    content = f.read()
                in_deps = False
                for line in content.split("\n"):
                    line = line.strip()
                    if line.startswith("[dependencies]"):
                        in_deps = True
                        continue
                    if line.startswith("["):
                        in_deps = False
                        continue
                    if in_deps and "=" in line:
                        dep_name = line.split("=")[0].strip()
                        deps.append(dep_name)
            except Exception:
                pass

        pm.dependencies = deps[:30]

    # ------------------------------------------------------------------
    # Session Summary Generation (T-22)
    # ------------------------------------------------------------------

    def _generate_session_summary(self, final_response: str = "") -> str:
        """Genera un resumen automatico de la sesion."""
        parts = []

        # Duracion
        duration = time.time() - self._session_start
        minutes = int(duration // 60)
        parts.append(f"Duracion: {minutes} min")

        # Acciones principales
        if self._session_actions:
            significant = [
                a for a in self._session_actions
                if a != "Sesion iniciada" and len(a) > 10
            ][:10]
            if significant:
                parts.append(f"Acciones: {'; '.join(significant)}")

        # Estadisticas de herramientas
        tool_summary = self.long_term.get_tool_stats_summary()
        if tool_summary["total_calls"] > 0:
            parts.append(
                f"Herramientas: {tool_summary['total_calls']} llamadas, "
                f"{tool_summary['success_rate']*100:.0f}% exito"
            )

        # Patrones aprendidos
        new_patterns = len(self._error_fix_pairs)
        if new_patterns > 0:
            parts.append(f"Patrones aprendidos: {new_patterns}")

        # Uso de memoria
        stm_stats = self.short_term.stats
        if stm_stats["total_turns_added"] > 0:
            parts.append(
                f"Turnos: {stm_stats['total_turns_added']}, "
                f"compresiones: {stm_stats['total_compressions']}"
            )

        # Fragmento de la respuesta final
        if final_response:
            snippet = final_response.replace("\n", " ").strip()[:200]
            parts.append(f"Ultima respuesta: {snippet}")

        return " | ".join(parts)

    # ------------------------------------------------------------------
    # Properties & Debug
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict[str, Any]:
        """Estadisticas completas del sistema de memoria."""
        return {
            "session_active": self._session_active,
            "session_duration_min": round(
                (time.time() - self._session_start) / 60, 1
            ) if self._session_active else 0,
            "short_term": self.short_term.stats,
            "long_term": {
                "project_loaded": self.long_term.is_loaded,
                "patterns": len(self.long_term.learned_patterns),
                "tool_stats": self.long_term.get_tool_stats_summary(),
                "project_conventions": (
                    self.long_term.project_memory.conventions.to_dict()
                    if self.long_term.project_memory else None
                ),
            },
            "pending_patterns": len(self._error_fix_pairs),
        }

    def __repr__(self) -> str:
        status = "active" if self._session_active else "inactive"
        return (
            f"MemoryManager({status}, "
            f"stm={self.short_term}, "
            f"ltm={self.long_term})"
        )
