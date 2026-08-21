"""Memoria a Largo Plazo — Persistencia por proyecto y preferencias de usuario.

Almacena conocimiento que sobrevive entre sesiones:
- ProjectMemory: arquitectura, convenciones, errores comunes por proyecto
- UserPreferences: idioma, estilo, convenciones globales del usuario
- LearnedPatterns: pares error->fix aprendidos automaticamente
- ToolUsageStats: estadisticas de uso de herramientas

Persistencia: JSON en ~/.cline-agent/memory/<project_hash>/

Tareas: T-20
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


# Base directory for all memory files
_MEMORY_BASE_DIR = Path.home() / ".cline-agent" / "memory"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class ProjectConventions:
    """Convenciones detectadas para un proyecto."""
    naming: str = "unknown"  # camelCase, snake_case, kebab-case, PascalCase
    test_framework: str = "unknown"  # pytest, jest, vitest, unittest...
    package_manager: str = "unknown"  # npm, pnpm, yarn, pip, poetry...
    language: str = "unknown"  # python, typescript, go, rust...
    indent_style: str = "unknown"  # spaces, tabs
    indent_size: int = 4
    quote_style: str = "unknown"  # single, double

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CommonError:
    """Un error frecuente y su solucion conocida."""
    pattern: str  # substring o regex del error
    solution: str  # como se resolvio
    occurrences: int = 1  # cuantas veces se ha visto
    last_seen: float = field(default_factory=time.time)


@dataclass
class ProjectMemory:
    """Memoria de un proyecto especifico.

    Se almacena un archivo project.json por directorio de trabajo.
    """
    project_path: str = ""
    architecture: str = ""  # resumen de la arquitectura detectada
    common_commands: list[str] = field(default_factory=list)  # npm run dev, pytest, etc.
    conventions: ProjectConventions = field(default_factory=ProjectConventions)
    common_errors: list[CommonError] = field(default_factory=list)
    last_session_summary: str = ""
    file_map: dict[str, str] = field(default_factory=dict)  # path -> descripcion breve
    entry_points: list[str] = field(default_factory=list)  # main.py, index.ts, etc.
    dependencies: list[str] = field(default_factory=list)  # deps principales detectadas
    last_updated: float = field(default_factory=time.time)
    session_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["conventions"] = asdict(self.conventions)
        data["common_errors"] = [asdict(e) for e in self.common_errors]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectMemory:
        conv_data = data.pop("conventions", {})
        errors_data = data.pop("common_errors", [])
        pm = cls(**data)
        pm.conventions = ProjectConventions(**conv_data)
        pm.common_errors = [CommonError(**e) for e in errors_data]
        return pm


@dataclass
class UserPreferences:
    """Preferencias globales del usuario.

    Se almacena un solo archivo global user_preferences.json.
    """
    preferred_language: str = "es"  # idioma de respuesta
    response_style: str = "concise"  # concise, detailed, technical
    auto_approve_tools: list[str] = field(default_factory=list)  # herramientas auto-aprobadas
    blocked_tools: list[str] = field(default_factory=list)
    preferred_model: str = ""
    custom_instructions: str = ""  # instrucciones adicionales del usuario
    theme: str = "dark"
    last_updated: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UserPreferences:
        return cls(**data)


@dataclass
class LearnedPattern:
    """Un patron aprendido: error -> fix.

    Se detecta automaticamente cuando:
    1. Una herramienta falla con error X
    2. El agente aplica correccion Y
    3. La correccion tiene exito
    """
    error_pattern: str  # substring del error
    fix_description: str  # que se hizo para arreglarlo
    tool_name: str = ""  # herramienta donde ocurrio
    file_pattern: str = ""  # patron de archivo (ej. *.py)
    success_count: int = 1  # cuantas veces funciono este fix
    fail_count: int = 0  # cuantas veces fallo
    last_applied: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    confidence: float = 1.0  # success_count / (success_count + fail_count)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LearnedPattern:
        return cls(**data)

    def update_confidence(self) -> None:
        """Recalcula la confianza del patron."""
        total = self.success_count + self.fail_count
        self.confidence = self.success_count / total if total > 0 else 0


@dataclass
class ToolUsageStat:
    """Estadisticas de uso de una herramienta."""
    tool_name: str
    call_count: int = 0
    success_count: int = 0
    fail_count: int = 0
    total_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    last_used: float = 0.0

    def record_call(self, success: bool, latency_ms: float = 0) -> None:
        """Registra una llamada a esta herramienta."""
        self.call_count += 1
        if success:
            self.success_count += 1
        else:
            self.fail_count += 1
        self.total_latency_ms += latency_ms
        self.avg_latency_ms = self.total_latency_ms / self.call_count
        self.last_used = time.time()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolUsageStat:
        return cls(**data)


# ---------------------------------------------------------------------------
# LongTermMemory
# ---------------------------------------------------------------------------


class LongTermMemory:
    """Gestor de memoria a largo plazo con persistencia JSON.

    Estructura en disco::

        ~/.cline-agent/memory/
        ├── user_preferences.json          # preferencias globales
        ├── <project_hash_1>/
        │   ├── project.json               # memoria del proyecto
        │   ├── patterns.json              # patrones aprendidos
        │   └── tool_stats.json            # estadisticas de herramientas
        ├── <project_hash_2>/
        │   └── ...
        └── ...

    Uso tipico::
        ltm = LongTermMemory()
        ltm.load_project("/home/user/myproject")
        ltm.add_pattern(error="SyntaxError", fix="check indentation")
        ltm.save()
    """

    def __init__(self, base_dir: Path | str | None = None) -> None:
        self.base_dir = Path(base_dir) if base_dir else _MEMORY_BASE_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Datos cargados
        self.user_preferences = UserPreferences()
        self.project_memory: ProjectMemory | None = None
        self.learned_patterns: list[LearnedPattern] = []
        self.tool_stats: dict[str, ToolUsageStat] = {}

        # Path del proyecto actual
        self._current_project_path: str = ""
        self._current_project_dir: Path | None = None

    # ------------------------------------------------------------------
    # Project loading
    # ------------------------------------------------------------------

    def load_project(self, project_path: str) -> ProjectMemory:
        """Carga la memoria de un proyecto especifico.

        Si no existe, crea una nueva. También carga las preferencias globales.

        Args:
            project_path: Ruta absoluta al directorio del proyecto.

        Returns:
            La ProjectMemory cargada o creada.
        """
        self._current_project_path = os.path.abspath(project_path)
        project_hash = self._hash_path(self._current_project_path)
        self._current_project_dir = self.base_dir / project_hash
        self._current_project_dir.mkdir(parents=True, exist_ok=True)

        # Cargar preferencias globales
        self._load_user_preferences()

        # Cargar memoria del proyecto
        project_file = self._current_project_dir / "project.json"
        if project_file.exists():
            try:
                with open(project_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.project_memory = ProjectMemory.from_dict(data)
                # Actualizar path por si el proyecto se movio
                self.project_memory.project_path = self._current_project_path
            except Exception:
                self.project_memory = ProjectMemory(
                    project_path=self._current_project_path
                )
        else:
            self.project_memory = ProjectMemory(
                project_path=self._current_project_path
            )

        # Cargar patrones aprendidos
        self._load_patterns()

        # Cargar estadisticas de herramientas
        self._load_tool_stats()

        return self.project_memory

    def save(self) -> None:
        """Guarda toda la memoria a disco."""
        if not self._current_project_dir:
            return

        self._save_user_preferences()
        self._save_project()
        self._save_patterns()
        self._save_tool_stats()

    # ------------------------------------------------------------------
    # User Preferences
    # ------------------------------------------------------------------

    def _load_user_preferences(self) -> None:
        """Carga preferencias globales del usuario."""
        pref_file = self.base_dir / "user_preferences.json"
        if pref_file.exists():
            try:
                with open(pref_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.user_preferences = UserPreferences.from_dict(data)
            except Exception:
                self.user_preferences = UserPreferences()

    def _save_user_preferences(self) -> None:
        """Guarda preferencias globales del usuario."""
        self.user_preferences.last_updated = time.time()
        pref_file = self.base_dir / "user_preferences.json"
        try:
            with open(pref_file, "w", encoding="utf-8") as f:
                json.dump(self.user_preferences.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Memory] Error guardando preferencias: {e}")

    # ------------------------------------------------------------------
    # Project Memory
    # ------------------------------------------------------------------

    def _save_project(self) -> None:
        """Guarda la memoria del proyecto actual."""
        if not self.project_memory:
            return

        self.project_memory.last_updated = time.time()
        project_file = self._current_project_dir / "project.json"
        try:
            with open(project_file, "w", encoding="utf-8") as f:
                json.dump(
                    self.project_memory.to_dict(),
                    f, indent=2, ensure_ascii=False,
                )
        except Exception as e:
            print(f"[Memory] Error guardando proyecto: {e}")

    def update_architecture(self, description: str) -> None:
        """Actualiza la descripcion de arquitectura del proyecto."""
        if self.project_memory:
            self.project_memory.architecture = description
            self._save_project()

    def add_common_command(self, command: str) -> None:
        """Agrega un comando comun al proyecto (si no existe ya)."""
        if not self.project_memory:
            return
        if command not in self.project_memory.common_commands:
            self.project_memory.common_commands.append(command)
            # Mantener maximo 20 comandos
            if len(self.project_memory.common_commands) > 20:
                self.project_memory.common_commands = (
                    self.project_memory.common_commands[-20:]
                )

    def add_entry_point(self, path: str) -> None:
        """Agrega un punto de entrada del proyecto."""
        if not self.project_memory:
            return
        if path not in self.project_memory.entry_points:
            self.project_memory.entry_points.append(path)

    def add_file_to_map(self, path: str, description: str) -> None:
        """Agrega un archivo al mapa del proyecto."""
        if not self.project_memory:
            return
        self.project_memory.file_map[path] = description
        # Mantener maximo 100 archivos
        if len(self.project_memory.file_map) > 100:
            # Evictar los mas antiguos (primeros en el dict)
            keys = list(self.project_memory.file_map.keys())
            for k in keys[: len(keys) - 100]:
                del self.project_memory.file_map[k]

    # ------------------------------------------------------------------
    # Common Errors
    # ------------------------------------------------------------------

    def record_error(self, error_msg: str, solution: str) -> None:
        """Registra un error comun y su solucion.

        Si el patron ya existe, incrementa occurrences y actualiza la solucion
        si es diferente.
        """
        if not self.project_memory:
            return

        # Buscar patron existente (substring match)
        pattern_key = error_msg[:100]  # usar primeros 100 chars como clave
        for existing in self.project_memory.common_errors:
            if existing.pattern in error_msg or error_msg in existing.pattern:
                existing.occurrences += 1
                existing.last_seen = time.time()
                # Actualizar solucion si la nueva es mas larga (mas detallada)
                if len(solution) > len(existing.solution):
                    existing.solution = solution
                return

        # Nuevo error
        self.project_memory.common_errors.append(
            CommonError(
                pattern=pattern_key,
                solution=solution,
                occurrences=1,
            )
        )
        # Mantener maximo 50 errores
        if len(self.project_memory.common_errors) > 50:
            self.project_memory.common_errors = sorted(
                self.project_memory.common_errors,
                key=lambda e: e.last_seen,
                reverse=True,
            )[:50]

    def find_known_error(self, error_msg: str) -> CommonError | None:
        """Busca un error conocido que coincida con el mensaje dado.

        Returns el CommonError si hay match, None si no.
        """
        if not self.project_memory:
            return None
        for err in self.project_memory.common_errors:
            if err.pattern in error_msg or error_msg[:100] in err.pattern:
                return err
        return None

    # ------------------------------------------------------------------
    # Learned Patterns (error -> fix)
    # ------------------------------------------------------------------

    def _load_patterns(self) -> None:
        """Carga patrones aprendidos del proyecto."""
        if not self._current_project_dir:
            return
        patterns_file = self._current_project_dir / "patterns.json"
        if patterns_file.exists():
            try:
                with open(patterns_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.learned_patterns = [
                    LearnedPattern.from_dict(p) for p in data
                ]
            except Exception:
                self.learned_patterns = []

    def _save_patterns(self) -> None:
        """Guarda patrones aprendidos del proyecto."""
        if not self._current_project_dir:
            return
        patterns_file = self._current_project_dir / "patterns.json"
        try:
            data = [p.to_dict() for p in self.learned_patterns]
            with open(patterns_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Memory] Error guardando patrones: {e}")

    def add_pattern(
        self,
        error_pattern: str,
        fix_description: str,
        tool_name: str = "",
        file_pattern: str = "",
    ) -> LearnedPattern:
        """Agrega un patron aprendido o actualiza uno existente.

        Si ya existe un patron con el mismo error_pattern y fix similar,
        incrementa success_count en vez de crear duplicado.

        Returns:
            El patron creado o actualizado.
        """
        now = time.time()

        # Buscar patron existente
        for p in self.learned_patterns:
            if (p.error_pattern == error_pattern[:100]
                    and p.tool_name == tool_name):
                p.success_count += 1
                p.last_applied = now
                if len(fix_description) > len(p.fix_description):
                    p.fix_description = fix_description
                p.update_confidence()
                return p

        # Nuevo patron
        pattern = LearnedPattern(
            error_pattern=error_pattern[:100],
            fix_description=fix_description,
            tool_name=tool_name,
            file_pattern=file_pattern,
            created_at=now,
            last_applied=now,
            confidence=1.0,
        )
        self.learned_patterns.append(pattern)

        # Mantener maximo 100 patrones
        if len(self.learned_patterns) > 100:
            self.learned_patterns = sorted(
                self.learned_patterns,
                key=lambda p: p.confidence * p.success_count,
                reverse=True,
            )[:100]

        return pattern

    def find_pattern(self, error_msg: str, tool_name: str = "") -> LearnedPattern | None:
        """Busca un patron aprendido que coincida con el error.

        Prioriza patrones con mayor confianza y que coincidan con la herramienta.

        Returns:
            El mejor patron encontrado, o None.
        """
        candidates = []
        error_lower = error_msg.lower()

        for p in self.learned_patterns:
            if p.error_pattern.lower() in error_lower:
                score = p.confidence * p.success_count
                # Bonus si coincide la herramienta
                if tool_name and p.tool_name == tool_name:
                    score *= 2.0
                candidates.append((score, p))

        if not candidates:
            return None

        # Retornar el de mayor score
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def record_pattern_failure(self, error_pattern: str) -> None:
        """Registra que un patron fallo (baja su confianza)."""
        for p in self.learned_patterns:
            if p.error_pattern in error_pattern:
                p.fail_count += 1
                p.update_confidence()
                return

    # ------------------------------------------------------------------
    # Tool Usage Stats
    # ------------------------------------------------------------------

    def _load_tool_stats(self) -> None:
        """Carga estadisticas de herramientas del proyecto."""
        if not self._current_project_dir:
            return
        stats_file = self._current_project_dir / "tool_stats.json"
        if stats_file.exists():
            try:
                with open(stats_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.tool_stats = {
                    name: ToolUsageStat.from_dict(stat)
                    for name, stat in data.items()
                }
            except Exception:
                self.tool_stats = {}

    def _save_tool_stats(self) -> None:
        """Guarda estadisticas de herramientas."""
        if not self._current_project_dir:
            return
        stats_file = self._current_project_dir / "tool_stats.json"
        try:
            data = {
                name: stat.to_dict()
                for name, stat in self.tool_stats.items()
            }
            with open(stats_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Memory] Error guardando tool stats: {e}")

    def record_tool_call(
        self,
        tool_name: str,
        success: bool,
        latency_ms: float = 0,
    ) -> None:
        """Registra una llamada a herramienta."""
        if tool_name not in self.tool_stats:
            self.tool_stats[tool_name] = ToolUsageStat(tool_name=tool_name)
        self.tool_stats[tool_name].record_call(success, latency_ms)

    def get_tool_stats_summary(self) -> dict[str, Any]:
        """Retorna resumen de estadisticas de herramientas."""
        total_calls = sum(s.call_count for s in self.tool_stats.values())
        total_success = sum(s.success_count for s in self.tool_stats.values())
        return {
            "total_calls": total_calls,
            "total_success": total_success,
            "total_fail": total_calls - total_success,
            "success_rate": round(total_success / total_calls, 3) if total_calls > 0 else 0,
            "tools": {
                name: {
                    "calls": s.call_count,
                    "success_rate": round(s.success_count / s.call_count, 3) if s.call_count > 0 else 0,
                    "avg_latency_ms": round(s.avg_latency_ms, 1),
                }
                for name, s in self.tool_stats.items()
            },
        }

    # ------------------------------------------------------------------
    # Session Management
    # ------------------------------------------------------------------

    def start_session(self) -> None:
        """Marca el inicio de una nueva sesion."""
        if self.project_memory:
            self.project_memory.session_count += 1

    def end_session(self, summary: str = "") -> None:
        """Marca el fin de la sesion y guarda todo.

        Args:
            summary: Resumen opcional de lo que se hizo en la sesion.
        """
        if self.project_memory:
            self.project_memory.last_session_summary = summary
            self.project_memory.last_updated = time.time()
        self.save()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_path(path: str) -> str:
        """Genera un hash corto y legible para un path de proyecto.

        Usa los ultimos 2 componentes del path + hash para evitar
        colisiones mientras mantiene legibilidad.
        """
        import hashlib
        parts = Path(path).parts
        # Tomar hasta los ultimos 2 componentes
        short_name = "_".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
        # Sanitizar
        safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in short_name)
        # Agregar hash corto para evitar colisiones
        h = hashlib.md5(path.encode()).hexdigest()[:6]
        return f"{safe}_{h}"

    def get_context_injection(self) -> str:
        """Genera un bloque de texto para inyectar en el system prompt.

        Incluye informacion relevante del proyecto y patrones aprendidos
        para que el agente pueda beneficiarse de la memoria a largo plazo.

        Returns:
            Bloque de texto formateado, o cadena vacia si no hay contexto.
        """
        parts = []

        if not self.project_memory:
            return ""

        pm = self.project_memory

        # Informacion del proyecto
        if pm.architecture:
            parts.append(f"Arquitectura del proyecto: {pm.architecture}")

        if pm.conventions.language != "unknown":
            parts.append(f"Lenguaje principal: {pm.conventions.language}")

        if pm.conventions.naming != "unknown":
            parts.append(f"Convencion de nombres: {pm.conventions.naming}")

        if pm.conventions.test_framework != "unknown":
            parts.append(f"Framework de tests: {pm.conventions.test_framework}")

        if pm.common_commands:
            parts.append(f"Comandos comunes: {', '.join(pm.common_commands[:10])}")

        if pm.entry_points:
            parts.append(f"Puntos de entrada: {', '.join(pm.entry_points[:5])}")

        # Session previa
        if pm.last_session_summary:
            parts.append(f"Resumen de sesion anterior: {pm.last_session_summary[:300]}")

        # Errores comunes conocidos
        if pm.common_errors:
            error_lines = []
            for err in pm.common_errors[:5]:
                error_lines.append(f"  - {err.pattern[:80]}: {err.solution[:100]}")
            parts.append("Errores frecuentes y sus soluciones:\n" + "\n".join(error_lines))

        # Patrones aprendidos (solo los de alta confianza)
        high_conf = [p for p in self.learned_patterns if p.confidence >= 0.7][:5]
        if high_conf:
            pattern_lines = []
            for p in high_conf:
                pattern_lines.append(
                    f"  - Si ocurre '{p.error_pattern[:60]}': {p.fix_description[:100]}"
                )
            parts.append("Patrones aprendidos (alta confianza):\n" + "\n".join(pattern_lines))

        if not parts:
            return ""

        return "[Memoria del Proyecto]\n" + "\n".join(parts)

    @property
    def is_loaded(self) -> bool:
        """True si hay un proyecto cargado."""
        return self.project_memory is not None

    def __repr__(self) -> str:
        project = self.project_memory.project_path if self.project_memory else "none"
        patterns = len(self.learned_patterns)
        tools = len(self.tool_stats)
        return f"LongTermMemory(project={project}, patterns={patterns}, tools={tools})"
