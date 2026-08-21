"""Herramientas de busqueda y analisis de codigo."""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Any, Callable

from .base import BaseTool, ToolParameter, ToolResult


# Patrones para extraccion de definiciones por lenguaje
DEFINITION_PATTERNS: dict[str, list[tuple[str, str]]] = {
    ".py": [
        (r"^(\s*)class\s+(\w+)", "class"),
        (r"^(\s*)def\s+(\w+)", "function"),
        (r"^(\s*)async\s+def\s+(\w+)", "function"),
    ],
    ".js": [
        (r"^(\s*)(?:export\s+)?(?:default\s+)?function\s+(\w+)", "function"),
        (r"^(\s*)(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(", "function"),
        (r"^(\s*)(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\\{", "variable"),
        (r"^(\s*)class\s+(\w+)", "class"),
    ],
    ".ts": [
        (r"^(\s*)(?:export\s+)?(?:default\s+)?function\s+(\w+)", "function"),
        (r"^(\s*)(?:export\s+)?(?:const|let|var)\s+(\w+)\s*[:=]\s*(?:async\s*)?\(", "function"),
        (r"^(\s*)(?:export\s+)?(?:interface|type)\s+(\w+)", "type"),
        (r"^(\s*)(?:export\s+)?(?:const|let|var)\s+(\w+)\s*[:=]", "variable"),
        (r"^(\s*)class\s+(\w+)", "class"),
    ],
    ".java": [
        (r"^(\s*)(?:public|private|protected)\s+(?:static\s+)?(?:abstract\s+)?(?:final\s+)?class\s+(\w+)", "class"),
        (r"^(\s*)(?:public|private|protected)\s+(?:static\s+)?(?:\w+\s+)+(\w+)\s*\(", "method"),
        (r"^(\s*)(?:public|private|protected)\s+(?:static\s+)?(?:final\s+)?(\w+)\s+(\w+)\s*[;=]", "variable"),
    ],
    ".go": [
        (r"^(\s*)func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)", "function"),
        (r"^(\s*)type\s+(\w+)\s+struct", "type"),
        (r"^(\s*)type\s+(\w+)\s+interface", "type"),
        (r"^(\s*)var\s+(\w+)", "variable"),
        (r"^(\s*)const\s+(\w+)", "variable"),
    ],
    ".rs": [
        (r"^(\s*)fn\s+(\w+)", "function"),
        (r"^(\s*)struct\s+(\w+)", "type"),
        (r"^(\s*)enum\s+(\w+)", "type"),
        (r"^(\s*)trait\s+(\w+)", "type"),
        (r"^(\s*)impl\s+(\w+)", "type"),
        (r"^(\s*)(?:pub\s+)?(?:const|static)\s+(\w+)", "variable"),
    ],
    ".rb": [
        (r"^(\s*)def\s+(\w+)", "method"),
        (r"^(\s*)class\s+(\w+)", "class"),
        (r"^(\s*)module\s+(\w+)", "module"),
    ],
    ".php": [
        (r"^(\s*)function\s+(\w+)", "function"),
        (r"^(\s*)class\s+(\w+)", "class"),
        (r"^(\s*)(?:public|private|protected)\s+(?:static\s+)?function\s+(\w+)", "method"),
    ],
    ".c": [
        (r"^(\s*)(?:static\s+)?(?:inline\s+)?(\w+(?:\s*\*)?)\s+(\w+)\s*\(", "function"),
        (r"^(\s*)typedef\s+(?:struct|enum|union)\s*\{?", "type"),
        (r"^(\s*)(?:struct|enum|union)\s+(\w+)", "type"),
    ],
    ".cpp": [
        (r"^(\s*)(?:(?:virtual|static|inline|explicit)\s+)*(\w+(?:\s*\*)?)\s+(\w+)\s*\(", "function"),
        (r"^(\s*)class\s+(\w+)", "class"),
        (r"^(\s*)struct\s+(\w+)", "type"),
        (r"^(\s*)namespace\s+(\w+)", "namespace"),
    ],
    ".cs": [
        (r"^(\s*)(?:public|private|protected|internal)\s+(?:static\s+)?(?:virtual\s+)?(?:override\s+)?(\w+)\s+(\w+)\s*\(", "method"),
        (r"^(\s*)(?:public|private|protected|internal)\s+(?:static\s+)?(?:abstract\s+)?(?:partial\s+)?class\s+(\w+)", "class"),
        (r"^(\s*)(?:public|private|protected)\s+(?:const|readonly)\s+(\w+)\s+(\w+)", "variable"),
    ],
}


class GrepTool(BaseTool):
    """Busca con regex en archivos."""

    name = "grep"
    description = (
        "Busca un patron regex en archivos. Retorna las lineas coincidentes "
        "con numeros de linea y nombre de archivo."
    )
    parameters = [
        ToolParameter("pattern", "string", "Patron regex de busqueda", required=True),
        ToolParameter("path", "string", "Directorio o archivo donde buscar (por defecto: cwd)", required=False),
        ToolParameter("file_pattern", "string", "Filtro glob para archivos (ej: '*.py'). Por defecto: '*'", required=False, default="*"),
        ToolParameter("ignore_case", "boolean", "Busqueda insensible a mayusculas/minusculas. Por defecto: true", required=False, default=True),
        ToolParameter("max_results", "integer", "Maximo de resultados. Por defecto: 50", required=False, default=50),
    ]

    def __init__(self, cwd_getter: Callable[[], str]) -> None:
        super().__init__()
        self._cwd_getter = cwd_getter

    def execute(self, **kwargs: Any) -> ToolResult:
        pattern = kwargs["pattern"]
        path = kwargs.get("path", self._cwd_getter())
        file_pattern = kwargs.get("file_pattern", "*")
        ignore_case = kwargs.get("ignore_case", True)
        max_results = kwargs.get("max_results", 50)

        try:
            flags = re.IGNORECASE if ignore_case else 0
            regex = re.compile(pattern, flags)
        except re.error as e:
            return ToolResult(False, error=f"Regex invalido: {e}")

        base = Path(path)
        if not base.exists():
            return ToolResult(False, error=f"Ruta no existe: {path}")

        # Si es un archivo, buscar solo en ese archivo
        if base.is_file():
            files_to_search = [base]
        else:
            # Filtrar por glob
            files_to_search = sorted(base.rglob(file_pattern))

        results = []
        matches = 0

        for fpath in files_to_search:
            if matches >= max_results:
                break
            if not fpath.is_file():
                continue
            # Ignorar directorios ocultos y binarios comunes
            parts = fpath.parts
            if any(p.startswith(".") and p not in (".", "..") for p in parts):
                continue
            if fpath.suffix in (".pyc", ".pyo", ".so", ".dll", ".exe", ".zip", ".tar", ".gz", ".png", ".jpg", ".gif", ".ico"):
                continue

            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")
                for i, line in enumerate(content.splitlines(), 1):
                    if regex.search(line):
                        rel = fpath.relative_to(Path(self._cwd_getter())) if fpath.is_relative_to(Path(self._cwd_getter())) else fpath
                        results.append(f"{rel}:{i}: {line.strip()}")
                        matches += 1
                        if matches >= max_results:
                            break
            except (PermissionError, OSError):
                continue

        if not results:
            return ToolResult(True, output="No se encontraron coincidencias.")

        output = f"Se encontraron {matches} coincidencia(s):\n" + "\n".join(results)
        if matches >= max_results:
            output += f"\n(Resultado truncado a {max_results} lineas. Usa max_results para ver mas.)"
        return ToolResult(True, output=output)


class FindTool(BaseTool):
    """Encuentra archivos/directorios por nombre."""

    name = "find"
    description = (
        "Encuentra archivos y directorios por nombre usando patrones glob. "
        "Soporta filtros por tipo (file, dir, any) y profundidad maxima."
    )
    parameters = [
        ToolParameter("pattern", "string", "Patron de busqueda (ej: '*.py', 'test_*')", required=True),
        ToolParameter("path", "string", "Directorio donde buscar (por defecto: cwd)", required=False),
        ToolParameter("file_type", "string", "Tipo de resultado: 'file', 'dir', 'any'. Por defecto: 'any'", required=False, default="any", enum=["file", "dir", "any"]),
        ToolParameter("max_depth", "integer", "Profundidad maxima de busqueda. Por defecto: 10", required=False, default=10),
    ]

    def __init__(self, cwd_getter: Callable[[], str]) -> None:
        super().__init__()
        self._cwd_getter = cwd_getter

    def execute(self, **kwargs: Any) -> ToolResult:
        pattern = kwargs["pattern"]
        path = kwargs.get("path", self._cwd_getter())
        file_type = kwargs.get("file_type", "any")
        max_depth = kwargs.get("max_depth", 10)

        base = Path(path)
        if not base.exists():
            return ToolResult(False, error=f"Ruta no existe: {path}")

        results = []
        try:
            for match in base.rglob(pattern):
                # Verificar profundidad
                try:
                    rel = match.relative_to(base)
                    if len(rel.parts) > max_depth:
                        continue
                except ValueError:
                    continue

                # Filtrar tipo
                if file_type == "file" and not match.is_file():
                    continue
                if file_type == "dir" and not match.is_dir():
                    continue

                # Formatear salida
                kind = "DIR " if match.is_dir() else "FILE"
                size = ""
                if match.is_file():
                    try:
                        sz = match.stat().st_size
                        size = f" ({_human_size(sz)})"
                    except OSError:
                        pass
                rel_path = match.relative_to(base)
                results.append(f"[{kind}] {rel_path}{size}")
        except PermissionError as e:
            return ToolResult(False, error=f"Sin permisos: {e}")

        if not results:
            return ToolResult(True, output=f"No se encontraron resultados para '{pattern}' en {path}")

        return ToolResult(True, output=f"Se encontraron {len(results)} resultado(s):\n" + "\n".join(results))


class GlobTool(BaseTool):
    """Busca archivos con patrones glob."""

    name = "glob"
    description = "Busca archivos usando patrones glob (ej: '**/*.py', 'src/**/*.ts'). Muestra tipo y tamano."
    parameters = [
        ToolParameter("pattern", "string", "Patron glob de busqueda", required=True),
        ToolParameter("path", "string", "Directorio base (por defecto: cwd)", required=False),
    ]

    def __init__(self, cwd_getter: Callable[[], str]) -> None:
        super().__init__()
        self._cwd_getter = cwd_getter

    def execute(self, **kwargs: Any) -> ToolResult:
        pattern = kwargs["pattern"]
        path = kwargs.get("path", self._cwd_getter())

        base = Path(path)
        if not base.exists():
            return ToolResult(False, error=f"Ruta no existe: {path}")

        results = []
        try:
            for match in sorted(base.glob(pattern)):
                kind = "DIR " if match.is_dir() else "FILE"
                size = ""
                if match.is_file():
                    try:
                        size = f" ({_human_size(match.stat().st_size)})"
                    except OSError:
                        pass
                rel = match.relative_to(base)
                results.append(f"[{kind}] {rel}{size}")
        except Exception as e:
            return ToolResult(False, error=f"Error en glob: {e}")

        if not results:
            return ToolResult(True, output=f"Sin resultados para '{pattern}'")

        return ToolResult(True, output="\n".join(results))


class ListCodeDefinitionsTool(BaseTool):
    """Extrae nombres de definiciones (funciones, clases, etc) de archivos de codigo.

    Herramienta inspirada en Cline's list_code_definition_names - permite al agente
    entender la estructura de un archivo sin leerlo completo, ahorrando tokens.
    Soporta Python, JS, TS, Java, Go, Rust, Ruby, PHP, C, C++, C#.
    """

    name = "list_code_definition_names"
    description = (
        "Extrae todos los nombres de definiciones (clases, funciones, metodos, tipos, "
        "variables) de uno o mas archivos de codigo. Retorna una lista estructurada con "
        "el tipo de definicion, nombre, linea y archivo. Soporta Python, JS, TS, Java, Go, "
        "Rust, Ruby, PHP, C, C++, C#. Ideale para entender la estructura de un archivo "
        "sin leerlo completo, ahorrando tokens."
    )
    parameters = [
        ToolParameter(
            "path",
            "string",
            "Ruta al archivo o directorio. Si es directorio, escanea recursivamente "
            "todos los archivos de codigo soportados. Si es archivo, solo ese archivo.",
            required=True,
        ),
        ToolParameter(
            "file_pattern",
            "string",
            "Filtro glob adicional para archivos (ej: '*.py'). Solo aplica si path es un directorio. "
            "Por defecto: '*.py'",
            required=False,
            default="*.py",
        ),
    ]

    def __init__(self, cwd_getter: Callable[[], str]) -> None:
        super().__init__()
        self._cwd_getter = cwd_getter

    def execute(self, **kwargs: Any) -> ToolResult:
        path_str = kwargs["path"]
        file_pattern = kwargs.get("file_pattern", "*.py")

        target = Path(path_str)
        if not target.is_absolute():
            target = Path(self._cwd_getter()) / target

        if not target.exists():
            return ToolResult(False, error=f"Ruta no existe: {path_str}")

        # Determinar archivos a escanear
        if target.is_file():
            files = [target]
        else:
            files = sorted(target.rglob(file_pattern))
            files = [f for f in files if f.is_file()]

        if not files:
            return ToolResult(True, output="No se encontraron archivos de codigo para escanear.")

        all_definitions = []
        cwd = Path(self._cwd_getter())

        for fpath in files[:100]:  # Limitar a 100 archivos
            rel = fpath.relative_to(cwd) if fpath.is_relative_to(cwd) else fpath
            suffix = fpath.suffix.lower()
            patterns = DEFINITION_PATTERNS.get(suffix, [])

            if not patterns:
                # Intentar con AST para Python
                if suffix == ".py":
                    defs = self._extract_python_ast(fpath)
                    all_definitions.extend(defs)
                continue

            try:
                lines = fpath.read_text(encoding="utf-8", errors="replace").splitlines()
            except (PermissionError, OSError):
                continue

            for line_num, line in enumerate(lines, 1):
                # Ignorar comentarios y strings
                stripped = line.lstrip()
                if stripped.startswith("#") or stripped.startswith("//"):
                    continue

                for pattern, def_type in patterns:
                    match = re.match(pattern, line)
                    if match:
                        name = match.group(2)
                        indent = len(match.group(1))
                        # Estimar nivel de anidamiento
                        level = indent // 4 if indent > 0 else 0
                        all_definitions.append({
                            "type": def_type,
                            "name": name,
                            "line": line_num,
                            "file": str(rel),
                            "level": level,
                        })

        if not all_definitions:
            return ToolResult(True, output="No se encontraron definiciones en los archivos escaneados.")

        # Formatear salida
        output_lines = []
        current_file = None
        for d in all_definitions:
            if d["file"] != current_file:
                current_file = d["file"]
                output_lines.append(f"\n--- {current_file} ---")
            indent = "  " * d["level"]
            output_lines.append(f"  L{d['line']:4d} | {indent}{d['type']}: {d['name']}")

        summary = f"Se encontraron {len(all_definitions)} definicion(es) en {len(files)} archivo(s):"
        return ToolResult(True, output=summary + "\n" + "\n".join(output_lines))

    def _extract_python_ast(self, fpath: Path) -> list[dict]:
        """Extrae definiciones usando el AST de Python (mas preciso que regex)."""
        definitions = []
        try:
            source = fpath.read_text(encoding="utf-8")
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    definitions.append({
                        "type": "function" if not isinstance(node, ast.AsyncFunctionDef) else "async_function",
                        "name": node.name,
                        "line": node.lineno,
                        "file": str(fpath),
                        "level": 0,
                    })
                elif isinstance(node, ast.ClassDef):
                    definitions.append({
                        "type": "class",
                        "name": node.name,
                        "line": node.lineno,
                        "file": str(fpath),
                        "level": 0,
                    })
        except (SyntaxError, ValueError):
            pass
        return definitions


def _human_size(size: int) -> str:
    """Convierte bytes a formato legible."""
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"
