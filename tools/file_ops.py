"""Herramientas de operaciones de archivos: read_file, write_file, edit_file, search_files.

Mejorado con fuzzy matching en edit_file y diff unificado profesional,
inspirado en replace_in_file de Cline.
"""

from __future__ import annotations

import difflib
import os
from pathlib import Path
from typing import Any

from .base import BaseTool, ToolParameter, ToolResult


class ReadFileTool(BaseTool):
    """Lee el contenido completo o parcial de un archivo.

    Mejorado con soporte para lectura chunked como Cline.
    """

    name = "read_file"
    description = (
        "Lee el contenido de un archivo de texto. Soporta paginacion con offset y limit. "
        "Retorna el contenido con numeros de linea. Ideal para examinar codigo fuente. "
        "Para archivos grandes, usa offset y limit para leer por secciones."
    )
    parameters = [
        ToolParameter(
            name="file_path",
            type="string",
            description="Ruta del archivo a leer.",
            required=True,
        ),
        ToolParameter(
            name="offset",
            type="integer",
            description="Linea de inicio (1-based). Defecto: 1.",
            required=False,
            default=1,
        ),
        ToolParameter(
            name="limit",
            type="integer",
            description="Maximo de lineas a leer. Defecto: 500.",
            required=False,
            default=500,
        ),
        ToolParameter(
            name="encoding",
            type="string",
            description="Codificacion del archivo. Defecto: utf-8.",
            required=False,
            default="utf-8",
        ),
    ]

    def __init__(self, cwd_getter: callable) -> None:
        self._get_cwd = cwd_getter

    def _resolve(self, file_path: str) -> Path:
        if not os.path.isabs(file_path):
            file_path = str(Path(self._get_cwd()) / file_path)
        return Path(file_path)

    def execute(self, **kwargs: Any) -> ToolResult:
        file_path = kwargs.get("file_path", "")
        offset = kwargs.get("offset", 1)
        limit = kwargs.get("limit", 500)
        encoding = kwargs.get("encoding", "utf-8")

        try:
            path = self._resolve(file_path)
            if not path.exists():
                return ToolResult(success=False, error=f"Archivo no encontrado: {path}")
            if not path.is_file():
                return ToolResult(success=False, error=f"No es un archivo: {path}")

            with open(path, "r", encoding=encoding) as f:
                lines = f.readlines()

            total = len(lines)
            start = max(0, offset - 1)
            end = min(total, start + limit)
            selected = lines[start:end]

            numbered = []
            for i, line in enumerate(selected, start=start + 1):
                numbered.append(f"{i:>6} | {line.rstrip()}")

            content = "\n".join(numbered)
            if end < total:
                content += (
                    f"\n\n[{end + 1}-{total}] ({total - end} lineas mas. "
                    f"Usa offset={end + 1} para leer mas)"
                )

            return ToolResult(
                success=True,
                output=f"{path} ({total} lineas)\n{'=' * 70}\n{content}",
            )
        except UnicodeDecodeError:
            return ToolResult(
                success=False,
                error=f"No se pudo decodificar con '{encoding}'. Intenta con 'latin-1' o 'utf-16'.",
            )
        except PermissionError:
            return ToolResult(success=False, error="Permiso denegado")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class WriteFileTool(BaseTool):
    """Escribe contenido en un archivo.

    Crea o sobreescribe completamente. Igual que write_to_file de Cline.
    """

    name = "write_file"
    description = (
        "Escribe contenido en un archivo. Si el archivo existe, lo sobreescribe completamente. "
        "Crea directorios padre automaticamente si no existen. "
        "ADVERTENCIA: Esto sobreescribe el archivo existente. Para ediciones parciales, "
        "usa edit_file en su lugar."
    )
    parameters = [
        ToolParameter(
            name="file_path",
            type="string",
            description="Ruta del archivo donde escribir.",
            required=True,
        ),
        ToolParameter(
            name="content",
            type="string",
            description="Contenido a escribir en el archivo.",
            required=True,
        ),
        ToolParameter(
            name="encoding",
            type="string",
            description="Codificacion. Defecto: utf-8.",
            required=False,
            default="utf-8",
        ),
    ]

    def __init__(self, cwd_getter: callable) -> None:
        self._get_cwd = cwd_getter

    def execute(self, **kwargs: Any) -> ToolResult:
        file_path = kwargs.get("file_path", "")
        content = kwargs.get("content", "")
        encoding = kwargs.get("encoding", "utf-8")

        try:
            if not os.path.isabs(file_path):
                file_path = str(Path(self._get_cwd()) / file_path)

            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "w", encoding=encoding) as f:
                f.write(content)

            line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
            byte_size = path.stat().st_size
            return ToolResult(
                success=True,
                output=f"Escrito: {path} ({line_count} lineas, {byte_size} bytes)",
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class EditFileTool(BaseTool):
    """Edita un archivo reemplazando un fragmento de texto.

    Mejorado con fuzzy matching y diff profesional, inspirado en replace_in_file de Cline:
    - Si old_text no se encuentra exactamente, busca la mejor coincidencia fuzzy
    - Muestra diff unificado siempre
    - Reporta linea exacta del cambio
    """

    name = "edit_file"
    description = (
        "Reemplaza un fragmento de texto en un archivo con nuevo texto. "
        "Busca la primera ocurrencia exacta de 'old_text' y la reemplaza con 'new_text'. "
        "Si 'old_text' aparece multiples veces, usa 'replace_all' para reemplazar todas. "
        "Si no se encuentra exactamente, intenta una busqueda difusa (fuzzy match) para "
        "encontrar la mejor coincidencia cercana. PRECAUCION: old_text debe coincidir "
        "exactamente incluyendo espacios y saltos de linea. Siempre muestra el diff resultante."
    )
    parameters = [
        ToolParameter(
            name="file_path",
            type="string",
            description="Ruta del archivo a editar.",
            required=True,
        ),
        ToolParameter(
            name="old_text",
            type="string",
            description="Texto exacto a buscar y reemplazar. Debe coincidir exactamente.",
            required=True,
        ),
        ToolParameter(
            name="new_text",
            type="string",
            description="Texto de reemplazo.",
            required=True,
        ),
        ToolParameter(
            name="replace_all",
            type="boolean",
            description="Reemplazar todas las ocurrencias (no solo la primera). Defecto: false.",
            required=False,
            default=False,
        ),
        ToolParameter(
            name="fuzzy_threshold",
            type="number",
            description="Umbral de similitud fuzzy (0.0-1.0). Si old_text no coincide exactamente, "
                       "busca coincidencias con similitud >= este umbral. Defecto: 0.6.",
            required=False,
            default=0.6,
        ),
    ]

    def __init__(self, cwd_getter: callable) -> None:
        self._get_cwd = cwd_getter

    def execute(self, **kwargs: Any) -> ToolResult:
        file_path = kwargs.get("file_path", "")
        old_text = kwargs.get("old_text", "")
        new_text = kwargs.get("new_text", "")
        replace_all = kwargs.get("replace_all", False)
        fuzzy_threshold = kwargs.get("fuzzy_threshold", 0.6)

        try:
            if not os.path.isabs(file_path):
                file_path = str(Path(self._get_cwd()) / file_path)

            path = Path(file_path)
            if not path.exists():
                return ToolResult(success=False, error=f"Archivo no encontrado: {path}")

            with open(path, "r", encoding="utf-8") as f:
                original = f.read()

            lines = original.split("\n")

            # 1. Busqueda exacta
            if old_text in original:
                new_content, count, line_info = self._do_replace(
                    original, old_text, new_text, replace_all
                )
            else:
                # 2. Fuzzy match: buscar la mejor coincidencia
                fuzzy_result = self._fuzzy_find(old_text, lines, fuzzy_threshold)
                if fuzzy_result:
                    best_match_text, line_num = fuzzy_result
                    new_content = original.replace(best_match_text, new_text, 1)
                    count = 1
                    line_info = f"(fuzzy match en linea {line_num})"
                else:
                    # 3. No se encontro nada - mostrar contexto util
                    return ToolResult(
                        success=False,
                        error=self._build_not_found_error(old_text, lines, path),
                    )

            # Escribir archivo
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)

            # Generar diff unificado profesional
            diff_text = self._generate_diff(original, new_content, path)

            return ToolResult(
                success=True,
                output=f"Editado: {path} {line_info} ({count} reemplazo(s))\n\nDiff:\n{diff_text}",
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _do_replace(
        self, content: str, old_text: str, new_text: str, replace_all: bool
    ) -> tuple[str, int, str]:
        """Ejecuta el reemplazo y retorna (nuevo_contenido, cantidad, info_linea)."""
        # Encontrar la linea del primer match
        lines = content.split("\n")
        first_line = 0
        for i, line in enumerate(lines):
            if old_text in line:
                first_line = i + 1
                break
            # Verificar si old_text empieza en esta linea
            remaining = "\n".join(lines[i:])
            if old_text in remaining:
                first_line = i + 1
                break

        if replace_all:
            new_content = content.replace(old_text, new_text)
            count = content.count(old_text)
        else:
            new_content = content.replace(old_text, new_text, 1)
            count = 1

        return new_content, count, f"(linea ~{first_line})"

    def _fuzzy_find(
        self, old_text: str, lines: list[str], threshold: float
    ) -> tuple[str, int] | None:
        """Busca fuzzy match del old_text entre las lineas del archivo.

        Busca secuencias de lineas que se parezcan al old_text proporcionado.
        Retorna (texto_encontrado, numero_de_linea) o None.
        """
        old_lines = old_text.split("\n")
        old_stripped = [l.strip() for l in old_lines if l.strip()]

        if not old_stripped:
            return None

        best_ratio = 0.0
        best_match: tuple[str, int] | None = None

        # Normalizar las lineas del archivo
        window_size = min(len(old_lines), 20)  # Limitar ventana de busqueda

        for i in range(len(lines)):
            end = min(i + window_size, len(lines))
            window = lines[i:end]
            window_stripped = [l.strip() for l in window if l.strip()]

            # Comparar secuencias
            ratio = difflib.SequenceMatcher(
                None, old_stripped, window_stripped
            ).ratio()

            if ratio > best_ratio:
                best_ratio = ratio
                # Recrear el texto original con su indentacion real
                match_text = "\n".join(window)
                best_match = (match_text, i + 1)

        if best_match and best_ratio >= threshold:
            return best_match
        return None

    def _build_not_found_error(
        self, old_text: str, lines: list[str], path: Path
    ) -> str:
        """Construye un mensaje de error util cuando old_text no se encuentra."""
        old_lines = old_text.strip().split("\n")
        # Buscar lineas que contengan partes del old_text
        hints = []
        first_significant_line = old_lines[0].strip() if old_lines else ""
        for i, line in enumerate(lines, 1):
            if first_significant_line and first_significant_line[:30] in line:
                hints.append(f"  Linea {i}: {line.rstrip()[:100]}")
                if len(hints) >= 3:
                    break

        error_parts = [
            f"No se encontro 'old_text' en {path}.",
            "",
            "Asegurate de que coincide EXACTAMENTE:",
            "- Espacios y tabulaciones al inicio de cada linea",\n            "- Saltos de linea al final",\n            "- Mayusculas/minusculas exactas",\n        ]

        if hints:
            error_parts.append("")
            error_parts.append("Lineas similares encontradas:")
            error_parts.extend(hints)
            error_parts.append("")
            error_parts.append("Revisa estas lineas - podrian ser lo que buscas con diferente indentacion.")
        else:
            # Mostrar primeras lineas del archivo como referencia
            error_parts.append("")
            error_parts.append("Primeras lineas del archivo:")
            for i, line in enumerate(lines[:5], 1):
                error_parts.append(f"  {i}: {line.rstrip()[:100]}")

        return "\n".join(error_parts)

    def _generate_diff(
        self, original: str, new_content: str, path: Path
    ) -> str:
        """Genera un diff unificado profesional estilo Cline."""
        diff_lines = list(difflib.unified_diff(
            original.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{path.name}",
            tofile=f"b/{path.name}",
            lineterm="",
        ))

        if not diff_lines:
            return "(sin cambios visibles en el diff)"

        # Formatear con colores ANSI para terminal
        formatted = []
        for line in diff_lines:
            if line.startswith("+++") or line.startswith("---"):
                formatted.append(f"\033[36m{line}\033[0m")  # cyan
            elif line.startswith("@@"):
                formatted.append(f"\033[33m{line}\033[0m")  # yellow
            elif line.startswith("+"):
                formatted.append(f"\033[32m{line}\033[0m")  # green
            elif line.startswith("-"):
                formatted.append(f"\033[31m{line}\033[0m")  # red
            else:
                formatted.append(line)

        return "\n".join(formatted)


class SearchFilesTool(BaseTool):
    """Busca archivos por patron o contenido."""

    name = "search_files"
    description = (
        "Busca archivos que coincidan con un patron glob (ej: '**/*.py') o que contengan un texto. "
        "Combinar con 'pattern' para busqueda por nombre y 'content' para busqueda por contenido."
    )
    parameters = [
        ToolParameter(
            name="pattern",
            type="string",
            description="Patron glob para buscar archivos (ej: '**/*.py', 'src/**/*.ts').",
            required=False,
            default="**/*",
        ),
        ToolParameter(
            name="content",
            type="string",
            description="Texto a buscar dentro de los archivos (busqueda por contenido).",
            required=False,
        ),
        ToolParameter(
            name="path",
            type="string",
            description="Directorio base para la busqueda. Defecto: directorio actual.",
            required=False,
            default=".",
        ),
        ToolParameter(
            name="max_results",
            type="integer",
            description="Maximo de resultados. Defecto: 50.",
            required=False,
            default=50,
        ),
    ]

    def __init__(self, cwd_getter: callable) -> None:
        self._get_cwd = cwd_getter

    def execute(self, **kwargs: Any) -> ToolResult:
        pattern = kwargs.get("pattern", "**/*")
        content = kwargs.get("content", "")
        path = kwargs.get("path", ".")
        max_results = kwargs.get("max_results", 50)

        try:
            if os.path.isabs(path):
                base = Path(path)
            else:
                base = Path(self._get_cwd()) / path

            if not base.is_dir():
                return ToolResult(success=False, error=f"No es un directorio: {base}")

            results = []
            matched = 0

            for file_path in sorted(base.glob(pattern)):
                if matched >= max_results:
                    break
                if not file_path.is_file():
                    continue

                include = True
                snippet = ""
                if content:
                    try:
                        text = file_path.read_text(encoding="utf-8", errors="ignore")
                        if content.lower() not in text.lower():
                            include = False
                        else:
                            for i, line in enumerate(text.split("\n")):
                                if content.lower() in line.lower():
                                    start = max(0, i - 1)
                                    end = min(len(text.split("\n")), i + 2)
                                    snippet_lines = text.split("\n")[start:end]
                                    snippet = "\n".join(
                                        f"  {j + 1}: {l}" for j, l in enumerate(snippet_lines, start=start)
                                    )
                                    break
                    except Exception:
                        include = False

                if include:
                    rel = file_path.relative_to(base)
                    entry = f"  {rel}"
                    if snippet:
                        entry += f"\n{snippet}"
                    results.append(entry)
                    matched += 1

            if not results:
                msg = f"No se encontraron archivos"
                if content:
                    msg += f" que contengan '{content}'"
                msg += f" con patron '{pattern}' en {base}"
                return ToolResult(success=True, output=msg)

            header = f"Resultados ({matched}) en {base}/"
            return ToolResult(success=True, output=f"{header}\n{'-' * len(header)}\n" + "\n".join(results))
        except Exception as e:
            return ToolResult(success=False, error=str(e))
