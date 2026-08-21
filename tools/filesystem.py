"""Herramientas de sistema de archivos: cd, ls, cat, mkdir, touch, rm, pwd."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from .base import BaseTool, ToolParameter, ToolResult


class CdTool(BaseTool):
    """Cambia el directorio de trabajo actual."""

    name = "cd"
    description = (
        "Cambia el directorio de trabajo actual del agente. "
        "Usa '..' para ir al directorio padre, o una ruta absoluta/relativa."
    )
    parameters = [
        ToolParameter(
            name="path",
            type="string",
            description="Ruta del directorio al que navegar. Usa '..' para padre.",
            required=True,
        )
    ]

    def __init__(self, cwd: str | None = None) -> None:
        self._cwd = cwd or os.getcwd()

    @property
    def cwd(self) -> str:
        return self._cwd

    @cwd.setter
    def cwd(self, value: str) -> None:
        self._cwd = os.path.realpath(value)

    def execute(self, **kwargs: Any) -> ToolResult:
        path = kwargs.get("path", "")
        try:
            if path == "~":
                target = Path.home()
            elif path.startswith("~"):
                target = Path.home() / path[2:]
            else:
                target = Path(self._cwd) / path

            target = target.resolve()

            if not target.exists():
                return ToolResult(success=False, error=f"Directorio no encontrado: {target}")
            if not target.is_dir():
                return ToolResult(success=False, error=f"No es un directorio: {target}")

            self._cwd = str(target)
            return ToolResult(success=True, output=f"Directorio cambiado a: {self._cwd}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class LsTool(BaseTool):
    """Lista el contenido de un directorio."""

    name = "ls"
    description = (
        "Lista los archivos y directorios en el directorio actual o en la ruta dada. "
        "Muestra nombres, tipos (DIR/FILE) y tamanos."
    )
    parameters = [
        ToolParameter(
            name="path",
            type="string",
            description="Ruta del directorio a listar. Por defecto el directorio actual.",
            required=False,
            default=".",
        ),
        ToolParameter(
            name="show_hidden",
            type="boolean",
            description="Si se muestran archivos ocultos (que empiezan con '.').",
            required=False,
            default=False,
        ),
    ]

    def __init__(self, cwd_getter: callable) -> None:
        self._get_cwd = cwd_getter

    def execute(self, **kwargs: Any) -> ToolResult:
        path = kwargs.get("path", ".")
        show_hidden = kwargs.get("show_hidden", False)

        try:
            if os.path.isabs(path):
                target = Path(path)
            else:
                target = Path(self._get_cwd()) / path

            if not target.exists():
                return ToolResult(success=False, error=f"Ruta no encontrada: {target}")
            if not target.is_dir():
                return ToolResult(success=False, error=f"No es un directorio: {target}")

            entries = []
            for entry in sorted(target.iterdir()):
                name = entry.name
                if not show_hidden and name.startswith("."):
                    continue
                if entry.is_dir():
                    entries.append(f"  DIR   {name}/")
                else:
                    size = entry.stat().st_size
                    if size < 1024:
                        size_str = f"{size} B"
                    elif size < 1024 * 1024:
                        size_str = f"{size / 1024:.1f} KB"
                    else:
                        size_str = f"{size / (1024 * 1024):.1f} MB"
                    entries.append(f"  FILE  {name}  ({size_str})")

            if not entries:
                return ToolResult(success=True, output="(directorio vacio)")

            header = f"Contenido de {target}/"
            separator = "-" * len(header)
            output = f"{header}\n{separator}\n" + "\n".join(entries)
            return ToolResult(success=True, output=output)
        except PermissionError:
            return ToolResult(success=False, error="Permiso denegado")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class CatTool(BaseTool):
    """Muestra el contenido de un archivo."""

    name = "cat"
    description = (
        "Muestra el contenido de un archivo de texto. "
        "Soporta limites de lineas con offset para archivos grandes."
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
            description="Numero de linea inicial (1-based). Por defecto 1.",
            required=False,
            default=1,
        ),
        ToolParameter(
            name="limit",
            type="integer",
            description="Numero maximo de lineas a mostrar. Por defecto 200.",
            required=False,
            default=200,
        ),
    ]

    def __init__(self, cwd_getter: callable) -> None:
        self._get_cwd = cwd_getter

    def execute(self, **kwargs: Any) -> ToolResult:
        file_path = kwargs.get("file_path", "")
        offset = kwargs.get("offset", 1)
        limit = kwargs.get("limit", 200)

        try:
            if not os.path.isabs(file_path):
                file_path = str(Path(self._get_cwd()) / file_path)

            path = Path(file_path)
            if not path.exists():
                return ToolResult(success=False, error=f"Archivo no encontrado: {path}")
            if not path.is_file():
                return ToolResult(success=False, error=f"No es un archivo: {path}")

            # Check if file is text
            try:
                with open(path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except UnicodeDecodeError:
                return ToolResult(
                    success=False,
                    error="El archivo parece ser binario. No se puede mostrar como texto.",
                )

            total = len(lines)
            start = max(0, offset - 1)
            end = min(total, start + limit)
            selected = lines[start:end]

            # Add line numbers
            numbered = []
            for i, line in enumerate(selected, start=start + 1):
                numbered.append(f"{i:>6}\t{line.rstrip()}")

            content = "\n".join(numbered)
            if end < total:
                content += f"\n\n... ({total - end} lineas restantes, usa offset={end + 1} para continuar)"

            return ToolResult(
                success=True,
                output=f"Archivo: {path}  ({total} lineas totales)\n{'=' * 60}\n{content}",
            )
        except PermissionError:
            return ToolResult(success=False, error="Permiso denegado")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class MkdirTool(BaseTool):
    """Crea un directorio."""

    name = "mkdir"
    description = "Crea un nuevo directorio, incluyendo directorios padre si no existen (como mkdir -p)."
    parameters = [
        ToolParameter(
            name="path",
            type="string",
            description="Ruta del directorio a crear.",
            required=True,
        ),
    ]

    def __init__(self, cwd_getter: callable) -> None:
        self._get_cwd = cwd_getter

    def execute(self, **kwargs: Any) -> ToolResult:
        path = kwargs.get("path", "")
        try:
            if not os.path.isabs(path):
                path = str(Path(self._get_cwd()) / path)

            target = Path(path)
            if target.exists():
                return ToolResult(success=False, error=f"Ya existe: {target}")

            target.mkdir(parents=True, exist_ok=False)
            return ToolResult(success=True, output=f"Directorio creado: {target}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class TouchTool(BaseTool):
    """Crea un archivo vacio o actualiza su timestamp."""

    name = "touch"
    description = "Crea un archivo vacio si no existe, o actualiza su fecha de modificacion si existe."
    parameters = [
        ToolParameter(
            name="file_path",
            type="string",
            description="Ruta del archivo a crear/tocar.",
            required=True,
        ),
    ]

    def __init__(self, cwd_getter: callable) -> None:
        self._get_cwd = cwd_getter

    def execute(self, **kwargs: Any) -> ToolResult:
        file_path = kwargs.get("file_path", "")
        try:
            if not os.path.isabs(file_path):
                file_path = str(Path(self._get_cwd()) / file_path)

            path = Path(file_path)
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            return ToolResult(success=True, output=f"Archivo creado/actualizado: {path}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class RmTool(BaseTool):
    """Elimina archivos o directorios."""

    name = "rm"
    description = "Elimina un archivo o directorio. Para directorios, elimina recursivamente (como rm -rf). USAR CON CUIDADO."
    parameters = [
        ToolParameter(
            name="path",
            type="string",
            description="Ruta del archivo o directorio a eliminar.",
            required=True,
        ),
        ToolParameter(
            name="recursive",
            type="boolean",
            description="Eliminar directorios recursivamente. Requerido para directorios.",
            required=False,
            default=False,
        ),
    ]

    def __init__(self, cwd_getter: callable) -> None:
        self._get_cwd = cwd_getter

    def execute(self, **kwargs: Any) -> ToolResult:
        path = kwargs.get("path", "")
        recursive = kwargs.get("recursive", False)
        try:
            if not os.path.isabs(path):
                path = str(Path(self._get_cwd()) / path)

            target = Path(path)
            if not target.exists():
                return ToolResult(success=False, error=f"No encontrado: {target}")

            if target.is_dir():
                if not recursive:
                    return ToolResult(
                        success=False,
                        error="Es un directorio. Usa recursive=true para eliminarlo.",
                    )
                shutil.rmtree(target)
                return ToolResult(success=True, output=f"Directorio eliminado: {target}")
            else:
                target.unlink()
                return ToolResult(success=True, output=f"Archivo eliminado: {target}")
        except PermissionError:
            return ToolResult(success=False, error="Permiso denegado")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class PwdTool(BaseTool):
    """Muestra el directorio de trabajo actual."""

    name = "pwd"
    description = "Muestra el directorio de trabajo actual del agente."
    parameters = []

    def __init__(self, cwd_getter: callable) -> None:
        self._get_cwd = cwd_getter

    def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, output=self._get_cwd())