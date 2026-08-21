"""Utilidades de visualizacion con Rich para el agente Cline.

Usa la libreria Rich para Paneles, Tablas, Spinners, Progress bars,
formatted text y todo lo necesario para una UI de terminal profesional.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import threading
from typing import Any, Callable
from contextlib import contextmanager

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich.prompt import Prompt
from rich.spinner import Spinner
from rich.status import Status
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeElapsedColumn, MofNCompleteColumn, TaskProgressColumn,
)
from rich.rule import Rule
from rich.box import ROUNDED, DOUBLE, HEAVY, SIMPLE, SQUARE
from rich.syntax import Syntax


# ── Consola global ─────────────────────────────────────────────────────

def _make_console() -> Console:
    """Crea la consola Rich. Detecta soporte de color automaticamente."""
    if os.getenv("NO_COLOR"):
        return Console(force_terminal=False, no_color=True)
    return Console()


console = _make_console()


# ── Paleta de colores (mantener compatibilidad con agent.py) ────────────

RST   = ""
BOLD  = "bold"
DIM   = "dim"
UND   = "underline"

BLK   = "black"
RED   = "red"
GRN   = "green"
YLW   = "yellow"
BLU   = "blue"
MAG   = "magenta"
CYN   = "cyan"
WHT   = "white"
GRY   = "dim"

# Combinaciones
TOOL_CALL   = "cyan"
TOOL_ERROR  = "red"
TOOL_NAME   = "bold cyan"
USER_PROMPT = "green"
BOT_TEXT    = "white"
BOT_THINK   = "cyan"
ACCENT      = "yellow"
SEPARATOR   = "dim"
STATUS_OK   = "green"
STATUS_ERR  = "red"
STATUS_INFO = "cyan"


# ── Funciones compatibles (mantienen la misma firma) ───────────────────

def c(code: str, text: str) -> str:
    """Envuelve texto en markup de Rich.

    En lugar de ANSI codes, retorna texto plano.
    Las funciones que imprimen usan Rich directamente.
    Esta funcion existe para compatibilidad con agent.py que usa c() inline.
    """
    # Para uso inline en f-strings que se imprimen con console.print
    # Devolvemos texto plano ya que console.print maneja el estilo
    return text


def icon(kind: str) -> str:
    """Retorna un icono unicode segun el tipo."""
    icons = {
        "tool":     "\u2699",   # ⚙
        "ok":       "\u2713",    # ✓
        "err":      "\u2717",   # ✗
        "info":     "\u2139",  # ℹ
        "arrow":    "\u203a",          # ›
        "prompt":   "\u2757", # ❗
        "bot":      "\U0001f916",      # 🤖
        "bullet":   "\u2022",       # •
        "dim_bull": "\u2022",          # •
        "hline":    "\u2500" * 40, # ─
        "folder":   "\U0001f4c1",      # 📁
        "save":     "\U0001f4be",      # 💾
        "back":     "\u2190",          # ←
        "home":     "\u2302",          # ⌂
    }
    return icons.get(kind, "")


# ── Spinner context manager para llamadas API ─────────────────────────

class APICallSpinner:
    """Spinner animado tipo Node.js para llamadas a la API.

    Muestra puntitos girando en circulo mientras la API trabaja.
    Uso:
        with api_spinner("Consultando modelos..."):
            models = api.list_models()
    """

    def __init__(self, message: str = "Procesando..."):
        self.message = message

    def __enter__(self):
        self._status = Status(
            self.message,
            spinner="dots",
            spinner_style="cyan",
            console=console,
        )
        self._status.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._status.stop()
        if exc_type is not None:
            # Si hubo error, mostrar mensaje rojo
            console.print(f"  [red]\u2717 {self.message} [/red] [dim]fall\u00f3[/dim]")
        return False


def api_spinner(message: str = "Procesando...") -> APICallSpinner:
    """Retorna un context manager con spinner animado para la API."""
    return APICallSpinner(message)


# ── Progress bar para planificador ──────────────────────────────────────

class TaskProgressBar:
    """Barra de progreso para el planificador de tareas.

    Estilo tipo Node.js npm install con spinner y barra.
    """

    def __init__(self, total: int, description: str = "Ejecutando plan"):
        self.total = total
        self.description = description
        self._progress = Progress(
            SpinnerColumn(spinner_name="dots", style="cyan"),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=30, complete_style="green", finished_style="bold green",
                      pulse_style="cyan"),
            TaskProgressColumn(show_speed=False),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        )
        self._task_id = None

    def __enter__(self):
        self._progress.start()
        self._task_id = self._progress.add_task(
            self.description, total=self.total
        )
        return self

    def __exit__(self, *args):
        self._progress.stop()

    def advance(self, n: int = 1):
        """Avanza el progreso."""
        if self._task_id is not None:
            self._progress.advance(self._task_id, n)

    def update_description(self, desc: str):
        """Cambia la descripcion de la tarea actual."""
        if self._task_id is not None:
            self._progress.update(self._task_id, description=desc)


def task_progress_bar(total: int, description: str = "Ejecutando plan") -> TaskProgressBar:
    """Retorna un context manager con barra de progreso para tareas."""
    return TaskProgressBar(total, description)


# ── Formateo de argumentos y resultados (interno) ───────────────────────

def _format_args(args: dict[str, Any], max_len: int = 120) -> str:
    """Formatea argumentos de tool call de forma compacta y legible."""
    if not args:
        return "{}"
    truncated = {}
    for k, v in args.items():
        s = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
        if len(s) > 80:
            s = s[:77] + "..."
        truncated[k] = s
    parts = []
    for k, v in truncated.items():
        parts.append(f"[yellow]{k}[/yellow]={v}")
    result = ", ".join(parts)
    if len(result) > max_len:
        result = result[:max_len - 3] + "[dim]...[/dim]"
    return result


def _parse_tool_result(raw: str) -> tuple[bool, str, str]:
    """Parsea el resultado raw de una herramienta."""
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(data, dict):
            return (
                data.get("success", True),
                data.get("output", ""),
                data.get("error", ""),
            )
    except (json.JSONDecodeError, TypeError):
        pass
    return True, str(raw), ""


def _truncate_lines(text: str, max_lines: int = 12, max_chars: int = 600) -> str:
    """Trunca texto a N lineas y M caracteres."""
    lines = text.split("\n")
    truncated = False
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        truncated = True
    result = "\n".join(lines)
    if len(result) > max_chars:
        result = result[:max_chars]
        truncated = True
    if truncated:
        remaining = len(text.split("\n")) - max_lines
        result += f"\n  [dim]... ({remaining} lineas mas)[/dim]"
    return result


# ── Funciones principales de display ────────────────────────────────────

def print_tool_call(name: str, args: dict[str, Any]) -> None:
    """Muestra una llamada a herramienta con formato Rich."""
    args_str = _format_args(args)
    console.print(f"  [bold cyan]\u2699[/bold cyan] [bold cyan]{name}[/bold cyan]  {args_str}")


def print_tool_result(name: str, raw_result: str) -> None:
    """Muestra el resultado de una herramienta con formato Rich."""
    success, output, error = _parse_tool_result(raw_result)
    if success:
        if output:
            lines = output.strip().split("\n")
            title = lines[0] if lines else ""
            body = "\n".join(lines[1:]) if len(lines) > 1 else ""
            console.print(f"  [green]\u2713[/green] [dim]{title}[/dim]")
            if body.strip():
                formatted = _truncate_lines(body.strip(), max_lines=10, max_chars=500)
                for line in formatted.split("\n"):
                    console.print(f"    [dim]{line}[/dim]")
        else:
            console.print(f"  [green]\u2713[/green] [dim](sin salida)[/dim]")
    else:
        err_msg = error if error else "Error desconocido"
        console.print(f"  [red]\u2717[/red] [red]{err_msg}[/red]")


def print_assistant_text(text: str) -> None:
    """Muestra el texto de respuesta del asistente."""
    if text.strip():
        console.print()
        # Usar un Panel sutil para la respuesta
        console.print(Panel(
            Text(text, style="white"),
            border_style="dim",
            padding=(0, 2),
            expand=False,
        ))


def print_assistant_thinking(text: str) -> None:
    """Muestra texto de 'pensamiento' del asistente."""
    if text.strip():
        for line in text.strip().split("\n"):
            console.print(f"  [dim cyan]{line}[/dim cyan]")


def print_user_input(text: str) -> None:
    """Muestra el prompt del usuario."""
    console.print()
    console.print(f"[bold green]> Tu[/bold green] [white]{text}[/white]")


def print_separator() -> None:
    """Linea separadora elegante."""
    console.print(Rule(style="dim"))


def print_status(text: str, kind: str = "info") -> None:
    """Mensaje de estado con icono."""
    styles = {
        "info": ("cyan", "\u2139"),
        "ok": ("green", "\u2713"),
        "err": ("red", "\u2717"),
    }
    style, ico = styles.get(kind, styles["info"])
    console.print(f"  [{style}]{ico}[/] [dim]{text}[/dim]")


# ── Banner ─────────────────────────────────────────────────────────────

def print_banner(version: str, model: str, api_url: str, cwd: str, n_tools: int) -> None:
    """Banner de inicio del agente con Panel de Rich."""
    logo_text = Text()
    logo_text.append("\n  C L I N E", style="bold bright_cyan")
    logo_text.append("   A G E N T", style="bold white")
    logo_text.append(f"   {version}", style="dim")
    logo_text.append("\n", style="")

    console.print(Panel(
        logo_text,
        border_style="bright_cyan",
        box=DOUBLE,
        padding=(0, 6),
        expand=False,
        subtitle="[dim]Agente de Codigo Autonomo[/dim]",
    ))

    # Info
    info_table = Table(show_header=False, box=None, padding=(0, 0), expand=False)
    info_table.add_column("key", style="dim", min_width=14)
    info_table.add_column("value", style="")
    info_table.add_row("Modelo", f"[yellow]{model}[/yellow]")
    info_table.add_row("API", f"[dim]{api_url}[/dim]")
    info_table.add_row("Proyecto", f"[white]{cwd}[/white]")
    info_table.add_row("Herramientas", f"[yellow]{n_tools}[/yellow] herramientas registradas")

    console.print(info_table)
    console.print(Rule(style="dim"))
    console.print("  [dim]Escribe [yellow]/help[/yellow] para comandos. Presiona Ctrl+C para cancelar.[/dim]")
    console.print()

# ── Help ────────────────────────────────────────────────────────────────

def print_help(current_model: str = "") -> None:
    """Muestra la ayuda de comandos con Tablas de Rich."""
    console.print()

    # Tabla principal de ayuda
    table = Table(
        title="[bold bright_cyan]Comandos disponibles[/bold bright_cyan]",
        box=ROUNDED,
        border_style="bright_cyan",
        show_header=True,
        header_style="bold cyan",
        padding=(0, 2),
        expand=False,
    )
    table.add_column("Comando", style="yellow", min_width=24)
    table.add_column("Descripcion", style="dim", min_width=40)

    # PROYECTO
    table.add_section()
    table.add_row("[bold magenta]PROYECTO[/bold magenta]", "(directorio de trabajo)")
    table.add_row("/project", "Muestra el directorio de proyecto actual")
    table.add_row("/project <ruta>", "Cambia el directorio de proyecto")
    table.add_row("/project --home", "Vuelve al directorio original del proyecto")

    # INFO
    table.add_section()
    table.add_row("[bold green]INFO[/bold green]", "(solo lectura, 0 tokens)")
    table.add_row("/help", "Muestra esta ayuda")
    table.add_row("/models", "Lista modelos disponibles en la API")
    table.add_row("/tools", "Lista herramientas del agente")
    table.add_row("/history", "Muestra historial de la conversacion")
    table.add_row("/cwd", "Muestra directorio de trabajo actual")
    table.add_row("/prompt", "Muestra el system prompt actual")

    # CONFIGURACION
    table.add_section()
    table.add_row("[bold cyan]CONFIGURACION[/bold cyan]", "(menu interactivo, 0 tokens)")
    table.add_row("/settings", "Menu interactivo de configuracion")
    table.add_row("/settings model", "Ajustes del modelo (nombre, temp, tokens...)")
    table.add_row("/settings chat", "Ajustes del chat (historial, compresion...)")
    table.add_row("/settings tools", "Ajustes de herramientas (timeout, permisos...)")
    table.add_row("/settings general", "Ajustes generales (directorio, tema, verbose...)")
    table.add_row("/settings prompt", "Editar el system prompt")
    table.add_row("/settings save", "Guardar configuracion a archivo")

    # ATAJOS
    table.add_section()
    table.add_row("[bold yellow]ATAJOS[/bold yellow]", "(modificacion rapida)")
    table.add_row("/model", "Lista modelos disponibles")
    table.add_row("/model <nombre>", "Cambia el modelo activo")
    table.add_row("/model --info", "Info detallada del modelo actual")

    # CHAT
    table.add_section()
    table.add_row("[bold cyan]CHAT[/bold cyan]", "(gestion local, 0 tokens)")
    table.add_row("/compact", "Comprime el historial para ahorrar contexto")
    table.add_row("/clear", "Limpia todo el historial")

    # SALIR
    table.add_section()
    table.add_row("[bold red]SALIR[/bold red]", "")
    table.add_row("/quit", "Cierra el agente")
    table.add_row("/exit", "Alias de /quit")

    console.print(table)
    console.print()


def _cmd_row(cmd: str, desc: str) -> None:
    """Imprime una fila de comando (compatibilidad, ya no usada directamente)."""
    console.print(f"    [yellow]{cmd.ljust(24)}[/yellow] [dim]{desc}[/dim]")


def print_unknown_command(typed: str) -> None:
    """Mensaje cuando el usuario escribe un comando desconocido."""
    console.print()
    console.print(Panel(
        Text.from_markup(
            f"[red]\u2717 Comando desconocido:[/red] [yellow]{typed}[/yellow]\n\n"
            f"[dim]Los mensajes que empiezan con / son comandos.\n"
            f"Escribe [yellow]/help[/yellow] para ver los comandos disponibles.[/dim]"
        ),
        border_style="red",
        box=ROUNDED,
        padding=(1, 2),
        expand=False,
    ))


def print_confirm_action(action_desc: str) -> str:
    """Prompt de confirmacion para acciones destructivas."""
    try:
        response = Prompt.ask(
            f"  [bold yellow]?[/bold yellow] [white]{action_desc}[/white]",
            choices=["y", "n", "Y", "N"],
            default="n",
            console=console,
        )
        return response.lower().strip()
    except (EOFError, KeyboardInterrupt):
        console.print()
        return "n"


def print_prompt_view(system_prompt: str) -> None:
    """Muestra el system prompt actual con Panel de Rich."""
    lines = system_prompt.strip().split("\n")
    meta = f"{len(lines)} lineas | {len(system_prompt)} caracteres"

    console.print()
    console.print(Panel(
        Text(system_prompt.strip(), style="dim"),
        title="[bold bright_cyan]System Prompt actual[/bold bright_cyan]",
        subtitle=f"[dim]{meta}[/dim]",
        border_style="cyan",
        box=ROUNDED,
        padding=(1, 2),
        expand=False,
    ))
    console.print("  [dim]Para cambiarlo usa: [yellow]/settings prompt[/yellow] o [yellow]/settings[/yellow][/dim]")
    console.print()


def print_tool_list(tools: list[dict[str, Any]]) -> None:
    """Lista las herramientas con Table de Rich."""
    table = Table(
        title=f"[bold bright_cyan]Herramientas disponibles[/bold bright_cyan] [dim]({len(tools)})[/dim]",
        box=ROUNDED,
        border_style="cyan",
        show_header=True,
        header_style="bold cyan",
        padding=(0, 2),
        expand=False,
    )
    table.add_column("Herramienta", style="bold cyan", min_width=20)
    table.add_column("Parametros", style="yellow", min_width=30)
    table.add_column("Descripcion", style="dim", min_width=40)

    for t in tools:
        fn = t.get("function", {})
        name = fn.get("name", "?")
        desc = fn.get("description", "").split(".")[0]
        params = list(fn.get("parameters", {}).get("properties", {}).keys())
        params_str = ", ".join(params) if params else "(sin params)"
        table.add_row(name, params_str, desc)

    console.print()
    console.print(table)
    console.print()


# ── Funciones para el menu de settings interactivo ─────────────────────

def print_settings_menu() -> None:
    """Muestra el menu principal de /settings con Tabla."""
    table = Table(
        title="[bold bright_cyan]\u2699  Configuracion del Agente[/bold bright_cyan]",
        box=ROUNDED,
        border_style="cyan",
        show_header=False,
        padding=(0, 4),
        expand=False,
    )
    table.add_column("Opcion", style="yellow", min_width=14)
    table.add_column("Descripcion", style="dim", min_width=50)

    table.add_row("model", "Modelo: nombre, temperatura, max_tokens, top_p, API...")
    table.add_row("chat", "Chat: historial, compresion, system prompt...")
    table.add_row("tools", "Herramientas: timeout, permisos, comandos bloqueados...")
    table.add_row("general", "General: directorio, tema, verbose, iteraciones...")
    table.add_row("prompt", "Editar system prompt (modo editor)")
    table.add_section()
    table.add_row("[green]save[/green]", "Guardar configuracion actual a archivo")
    table.add_row("[dim]back[/dim]", "Volver al chat")

    console.print()
    console.print(table)
    console.print()


def print_settings_model_menu() -> None:
    """Submenu de ajustes de modelo."""
    table = Table(
        title="[bold bright_cyan]\u2699  Ajustes del Modelo[/bold bright_cyan]",
        box=ROUNDED,
        border_style="cyan",
        show_header=False,
        padding=(0, 4),
        expand=False,
    )
    table.add_column("Opcion", style="yellow", min_width=16)
    table.add_column("Descripcion", style="dim", min_width=50)

    table.add_row("model", "Cambiar modelo activo")
    table.add_row("api_url", "Cambiar URL de la API")
    table.add_row("api_key", "Cambiar API key")
    table.add_row("temperature", "Cambiar temperatura (0.0 - 2.0)")
    table.add_row("max_tokens", "Cambiar maximo de tokens de respuesta")
    table.add_row("top_p", "Cambiar top_p (0.0 - 1.0)")
    table.add_section()
    table.add_row("[dim]back[/dim]", "Volver al menu de configuracion")

    console.print()
    console.print(table)
    console.print()


def print_settings_chat_menu() -> None:
    """Submenu de ajustes de chat."""
    table = Table(
        title="[bold bright_cyan]\u2699  Ajustes del Chat[/bold bright_cyan]",
        box=ROUNDED,
        border_style="cyan",
        show_header=False,
        padding=(0, 4),
        expand=False,
    )
    table.add_column("Opcion", style="yellow", min_width=16)
    table.add_column("Descripcion", style="dim", min_width=50)

    table.add_row("max_history", "Maximo de mensajes en historial")
    table.add_row("save_history", "Guardar/No guardar historial a disco")
    table.add_row("auto_compress", "Comprimir historial automaticamente")
    table.add_row("compress_at", "Umbral de mensajes para comprimir")
    table.add_section()
    table.add_row("[dim]back[/dim]", "Volver al menu de configuracion")

    console.print()
    console.print(table)
    console.print()


def print_settings_tools_menu() -> None:
    """Submenu de ajustes de herramientas."""
    table = Table(
        title="[bold bright_cyan]\u2699  Ajustes de Herramientas[/bold bright_cyan]",
        box=ROUNDED,
        border_style="cyan",
        show_header=False,
        padding=(0, 4),
        expand=False,
    )
    table.add_column("Opcion", style="yellow", min_width=16)
    table.add_column("Descripcion", style="dim", min_width=50)

    table.add_row("max_timeout", "Timeout maximo de comandos (segundos)")
    table.add_row("max_file_read", "Maximo de lineas al leer archivos")
    table.add_row("dangerous", "Permitir/Denegar comandos peligrosos")
    table.add_row("blocked", "Ver/comandos bloqueados")
    table.add_section()
    table.add_row("[dim]back[/dim]", "Volver al menu de configuracion")

    console.print()
    console.print(table)
    console.print()


def print_settings_general_menu() -> None:
    """Submenu de ajustes generales."""
    table = Table(
        title="[bold bright_cyan]\u2699  Ajustes Generales[/bold bright_cyan]",
        box=ROUNDED,
        border_style="cyan",
        show_header=False,
        padding=(0, 4),
        expand=False,
    )
    table.add_column("Opcion", style="yellow", min_width=16)
    table.add_column("Descripcion", style="dim", min_width=50)

    table.add_row("directory", "Cambiar directorio de trabajo")
    table.add_row("theme", "Tema visual (dark / light)")
    table.add_row("verbose", "Mostrar/Ocultar detalle de herramientas")
    table.add_row("max_iter", "Maximo de iteraciones de herramientas por turno")
    table.add_section()
    table.add_row("[dim]back[/dim]", "Volver al menu de configuracion")

    console.print()
    console.print(table)
    console.print()


def print_model_info(model_name: str, api_url: str, api_key: str,
                     temperature: float, max_tokens: int, top_p: float) -> None:
    """Muestra informacion detallada del modelo con Panel."""
    masked = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "****"

    table = Table(show_header=False, box=None, padding=(0, 0), expand=False)
    table.add_column("key", style="dim", min_width=14)
    table.add_column("value", min_width=40)
    table.add_row("modelo", f"[yellow]{model_name}[/yellow]")
    table.add_row("api_url", f"[white]{api_url}[/white]")
    table.add_row("api_key", f"[white]{masked}[/white]")
    table.add_row("temperature", f"[white]{temperature}[/white]")
    table.add_row("max_tokens", f"[white]{max_tokens}[/white]")
    table.add_row("top_p", f"[white]{top_p}[/white]")

    console.print()
    console.print(Panel(
        table,
        title="[bold bright_cyan]Informacion del Modelo[/bold bright_cyan]",
        border_style="cyan",
        box=ROUNDED,
        padding=(1, 2),
        expand=False,
    ))
    console.print()


def print_project_info(project_path: str, home_path: str | None = None) -> None:
    """Muestra informacion del proyecto actual."""
    content = Text()
    content.append(f"\U0001f4c1 ", style="yellow")
    content.append(project_path, style="white")
    if home_path and home_path != project_path:
        content.append(f"\n\u2302 Inicio: {home_path}", style="dim")

    console.print()
    console.print(Panel(
        content,
        title="[bold bright_cyan]Proyecto[/bold bright_cyan]",
        border_style="cyan",
        box=ROUNDED,
        padding=(1, 2),
        expand=False,
    ))
    console.print()


def print_prompt_editor_header() -> None:
    """Header del editor de prompt."""
    console.print()
    console.print(Panel(
        Text.from_markup(
            "Escribe lineas de texto. Para terminar, escribe [yellow]/done[/yellow].\n"
            "Para cancelar, escribe [red]/cancel[/red].\n"
            "Para ver el actual, escribe [yellow]/view[/yellow].\n"
            "Para restaurar el por defecto, escribe [yellow]/reset[/yellow]."
        ),
        title="[bold bright_cyan]\u270f  Editor de System Prompt[/bold bright_cyan]",
        border_style="magenta",
        box=ROUNDED,
        padding=(1, 2),
        expand=False,
    ))


def print_settings_value_set(key: str, old_val: str, new_val: str) -> None:
    """Confirma que un valor fue cambiado."""
    console.print()
    console.print(
        f"  [green]\u2713[/green] [dim]{key}:[/dim] [dim]{old_val}[/dim] [dim]\u2192[/dim] [green]{new_val}[/green]"
    )
    console.print()


def print_settings_value_view(key: str, value: str) -> None:
    """Muestra el valor actual de un ajuste."""
    console.print()
    console.print(f"  [dim]{key}:[/dim] [white]{value}[/white]")
    console.print()


# ── Input helpers ───────────────────────────────────────────────────────

def get_input(prompt: str) -> str:
    """Obtiene input del usuario."""
    try:
        return console.input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def get_input_with_default(prompt: str, default: str) -> str:
    """Obtiene input con valor por defecto mostrado."""
    try:
        raw = Prompt.ask(
            f"  [bold yellow]>[/bold yellow] [white]{prompt}[/white]",
            default=default,
            console=console,
        )
        return raw.strip()
    except (EOFError, KeyboardInterrupt):
        console.print()
        return default


def get_input_multiline() -> str:
    """Lee multiples lineas hasta doble Enter vacio."""
    lines = []
    try:
        while True:
            line = console.input("  [dim]... [/dim]").rstrip("\n")
            if line == "" and lines and lines[-1] == "":
                lines.pop()
                break
            lines.append(line)
    except (EOFError, KeyboardInterrupt):
        console.print()
    return "\n".join(lines)


# ── Funciones para el planificador de tareas ─────────────────────────────

def print_plan(plan) -> None:
    """Muestra el plan de tareas con Panel y Tabla."""
    table = Table(
        title=f"[bold yellow]\U0001f4cb  Plan de Tareas[/bold yellow] [dim]({len(plan.tasks)} tareas)[/dim]",
        box=ROUNDED,
        border_style="yellow",
        show_header=False,
        padding=(0, 4),
        expand=False,
    )
    table.add_column("#", style="bold yellow", min_width=4, justify="right")
    table.add_column("Tarea", style="white", min_width=40)
    table.add_column("Detalle", style="dim", min_width=40)

    for t in plan.tasks:
        desc = t.description[:120] + ("..." if len(t.description) > 120 else "") if t.description else ""
        table.add_row(f"T{t.number}", t.title, desc)

    console.print()
    console.print(table)
    console.print()


def print_task_progress(plan) -> None:
    """Muestra el progreso actual del plan con barra y tabla."""
    done, total = plan.progress

    # Barra de progreso tipo Node.js
    bar_len = 30
    filled = int(bar_len * done / total) if total > 0 else 0
    filled_chars = "\u2588" * filled
    empty_chars = "\u2591" * (bar_len - filled)
    pct = int(100 * done / total) if total > 0 else 0

    console.print()
    console.print(
        f"  [cyan]Progreso:[/cyan] [green]{filled_chars}[/green][dim]{empty_chars}[/dim]"
        f" [white]{done}/{total}[/white] [dim]({pct}%)[/dim]"
    )

    # Tabla de estado de tareas
    table = Table(box=None, show_header=False, padding=(0, 1), expand=False)
    table.add_column("icon", min_width=3)
    table.add_column("task", min_width=6)
    table.add_column("title", min_width=50)

    for t in plan.tasks:
        if t.status == "done":
            table.add_row("[green]\u2713[/green]", f"[dim]T{t.number}[/dim]", f"[dim]{t.title}[/dim]")
        elif t.status == "error":
            table.add_row("[red]\u2717[/red]", f"[red]T{t.number}[/red]", f"[red]{t.title}[/red]")
        elif t.status == "in_progress":
            table.add_row("[yellow]\u25cf[/yellow]", f"[bold]T{t.number}[/bold]", f"[bold white]{t.title}[/bold white]")
        elif t.status == "skipped":
            table.add_row("[dim]\u25cb[/dim]", f"[dim]T{t.number}[/dim]", f"[dim]{t.title}[/dim]")
        else:
            table.add_row("[dim]\u25cb[/dim]", f"[dim]T{t.number}[/dim]", f"[dim]{t.title}[/dim]")

    console.print(table)
    console.print()


def print_task_start(task) -> None:
    """Muestra que una tarea esta empezando."""
    console.print()
    desc = task.description[:150] if task.description else ""
    if desc:
        console.print(Panel(
            Text(desc, style="dim"),
            title=f"[bold yellow]\u25b6  T{task.number}. {task.title}[/bold yellow]",
            border_style="yellow",
            box=SIMPLE,
            padding=(0, 2),
            expand=False,
        ))
    else:
        console.print(f"\n  [bold yellow]\u25b6  T{task.number}. {task.title}[/bold yellow]")


def print_task_result(task) -> None:
    """Muestra el resultado de una tarea completada."""
    if task.status == "done":
        console.print(f"  [green]\u2713[/green] [green]T{task.number} completada[/green]")
    elif task.status == "error":
        err_msg = task.result[:100] if task.result else "Error desconocido"
        console.print(f"  [red]\u2717[/red] [red]T{task.number} error: {err_msg}[/red]")


def print_plan_summary(plan) -> None:
    """Muestra el resumen final de la ejecucion del plan."""
    console.print()
    console.print(Panel(
        Text(plan.summary(), style="white"),
        title="[bold bright_cyan]Resumen del Plan[/bold bright_cyan]",
        border_style="cyan",
        box=ROUNDED,
        padding=(1, 2),
        expand=False,
    ))
    console.print()


def print_model_validation(result) -> None:
    """Muestra el resultado de la validacion de modelo."""
    if result.valid:
        console.print(f"  [green]\u2713[/green] [dim]{result.message}[/dim]")
    else:
        console.print()
        console.print(Panel(
            Text.from_markup(
                f"[red]\u2717 {result.message}[/red]"
                + (f"\n\n[bold yellow]Sugerencias:[/bold yellow]" if result.suggestions else "")
                + "\n".join(f"  [yellow]\u2022 {s}[/yellow]" for s in (result.suggestions or []))
            ),
            border_style="red",
            box=ROUNDED,
            padding=(1, 2),
            expand=False,
        ))
    console.print()
