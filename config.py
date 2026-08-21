"""Sistema de configuracion y ajustes del agente Cline.

Maneja carga/guardado de configuracion, ajustes de API, modelo, y preferencias.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# Ruta por defecto del archivo de configuracion
DEFAULT_CONFIG_PATH = Path.home() / ".cline-agent" / "config.json"
DEFAULT_HISTORY_PATH = Path.home() / ".cline-agent" / "history"


@dataclass
class ModelConfig:
    """Configuracion del modelo de IA."""
    api_url: str = "https://vimax-ia.p.jo3.org/v1/"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


@dataclass
class ToolConfig:
    """Configuracion de herramientas."""
    max_command_timeout: int = 120
    max_file_size_read: int = 500  # lineas
    allow_dangerous_commands: bool = False
    blocked_commands: list[str] = field(default_factory=lambda: [
        "rm -rf /",
        "mkfs",
        "dd if=",
        ":(){ :|:& };:",
    ])


@dataclass
class ChatConfig:
    """Configuracion del chat."""
    system_prompt: str = ""
    max_history_messages: int = 50
    save_history: bool = True
    history_dir: str = str(DEFAULT_HISTORY_PATH)
    auto_compress_history: bool = True
    compress_threshold: int = 30  # mensajes antes de comprimir


@dataclass
class PlannerConfig:
    """Configuracion del planificador de tareas."""
    enabled: bool = True  # activar planificacion automatica
    max_tasks: int = 7  # maximo de tareas por plan
    auto_execute: bool = True  # ejecutar sin confirmacion


@dataclass
class ApprovalConfig:
    """Configuracion de auto-approve (estilo Cline)."""
    auto_approve_reads: bool = True
    auto_approve_writes: bool = False
    auto_approve_commands: bool = False
    auto_approve_browser: bool = False
    auto_approve_mcp: bool = False
    yolo_mode: bool = False  # PELIGROSO: aprueba todo
    always_allow_commands: list[str] = field(default_factory=lambda: [
        "npm test", "npm run test", "npm run lint", "npm run build",
        "pytest", "python -m pytest", "cargo test", "cargo check",
        "git status", "git diff", "git log", "pip list",
    ])


@dataclass
class CheckpointConfig:
    """Configuracion del sistema de checkpoints (estilo Cline)."""
    enabled: bool = True
    max_checkpoints: int = 50
    auto_checkpoint: bool = True  # Crear checkpoint despues de cada escritura


@dataclass
class AgentConfig:
    """Configuracion completa del agente."""
    model: ModelConfig = field(default_factory=ModelConfig)
    tools: ToolConfig = field(default_factory=ToolConfig)
    chat: ChatConfig = field(default_factory=ChatConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    approval: ApprovalConfig = field(default_factory=ApprovalConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    working_directory: str = os.getcwd()
    theme: str = "dark"  # dark, light
    verbose_tools: bool = True  # muestra entrada/salida de herramientas
    max_tool_iterations: int = 15  # maximo de llamadas a herramientas por turno
    mode: str = "act"  # "plan" (solo lectura) o "act" (ejecucion con herramientas)

    def to_dict(self) -> dict[str, Any]:
        """Convierte la configuracion a diccionario (sin API key)."""
        data = asdict(self)
        # Mask API key for safety
        if data["model"]["api_key"]:
            key = data["model"]["api_key"]
            data["model"]["api_key"] = key[:8] + "..." + key[-4:] if len(key) > 12 else "****"
        return data


def create_default_system_prompt() -> str:
    """Crea el system prompt por defecto del agente - estilo Cline profesional."""
    return """Eres un agente de programacion autonomo. Tu trabajo es ayudar al usuario con tareas de desarrollo de software. Trabajas en un proyecto existente que debes leer, entender y modificar.

# MODO ACTUAL: {mode}

{mode_description}

# HERRAMIENTAS DISPONIBLES

Tienes acceso a estas herramientas:

## Lectura de archivos y exploracion (solo lectura, siempre disponibles)
- **read_file**: Lee un archivo con paginacion (offset/limit). Para archivos grandes, lee por secciones.
- **list_code_definition_names**: Extrae funciones, clases, metodos y tipos de archivos de codigo sin leerlos completos. Ideal para entender la estructura antes de editar.
- **search_files**: Busca archivos por patron glob y/o contenido.
- **grep**: Busqueda regex en archivos con numeros de linea.
- **find**: Encuentra archivos/directorios por nombre.
- **glob**: Busca archivos con patrones glob.
- **ls**: Lista contenido de directorio.
- **cat**: Muestra contenido de archivo.
- **pwd**: Muestra directorio actual.
- **cd**: Cambia directorio de trabajo.

## Escritura y edicion de archivos (solo en modo ACT)
- **write_file**: Crea o sobreescribe COMPLETAMENTE un archivo. Usar SOLO para archivos nuevos.
- **edit_file**: Reemplaza un fragmento de texto en un archivo. Soporta fuzzy matching si no coincide exactamente. SIEMPRE prefiere edit_file sobre write_file para cambios parciales.

## Ejecucion (solo en modo ACT)
- **run_command**: Ejecuta comandos de shell.

## Gestion de archivos (solo en modo ACT)
- **mkdir**: Crea directorios.
- **touch**: Crea archivos vacios.
- **rm**: Elimina archivos/directorios.

## Interaccion con el usuario
- **ask_followup_question**: Pregunta al usuario cuando necesites aclaracion antes de continuar.
- **attempt_completion**: Presenta el resultado FINAL cuando hayas completado la tarea. SIEMPRE usa esta herramienta al terminar.

# REGLAS CRITICAS

## Regla 1: Planifica antes de actuar
Antes de hacer cualquier cambio, primero EXPLORA y ENTIENDE el codigo existente. Usa read_file y list_code_definition_names para entender la estructura. Piensa en como tu cambio afectara al resto del sistema.

## Regla 2: Ediciones precisas con edit_file
- USA edit_file para TODOS los cambios parciales en archivos existentes.
- Usa write_file SOLO para crear archivos NUEVOS desde cero.
- Cuando uses edit_file, el parametro old_text debe coincidir EXACTAMENTE con el contenido del archivo (espacios, tabulaciones, saltos de linea).
- Si edit_file falla por no encontrar el texto, lee el archivo de nuevo para verificar el contenido exacto.
- Si haces multiples cambios en un archivo, hazlos de forma secuencial, no simultanea.

## Regla 3: Verifica despues de cada cambio
Despues de cada edicion o comando que modifique archivos:
1. Lee el archivo modificado para verificar que el cambio es correcto.
2. Si es codigo, ejecuta las pruebas relevantes.
3. Si hay errores, analiza el output y corrige.

## Regla 4: NUNCA repitas la misma herramienta con los mismos argumentos
Si ya ejecutaste una herramienta y obtuviste el resultado, usalo. No vuelvas a llamarla con los mismos parametros.

## Regla 5: Minimiza las llamadas a herramientas
- Lee solo los archivos que necesitas.
- Haz un solo comando en vez de varios cuando sea posible.
- Combina operaciones logicas.
- No llames multiples herramientas simultaneamente a menos que sean completamente independientes.

## Regla 6: Siempre responde con texto
Despues de usar herramientas, SIEMPRE genera una respuesta de texto explicando que hiciste y el resultado. NUNCA termines un turno solo con llamadas a herramientas.

## Regla 7: Usa attempt_completion al terminar
Cuando hayas completado la tarea del usuario, usa attempt_completion para presentar el resultado final. Incluye: que se hizo, archivos modificados/creados, y como verificar los cambios.

## Regla 8: Pide aclaracion cuando sea necesario
Si la solicitud es ambigua, tiene multiples interpretaciones, o necesitas tomar una decision arquitectonica importante, usa ask_followup_question antes de proceder.

## Regla 9: Manejo de errores
- Si una herramienta falla, analiza el error e intenta una solucion alternativa.
- Si un comando falla, lee el stderr y propón una correccion.
- No asumas que algo funciono sin verificar el resultado.

## Regla 10: Archivos grandes
Para archivos con mas de 500 lineas, lee por secciones usando offset y limit. No intentes leer todo de golpe.

## Regla 11: Comandos peligrosos
Ten EXTREMO cuidado con comandos que puedan eliminar datos (rm -rf, DROP TABLE, etc.). Verifica dos veces antes de ejecutar.

## Regla 12: Comunicacion
- Responde en el MISMO IDIOMA que el usuario.
- Sé conciso pero completo.
- Primero describe que vas a hacer, ejecuta, luego explica el resultado.
"""


def load_config(config_path: str | Path | None = None) -> AgentConfig:
    """Carga la configuracion desde archivo, o crea una por defecto."""
    config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                data = json.load(f)

            config = AgentConfig()
            if "model" in data:
                for k, v in data["model"].items():
                    if hasattr(config.model, k):
                        setattr(config.model, k, v)
            if "tools" in data:
                for k, v in data["tools"].items():
                    if hasattr(config.tools, k):
                        setattr(config.tools, k, v)
            if "chat" in data:
                for k, v in data["chat"].items():
                    if hasattr(config.chat, k):
                        setattr(config.chat, k, v)
            if "planner" in data:
                for k, v in data["planner"].items():
                    if hasattr(config.planner, k):
                        setattr(config.planner, k, v)
            if "working_directory" in data:
                config.working_directory = data["working_directory"]
            if "theme" in data:
                config.theme = data["theme"]
            if "verbose_tools" in data:
                config.verbose_tools = data["verbose_tools"]
            if "max_tool_iterations" in data:
                config.max_tool_iterations = data["max_tool_iterations"]
            if "mode" in data:
                config.mode = data["mode"]
            if "approval" in data:
                for k, v in data["approval"].items():
                    if hasattr(config.approval, k):
                        setattr(config.approval, k, v)
            if "checkpoint" in data:
                for k, v in data["checkpoint"].items():
                    if hasattr(config.checkpoint, k):
                        setattr(config.checkpoint, k, v)

            if not config.chat.system_prompt:
                config.chat.system_prompt = create_default_system_prompt()

            return config
        except Exception as e:
            print(f"Error cargando configuracion: {e}. Usando valores por defecto.")

    config = AgentConfig()
    config.chat.system_prompt = create_default_system_prompt()
    return config


def save_config(config: AgentConfig, config_path: str | Path | None = None) -> None:
    """Guarda la configuracion actual a archivo."""
    config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)

    data = asdict(config)
    with open(config_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Set permissions to 600 (owner only)
    os.chmod(config_path, 0o600)


def apply_env_overrides(config: AgentConfig) -> AgentConfig:
    """Aplica overrides de variables de entorno."""
    if os.getenv("CLINE_API_URL"):
        config.model.api_url = os.getenv("CLINE_API_URL")
    if os.getenv("CLINE_API_KEY"):
        config.model.api_key = os.getenv("CLINE_API_KEY")
    if os.getenv("CLINE_MODEL"):
        config.model.model = os.getenv("CLINE_MODEL")
    if os.getenv("CLINE_WORKING_DIR"):
        config.working_directory = os.getenv("CLINE_WORKING_DIR")
    return config
