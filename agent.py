"""Motor principal del agente Cline.

Orquesta el loop de chat, ejecucion de herramientas, y gestion del flujo
agente-herramientas-respuesta.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from .api_client import APIClient, APIError, ChatHistory, Message
from .config import AgentConfig, create_default_system_prompt, load_config, save_config
from .tools import ToolRegistry, ToolResult
from .model_validator import ModelValidator, ValidationResult
from .planner import TaskPlanner, TaskPlan
from .approval import ApprovalManager, ApprovalConfig as ApprovalCfg
from .checkpoint import CheckpointManager
from .token_counter import TokenCounter
from rich.box import ROUNDED
from rich.console import Console
from rich.table import Table

from .utils.display import (
    GRN, GRY, RED, WHT, CYN, YLW, BOLD, DIM,
    STATUS_OK, STATUS_ERR, STATUS_INFO,
    c, icon, print_tool_call, print_tool_result, print_assistant_text,
    print_assistant_thinking, print_user_input, print_separator,
    print_status, print_banner, print_help, print_tool_list,
    print_unknown_command, print_confirm_action, print_prompt_view,
    print_settings_menu, print_settings_model_menu, print_settings_chat_menu,
    print_settings_tools_menu, print_settings_general_menu,
    print_model_info, print_project_info, print_prompt_editor_header,
    print_settings_value_set, print_settings_value_view,
    print_plan, print_task_progress, print_task_start, print_task_result,
    print_plan_summary, print_model_validation,
    get_input, get_input_with_default, get_input_multiline,
    api_spinner, task_progress_bar,
    console,
)


class ClineAgent:
    """Agente de codigo autonomo estilo Cline.

    Maneja el ciclo completo:
    1. Recibe input del usuario
    2. Envia a la API con herramientas disponibles
    3. Si la API pide una herramienta, la ejecuta
    4. Repite hasta que la API responda solo con texto
    5. Muestra la respuesta al usuario
    """

    # Modos del agente
    MODE_CHAT = "chat"
    MODE_SETTINGS = "settings"
    MODE_PROMPT_EDITOR = "prompt_editor"

    # Modos de operacion (estilo Cline)
    OP_MODE_PLAN = "plan"  # Solo lectura, sin herramientas de escritura
    OP_MODE_ACT = "act"  # Ejecucion completa con todas las herramientas

    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or load_config()
        self.api = APIClient(self.config.model)
        self.history = ChatHistory(
            system_prompt=self._build_system_prompt(),
            max_messages=self.config.chat.max_history_messages,
        )
        self.tools = ToolRegistry()
        self._running = False
        self._mode = self.MODE_CHAT
        # Ruta "home" del proyecto (la que se establece con /project)
        self._project_home: str | None = None
        # Estado del modo settings
        self._settings_submenu: str | None = None
        self._prompt_editor_buffer: str | None = None
        # Model Validator (fuzzy matching + cache)
        self.validator = ModelValidator(self.api)
        # Task Planner (solo se activa para tareas complejas)
        self.planner = TaskPlanner(self.api, self.config) if self.config.planner.enabled else None
        # Validar modelo al inicio
        self._model_validated = False
        # Sistema de Plan/Act (estilo Cline)
        self._op_mode = self.config.mode  # "plan" o "act"
        # Sistema de Checkpoints (estilo Cline)
        self._checkpoint = None
        if self.config.checkpoint.enabled:
            self._checkpoint = CheckpointManager(self.config.working_directory)
            if not self._checkpoint.initialize():
                print_status("Checkpoints desactivados (git no disponible)", "warn")
                self._checkpoint = None
        # Sistema de Auto-Approve (estilo Cline)
        approval_cfg = ApprovalCfg(
            auto_approve_reads=self.config.approval.auto_approve_reads,
            auto_approve_writes=self.config.approval.auto_approve_writes,
            auto_approve_commands=self.config.approval.auto_approve_commands,
            auto_approve_browser=self.config.approval.auto_approve_browser,
            auto_approve_mcp=self.config.approval.auto_approve_mcp,
            yolo_mode=self.config.approval.yolo_mode,
            always_allow_commands=self.config.approval.always_allow_commands,
        )
        self._approval = ApprovalManager(approval_cfg)
        # Token counter
        self._token_counter = TokenCounter(self.config.model.model)
        # Herramientas de escritura (bloqueadas en modo plan)
        self._write_tools = {
            "write_file", "edit_file", "mkdir", "touch", "rm", "run_command",
        }
        self._setup_tools()

    def _build_system_prompt(self) -> str:
        """Construye el system prompt con el modo actual inyectado."""
        base = self.config.chat.system_prompt
        if not base:
            base = create_default_system_prompt()

        # Inyectar modo actual
        op_mode = self._op_mode
        if op_mode == self.OP_MODE_PLAN:
            mode_desc = (
                "ESTAS EN MODO PLAN (solo lectura). NO uses herramientas de escritura "
                "(write_file, edit_file, mkdir, touch, rm, run_command). "
                "Tu trabajo es LEER, ANALIZAR y PROPONER un plan paso a paso. "
                "NO crees ni modifiques archivos. NO ejecutes comandos. "
                "Solo usa herramientas de lectura (read_file, ls, grep, find, glob, search_files, "
                "list_code_definition_names) y ask_followup_question."
            )
        else:
            mode_desc = (
                "ESTAS EN MODO ACT (ejecucion completa). Puedes usar TODAS las herramientas "
                "disponibles incluyendo escritura y ejecucion de comandos. Cada cambio "
                "puede requerir aprobacion del usuario segun la configuracion."
            )

        if "{mode}" in base:
            return base.format(mode=op_mode.upper(), mode_description=mode_desc)
        return base + f"\n\n# MODO ACTUAL: {op_mode.upper()}\n{mode_desc}", 

    def _setup_tools(self) -> None:
        """Registra todas las herramientas integradas."""
        from .tools import (
            CdTool, CatTool, EditFileTool, FindTool, GlobTool,
            GrepTool, LsTool, MkdirTool, PwdTool, ReadFileTool,
            RmTool, RunCommandTool, SearchFilesTool, TouchTool, WriteFileTool,
            ListCodeDefinitionsTool, AskFollowupQuestionTool, AttemptCompletionTool,
        )

        cwd = self.config.working_directory
        cwd_getter = lambda: self.config.working_directory

        cd_tool = CdTool(cwd=cwd)
        self.tools.register(cd_tool)
        # Sync cwd after cd changes
        original_cd_execute = cd_tool.execute
        def cd_execute_cwd_sync(**kwargs: Any) -> ToolResult:
            result = original_cd_execute(**kwargs)
            if result.success:
                self.config.working_directory = cd_tool.cwd
            return result
        cd_tool.execute = cd_execute_cwd_sync

        # Lectura
        self.tools.register(LsTool(cwd_getter=cwd_getter))
        self.tools.register(CatTool(cwd_getter=cwd_getter))
        self.tools.register(PwdTool(cwd_getter=cwd_getter))
        self.tools.register(ReadFileTool(cwd_getter=cwd_getter))
        self.tools.register(SearchFilesTool(cwd_getter=cwd_getter))
        self.tools.register(GrepTool(cwd_getter=cwd_getter))
        self.tools.register(FindTool(cwd_getter=cwd_getter))
        self.tools.register(GlobTool(cwd_getter=cwd_getter))
        self.tools.register(ListCodeDefinitionsTool(cwd_getter=cwd_getter))

        # Escritura y ejecucion
        self.tools.register(MkdirTool(cwd_getter=cwd_getter))
        self.tools.register(TouchTool(cwd_getter=cwd_getter))
        self.tools.register(RmTool(cwd_getter=cwd_getter))
        self.tools.register(WriteFileTool(cwd_getter=cwd_getter))
        self.tools.register(EditFileTool(cwd_getter=cwd_getter))
        self.tools.register(RunCommandTool(cwd_getter=cwd_getter))

        # Herramientas interactivas del agente (estilo Cline)
        self.tools.register(AskFollowupQuestionTool(input_callback=self._agent_ask_user))
        self.tools.register(AttemptCompletionTool())

    def _agent_ask_user(self, question: str) -> str:
        """Callback para ask_followup_question - pregunta al usuario en la terminal."""
        print_separator()
        console.print(f"  {CYN}? Pregunta del agente:{WHT}")
        console.print(f"  {question}")
        response = get_input("> ")
        return response

    def _execute_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Ejecuta una herramienta y retorna el resultado como string.

        Incluye:
        - Verificacion de modo Plan (bloquea escritura)
        - Sistema de aprobacion (auto-approve por categoria)
        - Sistema de checkpoints (captura estado despues de escrituras)
        - Contador de tokens
        """
        # 1. Verificar modo Plan
        if self._op_mode == self.OP_MODE_PLAN and name in self._write_tools:
            return json.dumps({
                "success": False,
                "error": (
                    f"Herramienta '{name}' BLOQUEADA en modo PLAN. "
                    f"Cambia a modo ACT con /mode act para usar herramientas de escritura/ejecucion."
                ),
            })

        tool = self.tools.get(name)
        if not tool:
            return json.dumps({"success": False, "error": f"Herramienta no encontrada: {name}"})

        # 2. Sistema de aprobacion
        approval = self._approval.classify_tool_call(name, arguments)
        if approval.requires_approval:
            print_separator()
            # Mostrar info de la accion
            danger_icon = "\U0001f534" if approval.danger_level == "dangerous" else "\U0001f7e1" if approval.danger_level == "caution" else "\U0001f7e2"
            console.print(f"  {danger_icon} {BOLD}{approval.action_type.upper()}{WHT} | {approval.reason}")
            if self.config.verbose_tools:
                print_tool_call(name, arguments)
            answer = print_confirm_action(f"\u00bfAprobar ejecucion de '{name}'?")
            if not answer:
                return json.dumps({"success": False, "error": "Accion rechazada por el usuario."})

        # 3. Validar argumentos
        errors = tool.validate(**arguments)
        if errors:
            return json.dumps({"success": False, "error": "; ".join(errors)})

        # 4. Ejecutar
        try:
            result = tool.execute(**arguments)
            self._token_counter.record_tool_call()

            # 5. Crear checkpoint despues de escrituras
            if result.success and name in self._write_tools and self._checkpoint:
                files = []
                if "file_path" in arguments:
                    from pathlib import Path as P
                    try:
                        rel = str(P(arguments["file_path"]).relative_to(self.config.working_directory))
                        files.append(rel)
                    except (ValueError, TypeError):
                        files.append(arguments["file_path"])
                self._checkpoint.create_checkpoint(
                    tool_name=name,
                    description=f"{name}: {str(arguments)[:80]}",
                    files_changed=files,
                )

            # 6. Mostrar indicador de aprobacion si fue auto-aprobada
            if approval.auto_approved and approval.action_type != "read":
                print_status(f"Auto-aprobada ({approval.action_type}): {name}", "info")

            return json.dumps(result.to_dict(), ensure_ascii=False)
        except Exception as e:
            return json.dumps({"success": False, "error": f"Error ejecutando {name}: {e}"})

    def _process_tool_calls(self, tool_calls: list[dict[str, Any]]) -> bool:
        """Procesa todas las tool calls con display formateado.

        Returns True si hubo tool calls procesadas.
        """
        if not tool_calls:
            return False

        self.history.add_assistant(tool_calls=tool_calls)

        for tc in tool_calls:
            fn = tc.get("function", {})
            tool_name = fn.get("name", "")
            tool_args_raw = fn.get("arguments", "{}")
            tool_call_id = tc.get("id", "")

            try:
                tool_args = json.loads(tool_args_raw) if isinstance(tool_args_raw, str) else tool_args_raw
            except json.JSONDecodeError:
                tool_args = {}

            if self.config.verbose_tools:
                print_tool_call(tool_name, tool_args)

            result_str = self._execute_tool(tool_name, tool_args)

            if self.config.verbose_tools:
                print_tool_result(tool_name, result_str)

            self.history.add_tool_result(
                tool_call_id=tool_call_id,
                name=tool_name,
                content=result_str,
            )

        return True

    @staticmethod
    def _tool_signature(tool_calls: list[dict[str, Any]]) -> str:
        """Genera firma hashable de un lote de tool calls."""
        sigs = []
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            raw_args = fn.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                args = {}
            sig = f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
            sigs.append(sig)
        return "|".join(sigs)

    def process_turn(self, user_input: str) -> str:
        """Procesa un turno completo del usuario.

        Puede involucrar multiples idas y vueltas con la API
        (cuando la IA decide usar herramientas).

        Returns:
            La respuesta final de texto del asistente.
        """
        self.history.add_user(user_input)

        # Auto-compress history
        if self.config.chat.auto_compress_history:
            if len(self.history) >= self.config.chat.compress_threshold:
                removed = self.history.compress()
                if removed > 0:
                    print_status(f"Historial comprimido: {removed} mensajes", "info")

        tool_schemas = self.tools.get_schemas()
        max_iterations = self.config.max_tool_iterations
        iteration = 0
        final_text = ""

        # Anti-repetition: track recent tool call signatures
        recent_sigs: list[str] = []
        max_repeats = 1

        while iteration < max_iterations:
            iteration += 1
            api_messages = self.history.get_api_messages()

            try:
                with api_spinner(f"Llamando API (turno {iteration}/{max_iterations})..."):
                    response = self.api.chat_completion(
                        messages=api_messages,
                        tools=tool_schemas,
                    )
            except APIError as e:
                print_status(f"Error de API: {e}", "err")
                return f"Error de API: {e}"

            try:
                choice = response["choices"][0]
                message = choice.get("message", {})
                content = message.get("content", "") or ""
                tool_calls = message.get("tool_calls", []) or []
            except (KeyError, IndexError) as e:
                print_status(f"Respuesta inesperada de la API: {e}", "err")
                return f"Respuesta inesperada: {e}"

            if tool_calls:
                sig = self._tool_signature(tool_calls)
                repeat_count = sum(1 for s in recent_sigs if s == sig)

                if repeat_count >= max_repeats:
                    print_status(
                        f"Herramienta repetida detectada ({tool_calls[0].get('function', {}).get('name', '?')}), forzando respuesta de texto",
                        "info",
                    )
                    if (self.history._messages
                            and self.history._messages[-1].role == "tool"):
                        self.history._messages.pop()
                    if (self.history._messages
                            and self.history._messages[-1].role == "assistant"):
                        self.history._messages.pop()

                    try:
                        with api_spinner("Forzando respuesta de texto..."):
                            final_response = self.api.chat_completion(
                                messages=self.history.get_api_messages(),
                                tools=None,
                            )
                        final_msg = final_response["choices"][0].get("message", {})
                        final_text = final_msg.get("content", "") or "(el modelo no genero texto)"
                        self.history.add_assistant(content=final_text)
                    except Exception:
                        final_text = final_text or "(interrumpido por repeticion de herramienta)"
                    break

                recent_sigs.append(sig)
                if len(recent_sigs) > 6:
                    recent_sigs = recent_sigs[-6:]

                if content:
                    print_assistant_thinking(content)
                    final_text = content

                processed = self._process_tool_calls(tool_calls)
                if not processed:
                    break
                continue
            else:
                final_text = content
                self.history.add_assistant(content=content)
                break

        if iteration >= max_iterations:
            final_text += f"\n\n[red][Advertencia: maximo de {max_iterations} iteraciones alcanzado][/red]"
            self.history.add_assistant(content=final_text)

        return final_text

    # ── Set de comandos conocidos (para detectar desconocidos) ─────────
    _KNOWN_COMMANDS = frozenset({
        "/help", "/settings", "/models", "/tools", "/history",
        "/cwd", "/prompt", "/compact", "/clear",
        "/model", "/project",
        "/quit", "/exit", "/q", "/save",
        # Nuevos comandos (estilo Cline)
        "/mode", "/newtask", "/yolo", "/checkpoints", "/cost",
        "/approve", "/rollback",
        # Dentro de settings
        "model", "chat", "tools", "general", "prompt", "back", "save",
        "api_url", "api_key", "temperature", "max_tokens", "top_p",
        "max_history", "save_history", "auto_compress", "compress_at",
        "max_timeout", "max_file_read", "dangerous", "blocked",
        "directory", "theme", "verbose", "max_iter",
        # Dentro de prompt editor
        "/done", "/cancel", "/view", "/reset",
    })

    # ── Loop interactivo principal ────────────────────────────────────

    def run_interactive(self) -> None:
        """Ejecuta el agente en modo interactivo (REPL)."""
        self._running = True
        self._print_banner()

        while self._running:
            try:
                user_input = self._get_input()

                if not user_input.strip():
                    continue

                raw = user_input.strip()

                # Despachar segun el modo actual
                if self._mode == self.MODE_PROMPT_EDITOR:
                    self._handle_prompt_editor_input(raw)
                elif self._mode == self.MODE_SETTINGS:
                    self._handle_settings_input(raw)
                else:
                    self._handle_chat_input(raw)

            except KeyboardInterrupt:
                # Si estamos en modo settings o prompt_editor, volver a chat
                if self._mode != self.MODE_CHAT:
                    console.print(f"\n  [dim]Interrumpido. Volviendo al chat...[/dim]")
                    self._mode = self.MODE_CHAT
                    self._settings_submenu = None
                else:
                    console.print(f"\n[dim](Interrumpido. Escribe /quit para salir)[/dim]")
            except Exception as e:
                print_status(f"Error inesperado: {e}", "err")

        self.api.close()

    # ── Manejo de input segun modo ────────────────────────────────────

    def _handle_chat_input(self, raw: str) -> None:
        """Procesa input en modo chat normal."""
        cmd = raw.lower()

        if cmd.startswith("/"):
            cmd_word = cmd.split()[0]

            # Comando desconocido -> no enviar al LLM
            if cmd_word not in self._KNOWN_COMMANDS:
                print_unknown_command(raw.split()[0])
                return

            # ── SALIR ────────────────────────────────────────────
            if cmd_word in ("/quit", "/exit", "/q"):
                # Si hay cambios sin guardar, preguntar
                console.print(f"\n  [dim]Hasta luego![/dim]")
                self._running = False
                return

            # ── PROYECTO ────────────────────────────────────────
            if cmd_word == "/project":
                self._cmd_project(raw)
                return

            # ── MODO PLAN/ACT (estilo Cline) ───────────────────
            if cmd_word == "/mode":
                self._cmd_mode(raw)
                return

            # ── NEWTASK (destilar contexto) ─────────────────────
            if cmd_word == "/newtask":
                self._cmd_newtask(raw)
                return

            # ── YOLO MODE ──────────────────────────────────────
            if cmd_word == "/yolo":
                self._cmd_yolo()
                return

            # ── CHECKPOINTS ────────────────────────────────────
            if cmd_word == "/checkpoints":
                self._cmd_checkpoints()
                return

            # ── ROLLBACK ────────────────────────────────────────
            if cmd_word == "/rollback":
                self._cmd_rollback(raw)
                return

            # ── COST ────────────────────────────────────────────
            if cmd_word == "/cost":
                self._cmd_cost()
                return

            # ── APPROVE (configurar auto-approve) ───────────────
            if cmd_word == "/approve":
                self._cmd_approve(raw)
                return

            # ── INFO (solo lectura, 0 tokens) ───────────────────
            if cmd_word == "/help":
                print_help(current_model=self.config.model.model)
                return
            if cmd_word == "/models":
                self._list_models()
                return
            if cmd_word == "/tools":
                self._print_tools()
                return
            if cmd_word == "/history":
                self._print_history()
                return
            if cmd_word == "/cwd":
                console.print(f"  [dim]{self.config.working_directory}[/dim]")
                return
            if cmd_word == "/prompt":
                print_prompt_view(self.config.chat.system_prompt)
                return

            # ── CHAT (gestion local, 0 tokens) ──────────────────
            if cmd_word == "/compact":
                removed = self.history.compress()
                print_status(f"Comprimido: {removed} mensajes eliminados", "ok")
                return
            if cmd_word == "/clear":
                if len(self.history) == 0:
                    print_status("El historial ya esta vacio", "info")
                    return
                resp = print_confirm_action("Limpiar todo el historial?")
                if resp == "y":
                    self.history.clear()
                    print_status("Historial limpiado", "ok")
                else:
                    print_status("Cancelado", "info")
                return

            # ── MODEL (atajo rapido) ─────────────────────────────
            if cmd_word == "/model":
                self._cmd_model(raw)
                return

            # ── SETTINGS (entra al menu interactivo) ────────────
            if cmd_word == "/settings":
                # Ver si trae subcomando directo
                parts = raw.split(None, 1)
                sub = parts[1].strip().lower() if len(parts) > 1 else None
                if sub == "prompt":
                    # prompt es especial: va directo al editor
                    self._mode = self.MODE_SETTINGS
                    self._settings_submenu = None
                    self._enter_prompt_editor()
                elif sub in ("model", "chat", "tools", "general"):
                    self._mode = self.MODE_SETTINGS
                    self._settings_submenu = sub
                    self._print_settings_submenu()
                elif sub == "save":
                    self._save_settings()
                elif sub:
                    print_status(f"Submenu desconocido: {sub}", "err")
                    console.print(f"  [dim]Opciones: model, chat, tools, general, prompt, save[/dim]")
                else:
                    # Sin subcomando: mostrar menu principal
                    self._mode = self.MODE_SETTINGS
                    self._settings_submenu = None
                    print_settings_menu()
                return

            # Fallback
            print_unknown_command(cmd_word)
            return

        # ── Mensaje normal -> enviar al LLM ───────────────────────
        # Mostrar modo actual en el banner del turno
        print()

        # Validar modelo antes de enviar
        if not self._validate_before_send():
            print_status("Modelo no validado. Usa /model para corregir.", "err")
            return

        # Intentar ejecutar via planificador (solo tareas complejas)
        planned_result = self._execute_with_planner(raw)
        if planned_result is not None:
            print_assistant_text(planned_result)
            return

        # Flujo normal: un solo turno
        response = self.process_turn(raw)
        if response:
            print_assistant_text(response)

    def _handle_settings_input(self, raw: str) -> None:
        """Procesa input en modo settings. Nada llega al LLM."""
        choice = raw.lower().strip()

        # Comandos universales dentro de settings
        if choice in ("/quit", "/exit", "/q"):
            self._mode = self.MODE_CHAT
            self._settings_submenu = None
            print_status("Volviendo al chat", "info")
            return

        if choice.startswith("/"):
            # Dentro de settings, / vuelve al chat o ejecuta comando de salida
            cmd_word = choice.split()[0]
            if cmd_word == "/help":
                print_help(current_model=self.config.model.model)
                return
            if cmd_word == "/save":
                self._save_settings()
                return
            if cmd_word == "/settings":
                # /settings dentro de settings: volver al menu principal
                parts = raw.split(None, 1)
                sub = parts[1].strip().lower() if len(parts) > 1 else None
                if sub == "prompt":
                    self._enter_prompt_editor()
                elif sub in ("model", "chat", "tools", "general"):
                    self._settings_submenu = sub
                    self._print_settings_submenu()
                elif sub == "save":
                    self._save_settings()
                else:
                    # Sin sub o desconocido: menu principal
                    self._settings_submenu = None
                    print_settings_menu()
                return
            if cmd_word == "/project":
                self._mode = self.MODE_CHAT
                self._settings_submenu = None
                self._cmd_project(raw)
                return
            # Cualquier otro / es desconocido en modo settings
            print_unknown_command(cmd_word)
            return

        # Si estamos en el menu principal de settings
        if self._settings_submenu is None:
            if choice == "back":
                self._mode = self.MODE_CHAT
                print_status("Volviendo al chat", "info")
            elif choice == "save":
                self._save_settings()
            elif choice == "model":
                self._settings_submenu = "model"
                self._print_settings_submenu()
            elif choice == "chat":
                self._settings_submenu = "chat"
                self._print_settings_submenu()
            elif choice == "tools":
                self._settings_submenu = "tools"
                self._print_settings_submenu()
            elif choice == "general":
                self._settings_submenu = "general"
                self._print_settings_submenu()
            elif choice == "prompt":
                self._enter_prompt_editor()
            else:
                print_status(f"Opcion desconocida: {choice}", "err")
                console.print(f"  [dim]Opciones: model, chat, tools, general, prompt, save, back[/dim]")
            return

        # Submenu especifico
        if self._settings_submenu == "model":
            self._handle_settings_model(choice)
        elif self._settings_submenu == "chat":
            self._handle_settings_chat(choice)
        elif self._settings_submenu == "tools":
            self._handle_settings_tools(choice)
        elif self._settings_submenu == "general":
            self._handle_settings_general(choice)
        else:
            # Safety net: submenu invalido, volver al menu principal
            print_status(f"Submenu invalido: {self._settings_submenu}", "err")
            self._settings_submenu = None
            print_settings_menu()

    def _handle_prompt_editor_input(self, raw: str) -> None:
        """Procesa input en el editor de prompt. Nada llega al LLM."""
        lower = raw.lower().strip()

        if lower == "/done":
            # Terminar edicion y aplicar
            if self._prompt_editor_buffer is not None:
                new_prompt = self._prompt_editor_buffer.strip()
                if new_prompt:
                    old_len = len(self.config.chat.system_prompt)
                    self.config.chat.system_prompt = new_prompt
                    self.history.system_prompt = new_prompt
                    print_status(f"System prompt actualizado ({old_len} -> {len(new_prompt)} chars)", "ok")
                    self._auto_save()
                else:
                    print_status("Prompt vacio, no se aplicaron cambios", "info")
            self._mode = self.MODE_SETTINGS
            self._settings_submenu = None
            self._prompt_editor_buffer = None
            print_settings_menu()
            return

        if lower == "/cancel":
            print_status("Edicion cancelada", "info")
            self._mode = self.MODE_SETTINGS
            self._settings_submenu = None
            self._prompt_editor_buffer = None
            print_settings_menu()
            return

        if lower == "/view":
            print_prompt_view(self.config.chat.system_prompt)
            return

        if lower == "/reset":
            from .config import create_default_system_prompt
            resp = print_confirm_action("Restaurar el system prompt al original?")
            if resp == "y":
                default = create_default_system_prompt()
                self.config.chat.system_prompt = default
                self.history.system_prompt = default
                self._prompt_editor_buffer = None
                print_status("System prompt restaurado al original", "ok")
                self._auto_save()
            return

        if lower.startswith("/"):
            print_status(f"Comando no reconocido en el editor. Usa /done, /cancel, /view o /reset", "err")
            return

        # Acumular linea al buffer
        if self._prompt_editor_buffer is None:
            self._prompt_editor_buffer = ""
        if self._prompt_editor_buffer:
            self._prompt_editor_buffer += "\n" + raw
        else:
            self._prompt_editor_buffer = raw

    # ── Comando /project ─────────────────────────────────────────────

    def _cmd_project(self, raw: str) -> None:
        """Maneja el comando /project."""
        parts = raw.split(None, 1)
        arg = parts[1].strip() if len(parts) > 1 else None

        # Sin argumentos: mostrar info del proyecto actual
        if not arg:
            print_project_info(
                self.config.working_directory,
                self._project_home,
            )
            console.print("  [dim]Para cambiar: [yellow]/project <ruta>[/yellow][/dim]")
            console.print("  [dim]Para volver al inicio del proyecto: [yellow]/project --home[/yellow][/dim]")
            console.print()
            return

        # --home: volver al la ruta del proyecto
        if arg.lower() == "--home":
            if self._project_home:
                old = self.config.working_directory
                self.config.working_directory = self._project_home
                # Re-setup tools con el nuevo cwd
                self._rebuild_tools()
                print_status(f"Directorio: {old}", "info")
                print_status(f"-> Proyecto: {self._project_home}", "ok")
                self._auto_save()
            else:
                print_status("No se ha establecido un proyecto home. Usa /project <ruta>", "info")
            return

        # Ruta especificada: cambiar directorio
        path = os.path.expanduser(arg)
        abs_path = os.path.abspath(path)

        if not os.path.isdir(abs_path):
            print_status(f"El directorio no existe: {abs_path}", "err")
            return

        old = self.config.working_directory

        # Si es el primer /project con ruta, guardar como home
        if self._project_home is None:
            self._project_home = abs_path

        self.config.working_directory = abs_path
        self._rebuild_tools()

        print_status(f"Directorio: {old}", "info")
        print_status(f"-> Proyecto: {abs_path}", "ok")
        if self._project_home == abs_path:
            console.print("  [dim](Establecido como proyecto home. Usa [yellow]/project --home[/yellow] para volver)[/dim]")
        self._auto_save()

    def _rebuild_tools(self) -> None:
        """Reconstruye las herramientas con el cwd actualizado."""
        self.tools = ToolRegistry()
        self._setup_tools()

    # ── Comando /model (atajo rapido) ────────────────────────────────

    def _cmd_model(self, raw: str) -> None:
        """Maneja el comando /model con flags."""
        parts = raw.split(None, 1)
        arg = parts[1].strip() if len(parts) > 1 else None

        # Sin argumentos o --list: listar modelos
        if not arg or arg == "--list":
            self._list_models()
            return

        # --info: info detallada del modelo actual
        if arg == "--info":
            print_model_info(
                model_name=self.config.model.model,
                api_url=self.config.model.api_url,
                api_key=self.config.model.api_key,
                temperature=self.config.model.temperature,
                max_tokens=self.config.model.max_tokens,
                top_p=self.config.model.top_p,
            )
            return

        # --set <nombre>: cambiar modelo
        if arg.startswith("--set "):
            model_name = arg[6:].strip()
            if not model_name:
                print_status("Uso: /model --set <nombre>", "err")
                return
            self._set_model(model_name)
            return

        # --temp <valor>: cambiar temperatura
        if arg.startswith("--temp "):
            val_str = arg[7:].strip()
            try:
                val = float(val_str)
                if not (0.0 <= val <= 2.0):
                    raise ValueError
            except ValueError:
                print_status("Temperatura debe ser un numero entre 0.0 y 2.0", "err")
                return
            old = self.config.model.temperature
            self.config.model.temperature = val
            print_settings_value_set("temperature", str(old), str(val))
            self._auto_save()
            return

        # --max-tokens <valor>: cambiar max tokens
        if arg.startswith("--max-tokens "):
            val_str = arg[13:].strip()
            try:
                val = int(val_str)
                if val < 1:
                    raise ValueError
            except ValueError:
                print_status("max_tokens debe ser un entero positivo", "err")
                return
            old = self.config.model.max_tokens
            self.config.model.max_tokens = val
            print_settings_value_set("max_tokens", str(old), str(val))
            self._auto_save()
            return

        # Argumento directo: nombre de modelo
        self._set_model(arg)

    def _set_model(self, model_name: str) -> None:
        """Valida y establece un modelo."""
        result = self.validator.validate(model_name)
        if not result.valid:
            print_model_validation(result)
            if not result.suggestions:
                return
            # Si hay sugerencias, intentar autocorregir
            corrected, was_fixed = self.validator.auto_correct(model_name)
            if was_fixed:
                model_name = corrected
            else:
                # Mostrar sugerencias y no cambiar
                return

        old = self.config.model.model
        self.config.model.model = model_name
        # Reconectar API y validator con el nuevo modelo
        self.api = APIClient(self.config.model)
        self.validator = ModelValidator(self.api)
        self._model_validated = True
        print_settings_value_set("modelo", old, model_name)
        self._auto_save()

    # ── Submenu: settings/model ──────────────────────────────────────

    def _handle_settings_model(self, choice: str) -> None:
        """Maneja el submenu de configuracion de modelo."""
        m = self.config.model

        if choice == "back":
            self._settings_submenu = None
            print_settings_menu()
        elif choice == "model":
            new = get_input_with_default("Modelo", m.model)
            if new != m.model:
                self._set_model(new)
        elif choice == "api_url":
            new = get_input_with_default("API URL", m.api_url)
            if new != m.api_url:
                old = m.api_url
                m.api_url = new
                # Reconectar API y validator
                self.api = APIClient(self.config.model)
                self.validator = ModelValidator(self.api)
                self._model_validated = False
                print_settings_value_set("api_url", old, new)
                self._auto_save()
        elif choice == "api_key":
            try:
                new = console.input("  [bold yellow]>[/bold yellow] [white]API Key[/white]: ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print()
                return
            if new:
                m.api_key = new
                self.api = APIClient(self.config.model)
                self.validator = ModelValidator(self.api)
                self._model_validated = False
                print_status("API key actualizada", "ok")
                self._auto_save()
        elif choice == "temperature":
            new_str = get_input_with_default("Temperatura (0.0 - 2.0)", str(m.temperature))
            try:
                new_val = float(new_str)
                if not (0.0 <= new_val <= 2.0):
                    print_status("Debe estar entre 0.0 y 2.0", "err")
                    return
                old = m.temperature
                m.temperature = new_val
                print_settings_value_set("temperature", str(old), str(new_val))
                self._auto_save()
            except ValueError:
                print_status("Valor invalido", "err")
        elif choice == "max_tokens":
            new_str = get_input_with_default("Max tokens", str(m.max_tokens))
            try:
                new_val = int(new_str)
                if new_val < 1:
                    raise ValueError
                old = m.max_tokens
                m.max_tokens = new_val
                print_settings_value_set("max_tokens", str(old), str(new_val))
                self._auto_save()
            except ValueError:
                print_status("Debe ser un entero positivo", "err")
        elif choice == "top_p":
            new_str = get_input_with_default("Top P (0.0 - 1.0)", str(m.top_p))
            try:
                new_val = float(new_str)
                if not (0.0 <= new_val <= 1.0):
                    print_status("Debe estar entre 0.0 y 1.0", "err")
                    return
                old = m.top_p
                m.top_p = new_val
                print_settings_value_set("top_p", str(old), str(new_val))
                self._auto_save()
            except ValueError:
                print_status("Valor invalido", "err")
        else:
            print_status(f"Opcion desconocida: {choice}", "err")
            self._print_settings_submenu()

    # ── Submenu: settings/chat ───────────────────────────────────────

    def _handle_settings_chat(self, choice: str) -> None:
        """Maneja el submenu de configuracion de chat."""
        ch = self.config.chat

        if choice == "back":
            self._settings_submenu = None
            print_settings_menu()
        elif choice == "max_history":
            new_str = get_input_with_default("Max mensajes en historial", str(ch.max_history_messages))
            try:
                new_val = int(new_str)
                if new_val < 1:
                    raise ValueError
                old = ch.max_history_messages
                ch.max_history_messages = new_val
                self.history.max_messages = new_val
                print_settings_value_set("max_history", str(old), str(new_val))
                self._auto_save()
            except ValueError:
                print_status("Debe ser un entero positivo", "err")
        elif choice == "save_history":
            new_str = get_input_with_default("Guardar historial (true/false)", str(ch.save_history))
            if new_str.lower() in ("true", "1", "yes", "si", "y"):
                new_val = True
            elif new_str.lower() in ("false", "0", "no", "n"):
                new_val = False
            else:
                print_status("Usa true/false, yes/no, o 1/0", "err")
                return
            old = ch.save_history
            ch.save_history = new_val
            print_settings_value_set("save_history", str(old), str(new_val))
            self._auto_save()
        elif choice == "auto_compress":
            new_str = get_input_with_default("Auto-comprimir (true/false)", str(ch.auto_compress_history))
            if new_str.lower() in ("true", "1", "yes", "si", "y"):
                new_val = True
            elif new_str.lower() in ("false", "0", "no", "n"):
                new_val = False
            else:
                print_status("Usa true/false, yes/no, o 1/0", "err")
                return
            old = ch.auto_compress_history
            ch.auto_compress_history = new_val
            print_settings_value_set("auto_compress", str(old), str(new_val))
            self._auto_save()
        elif choice == "compress_at":
            new_str = get_input_with_default("Umbral de compresion (mensajes)", str(ch.compress_threshold))
            try:
                new_val = int(new_str)
                if new_val < 2:
                    raise ValueError
                old = ch.compress_threshold
                ch.compress_threshold = new_val
                print_settings_value_set("compress_at", str(old), str(new_val))
                self._auto_save()
            except ValueError:
                print_status("Debe ser un entero >= 2", "err")
        else:
            print_status(f"Opcion desconocida: {choice}", "err")
            self._print_settings_submenu()

    # ── Submenu: settings/tools ──────────────────────────────────────

    def _handle_settings_tools(self, choice: str) -> None:
        """Maneja el submenu de configuracion de herramientas."""
        t = self.config.tools

        if choice == "back":
            self._settings_submenu = None
            print_settings_menu()
        elif choice == "max_timeout":
            new_str = get_input_with_default("Timeout maximo (segundos)", str(t.max_command_timeout))
            try:
                new_val = int(new_str)
                if new_val < 1:
                    raise ValueError
                old = t.max_command_timeout
                t.max_command_timeout = new_val
                print_settings_value_set("max_timeout", f"{old}s", f"{new_val}s")
                self._auto_save()
            except ValueError:
                print_status("Debe ser un entero positivo (segundos)", "err")
        elif choice == "max_file_read":
            new_str = get_input_with_default("Max lineas lectura archivo", str(t.max_file_size_read))
            try:
                new_val = int(new_str)
                if new_val < 1:
                    raise ValueError
                old = t.max_file_size_read
                t.max_file_size_read = new_val
                print_settings_value_set("max_file_read", f"{old} lineas", f"{new_val} lineas")
                self._auto_save()
            except ValueError:
                print_status("Debe ser un entero positivo", "err")
        elif choice == "dangerous":
            new_str = get_input_with_default("Permitir comandos peligrosos (true/false)", str(t.allow_dangerous_commands))
            if new_str.lower() in ("true", "1", "yes", "si", "y"):
                new_val = True
            elif new_str.lower() in ("false", "0", "no", "n"):
                new_val = False
            else:
                print_status("Usa true/false, yes/no, o 1/0", "err")
                return
            old = t.allow_dangerous_commands
            t.allow_dangerous_commands = new_val
            status_str = "[green]PERMITIDOS[/green]" if new_val else "[red]BLOQUEADOS[/red]"
            console.print(f"  [green]✓[/green] Comandos peligrosos: {status_str}")
            self._auto_save()
        elif choice == "blocked":
            console.print()
            bt = Table(
                title=f"[bold bright_cyan]Comandos bloqueados[/bold bright_cyan] [dim]({len(t.blocked_commands)})[/dim]",
                box=ROUNDED, border_style="cyan",
                show_header=False, padding=(0, 2), expand=False,
            )
            bt.add_column("#", style="yellow", min_width=4, justify="right")
            bt.add_column("Patron", style="white", min_width=40)
            for i, cmd in enumerate(t.blocked_commands, 1):
                bt.add_row(str(i), cmd)
            console.print(bt)
            console.print("  [dim]Para agregar: escribe el patron (ej: rm -rf *)[/dim]")
            console.print("  [dim]Para eliminar: escribe el numero (ej: 2)[/dim]")
            console.print("  [dim]Para salir: escribe [yellow]back[/yellow][/dim]")

            action = get_input("")
            if not action or action.lower() == "back":
                return
            # Si es un numero, eliminar de la lista
            if action.isdigit():
                idx = int(action) - 1
                if 0 <= idx < len(t.blocked_commands):
                    removed = t.blocked_commands.pop(idx)
                    print_status(f"Eliminado: {removed}", "ok")
                    self._auto_save()
                else:
                    print_status("Numero fuera de rango", "err")
            else:
                # Agregar patron
                t.blocked_commands.append(action)
                print_status(f"Agregado: {action}", "ok")
                self._auto_save()
        else:
            print_status(f"Opcion desconocida: {choice}", "err")
            self._print_settings_submenu()

    # ── Submenu: settings/general ────────────────────────────────────

    def _handle_settings_general(self, choice: str) -> None:
        """Maneja el submenu de configuracion general."""

        if choice == "back":
            self._settings_submenu = None
            print_settings_menu()
        elif choice == "directory":
            new = get_input_with_default("Directorio de trabajo", self.config.working_directory)
            abs_path = os.path.abspath(os.path.expanduser(new))
            if os.path.isdir(abs_path):
                old = self.config.working_directory
                self.config.working_directory = abs_path
                if self._project_home is None:
                    self._project_home = abs_path
                self._rebuild_tools()
                print_settings_value_set("directorio", old, abs_path)
                self._auto_save()
            else:
                print_status(f"Directorio no existe: {abs_path}", "err")
        elif choice == "theme":
            new = get_input_with_default("Tema (dark/light)", self.config.theme)
            if new.lower() in ("dark", "light"):
                old = self.config.theme
                self.config.theme = new.lower()
                print_settings_value_set("theme", old, new.lower())
                self._auto_save()
            else:
                print_status("Tema debe ser 'dark' o 'light'", "err")
        elif choice == "verbose":
            new_str = get_input_with_default("Verbose tools (true/false)", str(self.config.verbose_tools))
            if new_str.lower() in ("true", "1", "yes", "si", "y"):
                new_val = True
            elif new_str.lower() in ("false", "0", "no", "n"):
                new_val = False
            else:
                print_status("Usa true/false, yes/no, o 1/0", "err")
                return
            old = self.config.verbose_tools
            self.config.verbose_tools = new_val
            print_settings_value_set("verbose_tools", str(old), str(new_val))
            self._auto_save()
        elif choice == "max_iter":
            new_str = get_input_with_default("Max iteraciones por turno", str(self.config.max_tool_iterations))
            try:
                new_val = int(new_str)
                if new_val < 1:
                    raise ValueError
                old = self.config.max_tool_iterations
                self.config.max_tool_iterations = new_val
                print_settings_value_set("max_tool_iterations", str(old), str(new_val))
                self._auto_save()
            except ValueError:
                print_status("Debe ser un entero positivo", "err")
        else:
            print_status(f"Opcion desconocida: {choice}", "err")
            self._print_settings_submenu()

    # ── Editor de prompt ─────────────────────────────────────────────

    def _enter_prompt_editor(self) -> None:
        """Entra al modo editor de system prompt."""
        self._mode = self.MODE_PROMPT_EDITOR
        self._prompt_editor_buffer = self.config.chat.system_prompt
        print_prompt_editor_header()
        console.print("  [dim]Contenido actual del prompt (edita o reemplaza):[/dim]")
        if self._prompt_editor_buffer:
            for line in self._prompt_editor_buffer.split("\n"):
                console.print(f"  [dim]{line}[/dim]")
        print_separator()

    # ── Utilidades de settings ───────────────────────────────────────

    def _print_settings_submenu(self) -> None:
        """Muestra el submenu de settings correspondiente."""
        # Primero mostrar valores actuales
        self._print_current_settings_values()
        # Luego el menu
        if self._settings_submenu == "model":
            print_settings_model_menu()
        elif self._settings_submenu == "chat":
            print_settings_chat_menu()
        elif self._settings_submenu == "tools":
            print_settings_tools_menu()
        elif self._settings_submenu == "general":
            print_settings_general_menu()

    def _print_current_settings_values(self) -> None:
        """Muestra los valores actuales del submenu activo con Rich Table."""
        sub = self._settings_submenu
        if not sub:
            return

        table = Table(box=None, show_header=False, padding=(0, 0), expand=False)
        table.add_column("key", style="dim", min_width=20)
        table.add_column("value", min_width=40)

        if sub == "model":
            m = self.config.model
            table.add_row("modelo", f"[yellow]{m.model}[/yellow]")
            table.add_row("api_url", f"[white]{m.api_url}[/white]")
            table.add_row("temperature", f"[white]{m.temperature}[/white]")
            table.add_row("max_tokens", f"[white]{m.max_tokens}[/white]")
            table.add_row("top_p", f"[white]{m.top_p}[/white]")
        elif sub == "chat":
            ch = self.config.chat
            table.add_row("max_history", f"[white]{ch.max_history_messages}[/white]")
            sh_style = "green" if ch.save_history else "red"
            table.add_row("save_history", f"[{sh_style}]{ch.save_history}[/{sh_style}]")
            ac_style = "green" if ch.auto_compress_history else "red"
            table.add_row("auto_compress", f"[{ac_style}]{ch.auto_compress_history}[/{ac_style}]")
            table.add_row("compress_at", f"[white]{ch.compress_threshold} msgs[/white]")
        elif sub == "tools":
            t = self.config.tools
            table.add_row("max_timeout", f"[white]{t.max_command_timeout}s[/white]")
            table.add_row("max_file_read", f"[white]{t.max_file_size_read} lineas[/white]")
            d_style = "green" if t.allow_dangerous_commands else "red"
            table.add_row("dangerous", f"[{d_style}]{t.allow_dangerous_commands}[/{d_style}]")
            table.add_row("blocked", f"[white]{len(t.blocked_commands)} patrones[/white]")
        elif sub == "general":
            table.add_row("directory", f"[white]{self.config.working_directory}[/white]")
            table.add_row("theme", f"[white]{self.config.theme}[/white]")
            v_style = "green" if self.config.verbose_tools else "red"
            table.add_row("verbose_tools", f"[{v_style}]{self.config.verbose_tools}[/{v_style}]")
            table.add_row("max_iterations", f"[white]{self.config.max_tool_iterations}[/white]")

        console.print(table)
        print_separator()

    def _save_settings(self) -> None:
        """Guarda la configuracion actual a archivo."""
        try:
            save_config(self.config)
            print_status(f"Configuracion guardada", "ok")
        except Exception as e:
            print_status(f"Error guardando: {e}", "err")

    def _auto_save(self) -> None:
        """Guarda automaticamente la configuracion (silencioso)."""
        try:
            save_config(self.config)
        except Exception:
            pass  # Fallo silencioso, no interrumpir al usuario

    # ── Validacion inteligente de modelo ─────────────────────────────

    def _validate_model_on_startup(self) -> None:
        """Valida el modelo al iniciar. Si hay sugerencia unica, la aplica."""
        if self._model_validated:
            return
        with api_spinner(f"Validando modelo {self.config.model.model}..."):
            result = self.validator.validate(self.config.model.model)
        if result.valid:
            self._model_validated = True
            return
        # Si hay 1 sugerencia clara, autocorregir
        corrected, was_fixed = self.validator.auto_correct(self.config.model.model)
        if was_fixed:
            old = self.config.model.model
            self.config.model.model = corrected
            self.api = APIClient(self.config.model)
            self.validator = ModelValidator(self.api)
            print_status(f"Modelo corregido: {old} -> {corrected}", "ok")
            self._auto_save()
            self._model_validated = True
        else:
            print_model_validation(result)
            self._model_validated = False

    def _validate_before_send(self) -> bool:
        """Valida el modelo antes de enviar. Retorna True si es seguro enviar.

        Se ejecuta solo la primera vez (usa cache de validacion).
        Si el modelo es invalido, intenta autocorregir.
        Si no puede, bloquea el envio.
        """
        if self._model_validated:
            return True
        self._validate_model_on_startup()
        return self._model_validated

    # ── Agent Loop con Planificador ─────────────────────────────────

    def _execute_with_planner(self, user_input: str) -> str | None:
        """Intenta ejecutar la solicitud via planificador.

        Returns:
            La respuesta final, o None si no se uso el planificador.
        """
        if not self.planner:
            return None
        if not self.config.planner.enabled:
            return None
        # No planificar si esta deshabilitado o si el mensaje es corto
        if len(user_input.strip()) < 15:
            return None
        # Detectar si necesita planificacion
        with api_spinner("Analizando complejidad de la tarea..."):
            needs_plan = self.planner.should_plan(user_input)
        if not needs_plan:
            return None
        # Generar plan
        with api_spinner("Generando plan de tareas..."):
            plan = self.planner.create_plan(
                user_input,
                cwd=self.config.working_directory,
            )
        if plan is None or not plan.tasks:
            print_status("No se pudo generar un plan. Ejecutando normalmente.", "info")
            return None
        # Solo 1 tarea? No vale la pena el plan, ejecutar directamente
        if len(plan.tasks) == 1:
            return None
        # Mostrar plan
        print_plan(plan)
        # Pedir confirmacion (a menos que auto_execute este activado)
        if not self.config.planner.auto_execute:
            resp = print_confirm_action(
                f"Ejecutar {len(plan.tasks)} tareas?"
            )
            if resp != "y":
                print_status("Plan cancelado. Ejecutando como mensaje normal.", "info")
                return None
        # Ejecutar tareas secuencialmente (agent loop)
        console.print()
        all_results = []
        with task_progress_bar(len(plan.tasks), "Ejecutando plan") as pbar:
            task = plan.advance()  # primera tarea
            while task is not None:
                print_task_start(task)
                if task.title:
                    pbar.update_description(f"T{task.number}. {task.title}")
                print_task_progress(plan)

                # Construir prompt para esta tarea:
                # Incluye la solicitud original y el contexto de las tareas anteriores
                context_parts = [f"Solicitud original: {user_input}"]
                # Resultados de tareas anteriores como contexto
                for prev in plan.tasks:
                    if prev.status == "done" and prev.result:
                        context_parts.append(
                            f"Tarea {prev.number} ({prev.title}) completada. Resultado: {prev.result[:500]}"
                        )
                context = "\n".join(context_parts)
                task_prompt = (
                    f"{context}\n\n"
                    f"Ejecuta esta tarea: {task.title}. {task.description}"
                )

                try:
                    result = self.process_turn(task_prompt)
                    task.mark_done(result)
                    print_task_result(task)
                    all_results.append(result)
                except Exception as e:
                    task.mark_error(str(e))
                    print_task_result(task)
                    # Si una tarea falla, preguntar si continuar
                    resp = print_confirm_action("Tarea fallo. Continuar con las restantes?")
                    if resp != "y":
                        plan.skip_remaining("Tarea anterior fallo")
                        break

                pbar.advance()
                task = plan.advance()  # siguiente tarea

        # Resumen final
        print_task_progress(plan)
        print_plan_summary(plan)

        # Generar respuesta final resumiendo todo
        done_count = sum(1 for t in plan.tasks if t.status == "done")
        error_count = sum(1 for t in plan.tasks if t.status == "error")
        if done_count == len(plan.tasks):
            summary = f"Todas las {len(plan.tasks)} tareas completadas exitosamente."
        elif error_count == len(plan.tasks):
            summary = f"Todas las tareas fallaron."
        else:
            summary = f"{done_count} de {len(plan.tasks)} tareas completadas."
            if error_count > 0:
                summary += f" {error_count} con errores."

        return summary + "\n\n" + "\n".join(all_results)

    # ── Funciones de display originales ──────────────────────────────

    def _print_banner(self) -> None:
        """Muestra el banner de inicio."""
        print_banner(
            version="v1.3",
            model=self.config.model.model,
            api_url=self.config.model.api_url,
            cwd=self.config.working_directory,
            n_tools=len(self.tools),
        )
        # Mostrar modo actual (Plan/Act)
        mode_label = self._op_mode.upper()
        mode_color = YLW if self._op_mode == "plan" else GRN
        console.print(f"  Modo: [{mode_color}bold]{mode_label}[/{mode_color}bold]  [dim](cambia con /mode plan|act)[/dim]")
        # Mostrar estado de aprobacion
        if self._approval.config.yolo_mode:
            console.print(f"  Aprobacion: {RED}YOLO MODE{WHT} {RED}\U0001f525{WHT}")
        elif not self.config.approval.auto_approve_writes:
            console.print(f"  Aprobacion: escrituras requieren confirmacion [dim](/approve writes para auto)[/dim]")
        # Checkpoints
        if self._checkpoint:
            console.print(f"  Checkpoints: {GRN}activos{WHT} [dim](/checkpoints, /rollback <n>)[/dim]")
        # Validacion inteligente del modelo al inicio
        self._validate_model_on_startup()

    def _get_input(self) -> str:
        """Obtiene input del usuario con prompt coloreado segun el modo."""
        if self._mode == self.MODE_SETTINGS:
            if self._settings_submenu:
                prompt_str = f"[bold cyan]settings[/bold cyan]/[yellow]{self._settings_submenu}[/yellow][dim]> [/dim]"
            else:
                prompt_str = "[bold cyan]settings[/bold cyan][dim]> [/dim]"
        elif self._mode == self.MODE_PROMPT_EDITOR:
            prompt_str = "[bold magenta]prompt[/bold magenta][dim]> [/dim]"
        else:
            prompt_str = "[bold green]> Tu[/bold green] "
        try:
            return console.input(prompt_str).strip()
        except EOFError:
            return "/quit"

    def _print_tools(self) -> None:
        """Lista las herramientas con formato."""
        schemas = self.tools.get_schemas()
        print_tool_list(schemas)

    def _print_history(self) -> None:
        """Muestra el historial del chat con Rich Table."""
        n = len(self.history)
        console.print()
        console.print(f"  [bold bright_cyan]Historial[/bold bright_cyan] [dim]({n} mensajes)[/dim]")
        print_separator()
        for msg in self.history.messages:
            role = msg.role
            if role == "user":
                console.print(f"  [bold green]> Tu[/bold green] [white]{msg.content[:200]}[/white]")
            elif role == "assistant":
                if msg.content:
                    console.print(f"  [cyan]> IA[/cyan] [dim]{msg.content[:200]}[/dim]")
                for tc in (msg.tool_calls or []):
                    fn = tc.get("function", {})
                    console.print(f"  [dim]  \u2699 {fn.get('name', '?')}[/dim]")
            elif role == "tool":
                content = (msg.content or "")[:120]
                ico = msg.name or "tool"
                console.print(f"  [dim]  \u2713 {ico}[/dim] [dim]{content}[/dim]")
        print_separator()
        console.print()

    def _print_settings(self) -> None:
        """Muestra la configuracion actual con Rich Table."""
        m = self.config.model
        t = self.config.tools
        ch = self.config.chat

        table = Table(
            title="[bold bright_cyan]Configuracion actual[/bold bright_cyan]",
            box=ROUNDED, border_style="cyan",
            show_header=False, padding=(0, 2), expand=False,
        )
        table.add_column("key", style="dim", min_width=20)
        table.add_column("value", min_width=50)

        # Modelo
        table.add_section()
        table.add_row("[bold]Modelo[/bold]", "")
        table.add_row("modelo", f"[yellow]{m.model}[/yellow]")
        table.add_row("api_url", f"[white]{m.api_url}[/white]")
        masked = m.api_key[:8] + "..." + m.api_key[-4:] if len(m.api_key) > 12 else "****"
        table.add_row("api_key", f"[white]{masked}[/white]")
        table.add_row("temperature", f"[white]{m.temperature}[/white]")
        table.add_row("max_tokens", f"[white]{m.max_tokens}[/white]")
        table.add_row("top_p", f"[white]{m.top_p}[/white]")

        # Herramientas
        table.add_section()
        table.add_row("[bold]Herramientas[/bold]", "")
        table.add_row("max_timeout", f"[white]{t.max_command_timeout}s[/white]")
        table.add_row("max_file_read", f"[white]{t.max_file_size_read} lineas[/white]")
        d_style = "green" if t.allow_dangerous_commands else "red"
        table.add_row("dangerous_cmds", f"[{d_style}]{t.allow_dangerous_commands}[/{d_style}]")
        table.add_row("blocked", f"[dim]{len(t.blocked_commands)} patrones[/dim]")

        # Chat
        table.add_section()
        table.add_row("[bold]Chat[/bold]", "")
        table.add_row("max_history", f"[white]{ch.max_history_messages}[/white]")
        sh_style = "green" if ch.save_history else "red"
        table.add_row("save_history", f"[{sh_style}]{ch.save_history}[/{sh_style}]")
        ac_style = "green" if ch.auto_compress_history else "red"
        table.add_row("auto_compress", f"[{ac_style}]{ch.auto_compress_history}[/{ac_style}]")
        table.add_row("compress_at", f"[white]{ch.compress_threshold} msgs[/white]")
        sp = ch.system_prompt[:60] + "..." if len(ch.system_prompt) > 60 else ch.system_prompt
        table.add_row("system_prompt", f"[dim]{sp}[/dim]")

        # General
        table.add_section()
        table.add_row("[bold]General[/bold]", "")
        table.add_row("directorio", f"[white]{self.config.working_directory}[/white]")
        home = str(self._project_home) or "[dim](no establecido)[/dim]"
        table.add_row("proyecto_home", f"[white]{home}[/white]")
        table.add_row("theme", f"[white]{self.config.theme}[/white]")
        v_style = "green" if self.config.verbose_tools else "red"
        table.add_row("verbose_tools", f"[{v_style}]{self.config.verbose_tools}[/{v_style}]")
        table.add_row("max_iterations", f"[white]{self.config.max_tool_iterations}[/white]")

        console.print()
        console.print(table)
        console.print("  [dim]Usa [yellow]/settings[/yellow] para editar. Usa [yellow]/settings save[/yellow] para guardar.[/dim]")
        console.print()

    def _list_models(self) -> None:
        """Lista modelos disponibles."""
        with api_spinner("Consultando modelos disponibles..."):
            models = self.api.list_models()
        if models:
            table_data = []
            for m in models:
                marker = " (current)" if m == self.config.model.model else ""
                table_data.append((m, marker))
            table = Table(
                title="[bold bright_cyan]Modelos disponibles[/bold bright_cyan]",
                box=ROUNDED, border_style="cyan",
                show_header=False, padding=(0, 2), expand=False,
            )
            table.add_column("Modelo", style="yellow", min_width=30)
            table.add_column("", style="dim", min_width=10)
            for m, marker in table_data:
                style = "bold green" if marker else "yellow"
                table.add_row(m, f"[green]{marker}[/green]" if marker else "")
            console.print()
            console.print(table)
            if len(models) > 15:
                console.print(f"  [dim]... {len(models)} modelos en total[/dim]")
        else:
            print_status("No se pudieron obtener los modelos", "err")
        console.print()

    def run_single(self, user_input: str) -> str:
        """Ejecuta una sola consulta y retorna la respuesta (modo no interactivo)."""
        return self.process_turn(user_input)

    # ── Nuevos comandos (estilo Cline) ──────────────────────────

    def _cmd_mode(self, raw: str) -> None:
        """Cambia entre modo Plan y Act (estilo Cline)."""
        parts = raw.split(None, 1)
        if len(parts) < 2 or parts[1].lower() not in ("plan", "act"):
            current = self._op_mode.upper()
            console.print(f"  Modo actual: [bold]{current}[/bold]")
            console.print(f"  Uso: /mode plan  (solo lectura, sin escritura/ejecucion)")
            console.print(f"       /mode act   (ejecucion completa con herramientas)")
            return

        new_mode = parts[1].lower()
        if new_mode == self._op_mode:
            console.print(f"  Ya estas en modo [bold]{new_mode.upper()}[/bold]")
            return

        self._op_mode = new_mode
        self.config.mode = new_mode
        save_config(self.config)
        # Reconstruir system prompt con nuevo modo
        self.history._system_prompt = self._build_system_prompt()

        if new_mode == "plan":
            console.print(f"  {YLW}\u26a0  MODO PLAN activado.{WHT}")
            console.print(f"  Solo lectura. No se pueden escribir archivos ni ejecutar comandos.")
            console.print(f"  Usa [bold]/mode act[/bold] para volver a la ejecucion.")
        else:
            console.print(f"  {GRN}\u2713  MODO ACT activado.{WHT}")
            console.print(f"  Ejecucion completa. Todas las herramientas disponibles.")

    def _cmd_newtask(self, raw: str) -> None:
        """Destila el contexto actual y empieza una tarea fresca (estilo Cline new_task).

        Resume la conversacion actual en un resumen compacto, limpia el historial,
        y lo reemplaza con el resumen + contexto de archivos mencionados.
        """
        if len(self.history) < 4:
            print_status("Historial muy corto para destilar. Usa /clear en su lugar.", "info")
            return

        print_status("Destilando contexto...", "info")

        # Extraer preguntas del usuario y respuestas del asistente
        user_queries = []
        assistant_summaries = []
        tools_used = set()
        for msg in self.history._messages:
            if msg.role == "user":
                content = msg.content or ""
                user_queries.append(content[:200])
            elif msg.role == "assistant":
                content = msg.content or ""
                if content:
                    assistant_summaries.append(content[:300])
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        fn = tc.get("function", {})
                        tools_used.add(fn.get("name", "?"))

        # Construir resumen
        summary_parts = [
            "[Contexto de tarea anterior]",
            "",
            f"Consultas del usuario ({len(user_queries)}):",
        ]
        for i, q in enumerate(user_queries[-5:], 1):
            summary_parts.append(f"  {i}. {q}")

        if assistant_summaries:
            summary_parts.append("")
            summary_parts.append("Respuestas/recuerdos clave:")
            for s in assistant_summaries[-3:]:
                summary_parts.append(f"  - {s}")

        if tools_used:
            summary_parts.append("")
            summary_parts.append(f"Herramientas usadas: {', '.join(sorted(tools_used))}")

        summary_parts.append("")
        summary_parts.append("[Fin del contexto anterior. Nueva tarea a continuacion.]")

        summary = "\n".join(summary_parts)
        old_count = len(self.history)

        # Limpiar y dejar solo el resumen
        self.history.clear()
        self.history.add_user(summary)

        print_status(
            f"Nueva tarea iniciada. {old_count} mensajes destilados a 1 resumen.",
            "ok",
        )

    def _cmd_yolo(self) -> None:
        """Activa/desactiva YOLO mode (aprobacion automatica total).

        PELIGROSO: aprueba TODAS las acciones sin pedir confirmacion.
        """
        if self._approval.config.yolo_mode:
            self._approval.config.yolo_mode = False
            self.config.approval.yolo_mode = False
            save_config(self.config)
            console.print(f"  {GRN}\u2713  YOLO mode DESACTIVADO.{WHT}")
            console.print(f"  Las aprobaciones estan de vuelta.")
        else:
            console.print(f"  {RED}\u26a0  ADVERTENCIA: YOLO mode aprueba TODAS las acciones.{WHT}")
            console.print(f"  Incluyendo escritura de archivos, eliminacion y comandos de shell.")
            console.print(f"  {RED}Esto es PELIGROSO en proyectos reales.{WHT}")
            answer = print_confirm_action("Activar YOLO mode?")
            if answer == "y":
                self._approval.config.yolo_mode = True
                self.config.approval.yolo_mode = True
                save_config(self.config)
                console.print(f"  {RED}\U0001f525 YOLO mode ACTIVADO. Sin red de seguridad.{WHT}")
            else:
                console.print(f"  Cancelado. Las aprobaciones siguen activas.")

    def _cmd_checkpoints(self) -> None:
        """Muestra la lista de checkpoints disponibles (estilo Cline)."""
        if not self._checkpoint:
            print_status("Checkpoints no disponibles (git no encontrado o desactivado)", "err")
            return

        checkpoints = self._checkpoint.list_checkpoints(limit=20)
        if not checkpoints:
            console.print(f"  [dim]No hay checkpoints todavia.{WHT}")
            console.print(f"  Los checkpoints se crean automaticamente al editar archivos.")
            return

        console.print()
        table = Table(
            title="[bold bright_cyan]Checkpoints[/bold bright_cyan]",
            box=ROUNDED, border_style="cyan",
            show_header=True, padding=(0, 1), expand=False,
        )
        table.add_column("#", style="dim", justify="right", width=4)
        table.add_column("Tiempo", style="dim", width=19)
        table.add_column("Herramienta", style="cyan", width=16)
        table.add_column("Descripcion", style="white")
        table.add_column("Hash", style="dim", width=8)

        for cp in checkpoints:
            table.add_row(
                str(cp["id"]),
                cp["timestamp"],
                cp["tool"],
                cp["description"][:50],
                cp["hash"],
            )

        console.print(table)
        console.print(f"  [dim]Usa /rollback <numero> para restaurar.{WHT}")
        console.print(f"  Total: {self._checkpoint.total_checkpoints} checkpoints")
        console.print()

    def _cmd_rollback(self, raw: str) -> None:
        """Restaura archivos a un checkpoint especifico."""
        if not self._checkpoint:
            print_status("Checkpoints no disponibles", "err")
            return

        parts = raw.split(None, 1)
        if len(parts) < 2 or not parts[1].isdigit():
            console.print(f"  Uso: /rollback <numero_checkpoint>")
            console.print(f"  Usa /checkpoints para ver los numeros disponibles.")
            return

        checkpoint_id = int(parts[1])
        console.print(f"  {YLW}\u26a0 Restaurando archivos al checkpoint #{checkpoint_id}...{WHT}")
        success, message = self._checkpoint.restore_files(checkpoint_id)
        if success:
            print_status(message, "ok")
        else:
            print_status(message, "err")

    def _cmd_cost(self) -> None:
        """Muestra estadisticas de tokens y costo de la sesion."""
        summary = self._token_counter.get_session_summary()
        console.print()
        console.print(f"  [bold bright_cyan]Estadisticas de sesion[/bold bright_cyan]")
        console.print(f"  {'─' * 40}")
        for line in summary.split("\n"):
            console.print(f"  {line}")
        console.print(f"  {'─' * 40}")
        console.print(f"  Modelo: {self.config.model.model}")
        console.print()

    def _cmd_approve(self, raw: str) -> None:
        """Configura auto-approve por categoria."""
        parts = raw.split(None, 1)
        if len(parts) < 2:
            self._show_approve_status()
            return

        sub = parts[1].lower()
        cfg = self._approval.config

        if sub == "reads":
            cfg.auto_approve_reads = not cfg.auto_approve_reads
        elif sub == "writes":
            cfg.auto_approve_writes = not cfg.auto_approve_writes
        elif sub == "commands":
            cfg.auto_approve_commands = not cfg.auto_approve_commands
        elif sub == "all":
            cfg.auto_approve_reads = True
            cfg.auto_approve_writes = True
            cfg.auto_approve_commands = True
        else:
            console.print(f"  Uso: /approve [reads|writes|commands|all]")
            return

        # Sync to config
        self.config.approval.auto_approve_reads = cfg.auto_approve_reads
        self.config.approval.auto_approve_writes = cfg.auto_approve_writes
        self.config.approval.auto_approve_commands = cfg.auto_approve_commands
        save_config(self.config)
        self._show_approve_status()

    def _show_approve_status(self) -> None:
        """Muestra el estado actual de auto-approve."""
        cfg = self._approval.config
        yolo = "\U0001f525 YOLO" if cfg.yolo_mode else "off"

        console.print()
        console.print(f"  [bold bright_cyan]Auto-Approve[/bold bright_cyan]")
        console.print(f"  {'─' * 35}")
        reads = f"{GRN}ON{WHT}" if cfg.auto_approve_reads else f"{RED}OFF{WHT}"
        writes = f"{GRN}ON{WHT}" if cfg.auto_approve_writes else f"{RED}OFF{WHT}"
        cmds = f"{GRN}ON{WHT}" if cfg.auto_approve_commands else f"{RED}OFF{WHT}"
        console.print(f"  Lecturas (read):       {reads}")
        console.print(f"  Escrituras (write):    {writes}")
        console.print(f"  Comandos (command):    {cmds}")
        console.print(f"  YOLO mode:             {yolo}")
        console.print(f"  {'─' * 35}")
        console.print(f"  [dim]/approve [reads|writes|commands|all] para toggle{WHT}")
        console.print()

    def close(self) -> None:
        """Cierra recursos del agente."""
        self.api.close()
        self._running = False
