#!/usr/bin/env python3
"""Cline Agent - Punto de entrada principal.

Uso:
    python main.py                    # Modo interactivo
    python main.py --api-key SK-...   # Con API key
    python main.py --model gpt-4o     # Con modelo especifico
    python main.py --cwd /path        # Directorio de trabajo
    python main.py --single "mensaje"  # Un solo mensaje
    python main.py --setup            # Configuracion inicial interactiva
    python main.py --test-api         # Probar conexion API
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cline Agent - Agente de codigo autonomo con IA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  %(prog)s                              Modo interactivo
  %(prog)s --single "crea un hello.py"  Un solo mensaje
  %(prog)s --setup                       Configuracion inicial
  %(prog)s --test-api                    Probar API
  %(prog)s --model gpt-4o --api-key SK-  Con opciones
        """,
    )

    # API Options
    parser.add_argument(
        "--api-url",
        default=None,
        help="URL base de la API (default: desde config o env CLINE_API_URL)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (default: desde config o env CLINE_API_KEY)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Modelo a usar (default: desde config o env CLINE_MODEL)",
    )

    # Agent Options
    parser.add_argument(
        "--cwd",
        default=None,
        help="Directorio de trabajo del agente",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximo de iteraciones de herramientas por turno (default: 15)",
    )
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="Desactivar todas las herramientas (solo chat)",
    )

    # Chat Options
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="System prompt personalizado",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperatura de generacion (0.0 - 2.0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximo de tokens a generar",
    )

    # Modes
    parser.add_argument(
        "--single",
        default=None,
        help="Ejecutar un solo mensaje y salir",
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Ejecutar configuracion inicial interactiva",
    )
    parser.add_argument(
        "--test-api",
        action="store_true",
        help="Probar la conexion con la API",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Listar modelos disponibles en la API",
    )
    parser.add_argument(
        "--config-file",
        default=None,
        help="Ruta al archivo de configuracion",
    )

    args = parser.parse_args()

    # Add project root to path for imports
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    # Import after path is set
    from cline_agent import (
        APIClient,
        AgentConfig,
        ClineAgent,
        ModelConfig,
        ToolRegistry,
        apply_env_overrides,
        load_config,
        save_config,
    )

    # Handle special modes first
    if args.setup:
        run_setup()
        return

    # Load config
    config = load_config(args.config_file)

    # Apply CLI overrides
    if args.api_url:
        config.model.api_url = args.api_url
    if args.api_key:
        config.model.api_key = args.api_key
    if args.model:
        config.model.model = args.model
    if args.cwd:
        config.working_directory = os.path.abspath(args.cwd)
    if args.max_iterations:
        config.max_tool_iterations = args.max_iterations
    if args.system_prompt:
        config.chat.system_prompt = args.system_prompt
    if args.temperature is not None:
        config.model.temperature = args.temperature
    if args.max_tokens is not None:
        config.model.max_tokens = args.max_tokens

    # Apply env overrides
    config = apply_env_overrides(config)

    # Check for API key
    if not config.model.api_key:
        print("\033[91mError: No se encontro API key.\033[0m")
        print("Usa --api-key, CLINE_API_KEY env var, o ejecuta --setup para configurar.")
        sys.exit(1)

    # Test API mode
    if args.test_api:
        api = APIClient(config.model)
        success, msg = api.test_connection()
        if success:
            print(f"\033[92m{msg}\033[0m")
        else:
            print(f"\033[91m{msg}\033[0m")
            sys.exit(1)
        api.close()
        return

    # List models mode
    if args.list_models:
        api = APIClient(config.model)
        models = api.list_models()
        if models:
            print("Modelos disponibles:")
            for m in models:
                marker = " (current)" if m == config.model.model else ""
                print(f"  {m}{marker}")
        else:
            print("No se pudieron obtener los modelos.")
        api.close()
        return

    # Create agent
    agent = ClineAgent(config)

    if args.no_tools:
        agent.tools = ToolRegistry.__new__(ToolRegistry)
        agent.tools._tools = {}

    # Run mode
    if args.single:
        response = agent.run_single(args.single)
        print(response)
        agent.close()
    else:
        agent.run_interactive()


def run_setup() -> None:
    """Ejecuta la configuracion inicial interactiva."""
    from cline_agent.config import DEFAULT_CONFIG_PATH, create_default_system_prompt, save_config
    from cline_agent import AgentConfig

    print("\033[96m")
    print("  ╔═══════════════════════════════════════╗")
    print("  ║       CLINE AGENT - Configuracion      ║")
    print("  ╚═══════════════════════════════════════╝")
    print("\033[0m")
    print()

    config = AgentConfig()

    # API URL
    default_url = "https://vimax-ia.p.jo3.org/v1/"
    url = input(f"  API URL [{default_url}]: ").strip()
    config.model.api_url = url if url else default_url

    # API Key
    key = input("  API Key: ").strip()
    if not key:
        key = os.getenv("CLINE_API_KEY", "")
    config.model.api_key = key

    # Model
    default_model = "gpt-4o-mini"
    model = input(f"  Modelo [{default_model}]: ").strip()
    config.model.model = model if model else default_model

    # Working directory
    cwd = input(f"  Directorio de trabajo [{os.getcwd()}]: ").strip()
    config.working_directory = os.path.abspath(cwd) if cwd else os.getcwd()

    # System prompt
    use_default = input("  Usar system prompt por defecto? [Y/n]: ").strip().lower()
    if use_default in ("n", "no"):
        print("  Escribe el system prompt (presiona Enter dos veces para terminar):")
        lines = []
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        config.chat.system_prompt = "\n".join(lines).strip()
    else:
        config.chat.system_prompt = create_default_system_prompt()

    # Save
    save_config(config)
    print(f"\n  \033[92mConfiguracion guardada en: {DEFAULT_CONFIG_PATH}\033[0m")

    # Test connection
    if config.model.api_key:
        print("\n  Probando conexion...")
        from cline_agent import APIClient
        api = APIClient(config.model)
        success, msg = api.test_connection()
        if success:
            print(f"  \033[92m{msg}\033[0m")
        else:
            print(f"  \033[91m{msg}\033[0m")
        api.close()

    print()
    print("  Listo! Ejecuta el agente con:")
    print(f"    python main.py")
    print()


if __name__ == "__main__":
    main()
