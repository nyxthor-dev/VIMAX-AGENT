"""Cline Agent - Agente de codigo autonomo en Python.

Un agente de IA que puede navegar archivos, ejecutar comandos, y escribir codigo
usando una API compatible con OpenAI.

Uso basico:
    from cline_agent import ClineAgent
    agent = ClineAgent()
    agent.run_interactive()

Uso con configuracion personalizada:
    from cline_agent import ClineAgent, AgentConfig, ModelConfig
    config = AgentConfig()
    config.model.api_key = "sk-..."
    config.model.model = "gpt-4o"
    agent = ClineAgent(config)
    agent.run_single("Crea un archivo hello.py")
"""

__version__ = "1.0.0"
__author__ = "Cline Agent"

from .agent import ClineAgent
from .api_client import APIClient, ChatHistory, Message
from .config import (
    AgentConfig,
    ChatConfig,
    ModelConfig,
    ToolConfig,
    apply_env_overrides,
    create_default_system_prompt,
    load_config,
    save_config,
)
from .tools import ToolRegistry

__all__ = [
    "ClineAgent",
    "APIClient",
    "ChatHistory",
    "Message",
    "AgentConfig",
    "ModelConfig",
    "ToolConfig",
    "ChatConfig",
    "ToolRegistry",
    "load_config",
    "save_config",
    "apply_env_overrides",
    "create_default_system_prompt",
]
