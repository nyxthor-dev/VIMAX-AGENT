"""Herramientas integradas del agente Cline.

Incluye operaciones de sistema de archivos, shell, edicion de archivos,
herramientas de agente (ask_followup, attempt_completion), y
list_code_definition_names.
"""

from .base import BaseTool, ToolResult, ToolRegistry
from .filesystem import CdTool, LsTool, CatTool, MkdirTool, TouchTool, RmTool, PwdTool
from .file_ops import ReadFileTool, WriteFileTool, EditFileTool, SearchFilesTool
from .shell import RunCommandTool
from .code_tools import GrepTool, FindTool, GlobTool, ListCodeDefinitionsTool
from .agent_tools import AskFollowupQuestionTool, AttemptCompletionTool

__all__ = [
    "BaseTool",
    "ToolResult",
    "ToolRegistry",
    "CdTool",
    "LsTool",
    "CatTool",
    "MkdirTool",
    "TouchTool",
    "RmTool",
    "PwdTool",
    "ReadFileTool",
    "WriteFileTool",
    "EditFileTool",
    "SearchFilesTool",
    "RunCommandTool",
    "GrepTool",
    "FindTool",
    "GlobTool",
    "ListCodeDefinitionsTool",
    "AskFollowupQuestionTool",
    "AttemptCompletionTool",
]
