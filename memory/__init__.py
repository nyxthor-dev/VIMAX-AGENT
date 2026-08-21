"""Sistema de Memoria del Agente Cline.

Fase 1 — Implementacion completa:
- ShortTermMemory: LRU de ultimas N interacciones + compresion inteligente
- LongTermMemory: persistencia JSON por proyecto
- MemoryManager: orquestacion de carga/guardado automatico
- LearnedPatterns: deteccion automatica de patrones error->fix
"""

from .short_term import ShortTermMemory, ShortTermConfig
from .long_term import (
    LongTermMemory,
    ProjectMemory,
    UserPreferences,
    LearnedPattern,
    ToolUsageStat,
)
from .manager import MemoryManager, MemoryManagerConfig

__all__ = [
    "ShortTermMemory",
    "ShortTermConfig",
    "LongTermMemory",
    "MemoryManager",
    "MemoryManagerConfig",
    "ProjectMemory",
    "UserPreferences",
    "LearnedPattern",
    "ToolUsageStat",
]
