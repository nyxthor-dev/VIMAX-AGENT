"""Planificador de tareas para el agente Cline.

Detecta solicitudes complejas, genera un plan de tareas (max 7),
y las ejecuta secuencialmente con feedback de progreso.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


MAX_TASKS = 7


@dataclass
class Task:
    """Una tarea individual dentro de un plan."""
    number: int
    title: str
    description: str
    status: str = "pending"  # pending, in_progress, done, error, skipped
    result: str = ""

    def mark_done(self, result: str = "") -> None:
        """Marca esta tarea como completada."""
        self.status = "done"
        self.result = result

    def mark_error(self, result: str = "") -> None:
        """Marca esta tarea como error."""
        self.status = "error"
        self.result = result


class TaskPlan:
    """Un plan completo de tareas."""

    def __init__(self, tasks: list[Task], original_request: str) -> None:
        self.tasks = tasks
        self.original_request = original_request
        self._current_idx: int = -1

    @property
    def current(self) -> Task | None:
        """Tarea actual siendo ejecutada."""
        if 0 <= self._current_idx < len(self.tasks):
            return self.tasks[self._current_idx]
        return None

    @property
    def progress(self) -> tuple[int, int]:
        """(completadas, total)."""
        done = sum(1 for t in self.tasks if t.status in ("done", "skipped"))
        return done, len(self.tasks)

    @property
    def is_complete(self) -> bool:
        """True si todas las tareas estan terminadas."""
        return all(
            t.status in ("done", "skipped", "error")
            for t in self.tasks
        )

    def advance(self) -> Task | None:
        """Avanza a la siguiente tarea pendiente. Retorna la tarea o None."""
        for i, t in enumerate(self.tasks):
            if t.status == "pending":
                t.status = "in_progress"
                self._current_idx = i
                return t
        return None

    def mark_done(self, result: str = "") -> None:
        """Marca la tarea actual como completada."""
        if self.current:
            self.current.status = "done"
            self.current.result = result

    def mark_error(self, result: str = "") -> None:
        """Marca la tarea actual como error."""
        if self.current:
            self.current.status = "error"
            self.current.result = result

    def skip_remaining(self, reason: str = "") -> None:
        """Salta todas las tareas pendientes."""
        for t in self.tasks:
            if t.status == "pending":
                t.status = "skipped"
                t.result = reason

    def summary(self) -> str:
        """Genera un resumen textual del plan ejecutado."""
        lines = []
        for t in self.tasks:
            status_icon = {
                "done": "✓",
                "error": "✗",
                "skipped": "○",
                "in_progress": "…",
                "pending": " ",
            }.get(t.status, "?")
            lines.append(f"  {status_icon} T{t.number}. {t.title}")
            if t.result and t.status == "error":
                lines.append(f"      Error: {t.result[:100]}")
        return "\n".join(lines)


class TaskPlanner:
    """Planifica tareas para solicitudes complejas.

    Flujo:
    1. Recibe la solicitud del usuario
    2. Clasifica si necesita planificacion (LLM ligero)
    3. Si si: genera plan de max 7 tareas
    4. Muestra plan y pide confirmacion
    5. Ejecuta tareas secuencialmente
    6. Reporta resultados
    """

    # Prompt para clasificar si la solicitud necesita planificacion
    CLASSIFY_PROMPT = """Clasifica si la siguiente solicitud de un usuario necesita ser dividida en multiples tareas secuenciales para ser completada por un agente de codigo.

La solicitud necesita planificacion si:
- Involucra crear o modificar multiples archivos
- Requiere varios pasos secuenciales (instalar deps, crear config, escribir codigo, probar)
- Es un proyecto o feature completa, no una pregunta simple
- Dice cosas como "crea una app", "implementa un sistema", "refactoriza todo el modulo"

NO necesita planificacion si:
- Es una pregunta simple o conversacional
- Pide un solo cambio pequeno
- Es una consulta de informacion
- Pide explicar algo
- Es una sola edicion de archivo

Responde SOLO con "yes" o "no".

Solicitud: {request}"""

    # Prompt para generar el plan de tareas
    PLAN_PROMPT = """Eres un planificador de tareas para un agente de codigo. Dada la siguiente solicitud del usuario, genera un plan de tareas secuenciales para completarla.

REGLAS IMPORTANTES:
- Maximo {max_tasks} tareas
- Cada tarea debe ser una accion concreta y ejecutable
- Las tareas deben ser secuenciales (cada una depende de la anterior)
- Incluye el contexto necesario en cada tarea para que sea auto-contenida
- Usa el mismo idioma que la solicitud
- No incluyas tareas triviales como "verificar" o "confirmar"

Responde SOLO con un JSON valido con este formato exacto:
{{"tasks": [
  {{"title": "Titulo corto", "description": "Descripcion detallada de que hacer"}},
  {{"title": "Titulo corto", "description": "Descripcion detallada de que hacer"}}
]}}

Solicitud: {request}

Directorio de trabajo: {cwd}"""

    def __init__(self, api_client: Any, config: Any = None) -> None:
        self._api = api_client
        self._max_tasks = MAX_TASKS
        if config and hasattr(config, 'planner') and hasattr(config.planner, 'max_tasks'):
            self._max_tasks = min(config.planner.max_tasks, MAX_TASKS)

    def should_plan(self, user_input: str) -> bool:
        """Determina si la solicitud necesita planificacion.

        Usa una llamada LLM ligera (sin herramientas) para clasificar.
        Si la API falla, retorna False (modo seguro: no planificar).
        """
        # Heuristica rapida primero: si el mensaje es muy corto, probablemente no
        if len(user_input.strip()) < 15:
            return False

        # Palabras clave que tipicamente indican tareas simples
        simple_patterns = [
            r'^que es', r'^que son', r'^que significa',
            r'^explica', r'^como funciona', r'^cual es la diferencia',
            r'^ayuda', r'^help',
        ]
        for pat in simple_patterns:
            if re.search(pat, user_input, re.IGNORECASE):
                return False

        # Palabras clave que tipicamente indican tareas complejas
        complex_patterns = [
            r'crea(?:r)?\s+(?:una |un )?(?:app|aplicacion|sistema|modulo|proyecto|api|bot|plugin)',
            r'implementa(?:r)?\s+(?:un |una )?(?:sistema|modulo|feature|funcionalidad)',
            r'refactoriza',
            r'migra(?:r)?\s+(?:el |la )?(?:codigo|proyecto|base de datos)',
            r'agrega(?:r)?\s+(?:todas?|varias?|multiples)\s+funcionalidades',
            r'construye',
            r'desarrolla',
            r'convierte\s+(?:todo|el proyecto)',
        ]
        for pat in complex_patterns:
            if re.search(pat, user_input, re.IGNORECASE):
                return True

        # Si la heuristica no decide, preguntar al LLM
        try:
            prompt = self.CLASSIFY_PROMPT.format(request=user_input)
            messages = [{"role": "user", "content": prompt}]
            response = self._api.chat_completion(
                messages=messages,
                tools=None,
                temperature=0.0,
                max_tokens=10,
            )
            content = (response.get("choices", [{}])[0]
                      .get("message", {})
                      .get("content", "")
                      .strip().lower())
            return "yes" in content
        except Exception:
            return False

    def create_plan(self, user_input: str, cwd: str = ".") -> TaskPlan | None:
        """Genera un plan de tareas a partir de la solicitud del usuario.

        Returns:
            TaskPlan o None si falla la generacion.
        """
        try:
            prompt = self.PLAN_PROMPT.format(
                request=user_input,
                max_tasks=self._max_tasks,
                cwd=cwd,
            )
            messages = [{"role": "user", "content": prompt}]
            response = self._api.chat_completion(
                messages=messages,
                tools=None,
                temperature=0.3,
                max_tokens=2000,
            )
            content = (response.get("choices", [{}])[0]
                      .get("message", {})
                      .get("content", ""))

            # Extraer JSON de la respuesta
            tasks = self._parse_tasks(content)
            if not tasks:
                return None

            # Limitar a max_tasks
            tasks = tasks[:self._max_tasks]

            # Crear objetos Task
            task_objects = []
            for i, t in enumerate(tasks, 1):
                task_objects.append(Task(
                    number=i,
                    title=t.get("title", f"Tarea {i}"),
                    description=t.get("description", ""),
                ))

            return TaskPlan(task_objects, user_input)
        except Exception:
            return None

    def _parse_tasks(self, content: str) -> list[dict[str, str]]:
        """Parsea la respuesta del LLM para extraer la lista de tareas."""
        # Intentar extraer JSON de la respuesta
        # El LLM puede rodear el JSON con markdown o texto

        # 1. Buscar bloque JSON
        json_match = re.search(r'\{[\s\S]*"tasks"[\s\S]*\}', content)
        if json_match:
            try:
                data = json.loads(json_match.group())
                tasks = data.get("tasks", [])
                if isinstance(tasks, list) and len(tasks) > 0:
                    valid = []
                    for t in tasks:
                        if isinstance(t, dict) and "title" in t:
                            valid.append(t)
                    if valid:
                        return valid
            except json.JSONDecodeError:
                pass

        # 2. Intentar parsear todo como JSON
        try:
            data = json.loads(content.strip())
            tasks = data.get("tasks", [])
            if isinstance(tasks, list) and len(tasks) > 0:
                valid = []
                for t in tasks:
                    if isinstance(t, dict) and "title" in t:
                        valid.append(t)
                if valid:
                    return valid
        except json.JSONDecodeError:
            pass

        return []
