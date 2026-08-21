"""Herramientas interactivas del agente: ask_followup_question y attempt_completion.

Estas herramientas permiten al agente interactuar con el usuario de forma
estructurada, igual que Cline: pedir aclaraciones y presentar resultados.
"""

from __future__ import annotations

from typing import Any, Callable

from .base import BaseTool, ToolParameter, ToolResult


class AskFollowupQuestionTool(BaseTool):
    """Permite al agente hacer una pregunta al usuario y esperar su respuesta.

    Inspirada en la herramienta ask_followup_question de Cline. El agente puede
    usar esta herramienta cuando necesita mas informacion o aclaracion del
    usuario antes de continuar con una tarea.
    """

    name = "ask_followup_question"
    description = (
        "Pregunta al usuario una pregunta de seguimiento y espera su respuesta. "
        "Usa esta herramienta cuando necesites aclaracion, confirmacion, o cuando "
        "haya multiples formas de abordar una tarea y no estes seguro de la preferencia "
        "del usuario. La pregunta debe ser clara y especifica."
    )
    parameters = [
        ToolParameter(
            "question",
            "string",
            "La pregunta que quieres hacerle al usuario. Debe ser clara y concisa.",
            required=True,
        ),
    ]

    def __init__(self, input_callback: Callable[[str], str]) -> None:
        """Inicializa con un callback que recibe la pregunta y retorna la respuesta del usuario."""
        super().__init__()
        self._input_callback = input_callback

    def execute(self, **kwargs: Any) -> ToolResult:
        question = kwargs["question"]
        try:
            response = self._input_callback(question)
            if response.strip():
                return ToolResult(
                    True,
                    output=f"Respuesta del usuario: {response.strip()}",
                )
            else:
                return ToolResult(
                    True,
                    output="El usuario no proporciono una respuesta. Continua con tu mejor juicio.",
                )
        except (EOFError, KeyboardInterrupt):
            return ToolResult(False, error="El usuario interrumpio la entrada.")


class AttemptCompletionTool(BaseTool):
    """Permite al agente presentar el resultado final de una tarea.

    Inspirada en la herramienta attempt_completion de Cline. El agente usa esta
    herramienta cuando ha completado una tarea y quiere presentar el resultado
    al usuario con un resumen de lo que se hizo.
    """

    name = "attempt_completion"
    description = (
        "Presenta el resultado final de una tarea al usuario. Usa esta herramienta "
        "cuando hayas completado satisfactoriamente la solicitud del usuario. "
        "Incluye un resumen claro de lo que se hizo, los archivos modificados/creados, "
        "y cualquier instruccion para verificar o probar los cambios. "
        "Esto debe ser el ULTIMO paso de cualquier tarea completada."
    )
    parameters = [
        ToolParameter(
            "result",
            "string",
            "Descripcion del resultado completado. Debe incluir: que se hizo, "
            "archivos modificados o creados, y como verificar/probar los cambios.",
            required=True,
        ),
    ]

    def execute(self, **kwargs: Any) -> ToolResult:
        result_text = kwargs["result"]
        # Marca especial para que el agente sepa que la tarea fue completada
        return ToolResult(
            True,
            output=f"TAREA COMPLETADA\n\n{result_text}",
        )
