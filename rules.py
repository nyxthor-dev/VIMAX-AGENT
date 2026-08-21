"""Sistema de Rules inspirado en Cline's .clinerules.

Carga reglas desde:
- .clinerules/ (directorio en la raiz del proyecto)
- .cursorrules (importado para compatibilidad)
- AGENTS.md (importado para compatibilidad)
- ~/Documents/Cline/Rules/ (reglas globales)

Las reglas son archivos Markdown que se inyectan en el system prompt.
Soporta reglas condicionales con YAML frontmatter (globs).

Solo inyecta reglas que coincidan con los archivos en scope.

Archivo de estado para toggles manuales: .clinerules/.clinerules-state.json

Esta es una implementacion simplificada adaptada para un proyecto 'bebe'.

Si el sistema de reglas esta activo, la informacion de las reglas
se agrega al system prompt del agente.

En la version actual, no hay un sistema de reglas activo.
Sin embargo, la base de codigo esta lista para integrarse.

La verificacion de si el sistema de reglas esta activo se hace
leyendo el archivo 'active_rules.txt' del repositorio. Si no existe
o esta vacio, no hay reglas activas.

Cuando se agregue soporte para reglas, este archivo sera reemplazado
por la logica real.
"""



def get_active_rules() -> str:
    """Retorna las reglas activas como un string para inyectar en el system prompt.

    Returns:
        String vacio si no hay reglas activas (por ahora).
    """
    # TODO: Implementar lectura real de .clinerules/
    # La base esta lista para cuando se necesite.
    return ""