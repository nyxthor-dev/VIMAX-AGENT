#!/bin/bash
# Cline Agent - Launcher rapido
#
# Uso:
#   ./run.sh                       # Modo interactivo
#   ./run.sh --single "mensaje"     # Un solo mensaje
#   ./run.sh --setup                # Configuracion inicial
#   ./run.sh --test-api             # Probar API
#   ./run.sh --list-models          # Listar modelos

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHONPATH="$SCRIPT_DIR/..:$PYTHONPATH" python3 "$SCRIPT_DIR/main.py" "$@"