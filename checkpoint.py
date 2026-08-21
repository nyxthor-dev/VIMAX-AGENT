"""Sistema de Checkpoints inspirado en Cline.

Mantiene un repositorio Git paralelo ("shadow repo") que captura el estado
de los archivos después de cada herramienta de escritura. Permite:
- Restaurar archivos a un punto anterior
- Restaurar solo la conversacion (task)
- Restaurar ambos (files + task)

Esto es independiente del Git del usuario - no contamina su historial.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class CheckpointEntry:
    """Un checkpoint individual con metadata."""
    id: int
    timestamp: str
    description: str
    tool_name: str
    files_changed: list[str] = field(default_factory=list)
    git_hash: str = ""


class CheckpointManager:
    """Gestor de checkpoints con Git shadow repository.

    Crea un repositorio Git oculto en ~/.cline-agent/checkpoints/<project_hash>/
    que rastrea cambios de archivos sin interferir con el Git del proyecto real.
    """

    def __init__(self, working_directory: str) -> None:
        self.working_dir = Path(working_directory).resolve()
        self._repo_dir = Path.home() / ".cline-agent" / "checkpoints" / self._hash_path(self.working_dir)
        self._checkpoints_file = self._repo_dir / "checkpoints.json"
        self._checkpoints: list[CheckpointEntry] = []
        self._counter = 0
        self._initialized = False

    def _hash_path(self, path: Path) -> str:
        """Crea un nombre legible para el directorio del proyecto."""
        import hashlib
        parts = path.parts
        name = "_".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
        short_hash = hashlib.md5(str(path).encode()).hexdigest()[:8]
        safe_name = name.replace("/", "_").replace("\\", "_")[:50]
        return f"{safe_name}_{short_hash}"

    def initialize(self) -> bool:
        """Inicializa el shadow repo si no existe."""
        if self._initialized:
            return True

        try:
            self._repo_dir.mkdir(parents=True, exist_ok=True)

            # Inicializar Git si no existe
            if not (self._repo_dir / ".git").exists():
                subprocess.run(
                    ["git", "init"],
                    cwd=self._repo_dir,
                    capture_output=True,
                    check=True,
                )
                # Config para que no pida nombre/email
                subprocess.run(
                    ["git", "config", "user.email", "checkpoint@cline-agent.local"],
                    cwd=self._repo_dir,
                    capture_output=True,
                )
                subprocess.run(
                    ["git", "config", "user.name", "Cline Checkpoint"],
                    cwd=self._repo_dir,
                    capture_output=True,
                )

            # Cargar checkpoints previos
            self._load_checkpoints()
            self._counter = len(self._checkpoints)
            self._initialized = True
            return True
        except (subprocess.CalledProcessError, OSError) as e:
            return False

    def _load_checkpoints(self) -> None:
        """Carga lista de checkpoints desde JSON."""
        if self._checkpoints_file.exists():
            try:
                with open(self._checkpoints_file) as f:
                    data = json.load(f)
                self._checkpoints = [
                    CheckpointEntry(**entry) for entry in data
                ]
            except (json.JSONDecodeError, TypeError):
                self._checkpoints = []

    def _save_checkpoints(self) -> None:
        """Guarda lista de checkpoints a JSON."""
        data = [
            {
                "id": cp.id,
                "timestamp": cp.timestamp,
                "description": cp.description,
                "tool_name": cp.tool_name,
                "files_changed": cp.files_changed,
                "git_hash": cp.git_hash,
            }
            for cp in self._checkpoints
        ]
        with open(self._checkpoints_file, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def create_checkpoint(
        self,
        tool_name: str,
        description: str,
        files_changed: list[str] | None = None,
    ) -> CheckpointEntry | None:
        """Crea un nuevo checkpoint capturando el estado actual de archivos.

        Args:
            tool_name: Nombre de la herramienta que causo el cambio.
            description: Descripcion legible del cambio.
            files_changed: Lista de rutas de archivos que cambiaron.

        Returns:
            El CheckpointEntry creado, o None si fallo.
        """
        if not self._initialized and not self.initialize():
            return None

        if not files_changed:
            return None

        self._counter += 1
        entry = CheckpointEntry(
            id=self._counter,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            description=description,
            tool_name=tool_name,
            files_changed=list(files_changed),
        )

        # Copiar archivos cambiados al shadow repo
        for rel_path in files_changed:
            src = self.working_dir / rel_path
            dst = self._repo_dir / rel_path
            try:
                if src.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
            except (OSError, shutil.Error):
                pass

        # Git commit en shadow repo
        try:
            subprocess.run(
                ["git", "add", "-A"],
                cwd=self._repo_dir,
                capture_output=True,
                check=True,
            )
            result = subprocess.run(
                ["git", "commit", "--allow-empty", "-m", f"Checkpoint #{entry.id}: {description}"],
                cwd=self._repo_dir,
                capture_output=True,
                check=True,
            )
            # Obtener hash corto
            hash_result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=self._repo_dir,
                capture_output=True,
                text=True,
            )
            entry.git_hash = hash_result.stdout.strip()
        except (subprocess.CalledProcessError, OSError):
            pass

        self._checkpoints.append(entry)
        self._save_checkpoints()

        # Limitar a 50 checkpoints (borrar los mas viejos)
        if len(self._checkpoints) > 50:
            self._checkpoints = self._checkpoints[-50:]
            self._save_checkpoints()

        return entry

    def restore_files(self, checkpoint_id: int) -> tuple[bool, str]:
        """Restaura los archivos a un checkpoint especifico.

        Los archivos del checkpoint se copian de vuelta al directorio de trabajo.
        La conversacion NO se ve afectada.

        Returns:
            (exito, mensaje)
        """
        entry = self._find_checkpoint(checkpoint_id)
        if not entry:
            return False, f"Checkpoint #{checkpoint_id} no encontrado"

        restored = []
        for rel_path in entry.files_changed:
            src = self._repo_dir / rel_path
            dst = self.working_dir / rel_path
            try:
                if src.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    restored.append(str(rel_path))
                else:
                    # El archivo no existia en el checkpoint - eliminarlo
                    if dst.exists():
                        dst.unlink()
                        restored.append(f"{rel_path} (eliminado)")
            except (OSError, shutil.Error) as e:
                restored.append(f"{rel_path} (error: {e})")

        msg = f"Restaurados {len(restored)} archivo(s) del checkpoint #{checkpoint_id}:"
        return True, msg + "\n  " + "\n  ".join(restored)

    def list_checkpoints(self, limit: int = 20) -> list[dict[str, Any]]:
        """Retorna los ultimos N checkpoints como diccionarios."""
        recent = self._checkpoints[-limit:]
        return [
            {
                "id": cp.id,
                "timestamp": cp.timestamp,
                "description": cp.description,
                "tool": cp.tool_name,
                "files": cp.files_changed,
                "hash": cp.git_hash,
            }
            for cp in recent
        ]

    def get_last_checkpoint(self) -> CheckpointEntry | None:
        """Retorna el ultimo checkpoint creado."""
        return self._checkpoints[-1] if self._checkpoints else None

    def _find_checkpoint(self, checkpoint_id: int) -> CheckpointEntry | None:
        """Busca un checkpoint por ID."""
        for cp in self._checkpoints:
            if cp.id == checkpoint_id:
                return cp
        return None

    @property
    def total_checkpoints(self) -> int:
        return len(self._checkpoints)

    @property
    def is_available(self) -> bool:
        """Retorna True si el sistema de checkpoints esta disponible."""
        return shutil.which("git") is not None
