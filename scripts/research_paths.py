"""Rutas externas configurables para los pipelines científicos.

Los datos grandes viven fuera de Git. Cada ruta admite override por variable de
entorno; el fallback es siempre relativo a la raíz del repo (nunca un árbol local
del autor), para que los scripts depositados en el paquete de replicación corran
igual en cualquier máquina.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def configured_path(env_name: str, *candidates: Path) -> Path:
    """Devuelve el override o el primer candidato existente.

    Si ninguno existe, devuelve el primer candidato para que el consumidor emita
    un error accionable con una ruta concreta. No lee archivos ni credenciales.
    """

    override = os.environ.get(env_name)
    if override:
        return Path(override).expanduser().resolve()
    if not candidates:
        raise ValueError(f"{env_name}: se requiere al menos una ruta candidata")
    expanded = [path.expanduser() for path in candidates]
    return next((path for path in expanded if path.exists()), expanded[0])


#: Volcado crudo de denuncias de Lima (export MININTER/SIDPOL, no redistribuible).
#: Override: INFELIX_LIMA_DENUNCIAS (o INFELIX_LIMA_PARQUET, el nombre que usa
#: eval_selective_degradation.py); si no, rutas relativas al repo (gitignored).
LIMA_DENUNCIAS = configured_path(
    "INFELIX_LIMA_DENUNCIAS",
    *([Path(os.environ["INFELIX_LIMA_PARQUET"])] if os.environ.get("INFELIX_LIMA_PARQUET") else []),
    ROOT / "data/raw/external/LIMA.parquet",
    ROOT / "data/datasets/bronze/mininter/LIMA.parquet",
)
